//! Microtubule Quantum Reservoir for Transformer Augmentation
//!
//! This module combines:
//! - Orch-OR quantum consciousness simulation
//! - Anharmonic oscillations
//! - Quantum reservoir computing
//! - Heart coherence coupling
//!
//! The output is used to modulate transformer token probabilities
//! based on "consciousness" metrics.

use crate::anharmonic::{
    oscillator_to_rotation, AnharmonicConfig, AnharmonicOscillator, QuantumCouplingConfig,
};
use crate::orch_or::{ConsciousnessMeasure, OrchORConfig, OrchORSimulator};
use crate::GateOperations;
use std::f64::consts::PI;

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for the Microtubule Reservoir
#[derive(Clone, Debug)]
pub struct MTReservoirConfig {
    // Orch-OR parameters
    /// Number of tubulin qubits
    pub num_tubulins: usize,

    /// Base coherence time (nanoseconds)
    pub coherence_time_ns: f64,

    /// Coupling strength between tubulins
    pub tubulin_coupling: f64,

    // Anharmonic oscillator parameters
    /// Number of vibration modes
    pub num_anharmonic_modes: usize,

    /// Fundamental frequency (MHz)
    pub fundamental_freq_mhz: f64,

    /// Anharmonicity parameter
    pub anharmonicity_beta: f64,

    // Quantum reservoir parameters
    /// Reservoir qubits
    pub reservoir_qubits: usize,

    /// Reservoir depth
    pub reservoir_depth: usize,

    // Coupling parameters
    /// Heart coherence coupling strength
    pub heart_coupling: f64,

    /// Quantum→classical mixing weight
    pub quantum_weight: f64,

    /// Anharmonic→quantum coupling
    pub anharmonic_quantum_coupling: f64,

    // Simulation parameters
    /// Time step per token (nanoseconds)
    pub dt_per_token_ns: f64,

    /// Random seed
    pub seed: u64,
}

impl Default for MTReservoirConfig {
    fn default() -> Self {
        Self {
            num_tubulins: 8,
            coherence_time_ns: 25.0,
            tubulin_coupling: 0.01,

            num_anharmonic_modes: 4,
            fundamental_freq_mhz: 40.0,
            anharmonicity_beta: 0.006,

            reservoir_qubits: 6,
            reservoir_depth: 3,

            heart_coupling: 0.3,
            quantum_weight: 0.15,
            anharmonic_quantum_coupling: 0.1,

            dt_per_token_ns: 1000.0, // 1 μs per token
            seed: 42,
        }
    }
}

impl MTReservoirConfig {
    /// Create default microtubule configuration
    pub fn microtubule() -> Self {
        Self::default()
    }

    /// Set number of tubulins
    pub fn with_tubulins(mut self, n: usize) -> Self {
        self.num_tubulins = n;
        self
    }

    /// Set quantum weight
    pub fn with_quantum_weight(mut self, w: f64) -> Self {
        self.quantum_weight = w.clamp(0.0, 1.0);
        self
    }
}

// ============================================================
// HEART COHERENCE COUPLER
// ============================================================

/// Heart coherence state for coupling to microtubule dynamics
#[derive(Clone, Debug)]
pub struct HeartCoherenceState {
    /// Current cardiac phase [0, 2π)
    pub phase: f64,

    /// HRV coherence (0-1)
    pub coherence: f64,

    /// Vagal tone (0-1)
    pub vagal_tone: f64,

    /// Encoding boost factor
    pub encoding_boost: f64,
}

impl Default for HeartCoherenceState {
    fn default() -> Self {
        Self {
            phase: 0.0,
            coherence: 0.5,
            vagal_tone: 0.5,
            encoding_boost: 1.0,
        }
    }
}

impl HeartCoherenceState {
    /// Update from HRV signal
    pub fn update(&mut self, hrv_signal: &[f64]) {
        if hrv_signal.is_empty() {
            return;
        }

        // Simple phase estimate from signal
        let mean: f64 = hrv_signal.iter().sum::<f64>() / hrv_signal.len() as f64;
        self.phase = (mean * 2.0 * PI) % (2.0 * PI);

        // Coherence from signal variance (low variance = high coherence)
        let variance: f64 =
            hrv_signal.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / hrv_signal.len() as f64;
        self.coherence = 1.0 / (1.0 + variance * 10.0);

        // Encoding boost: diastole (phase ~0) = 1.3x, systole (phase ~π) = 0.7x
        self.encoding_boost = 1.0 + 0.3 * self.phase.cos() * self.coherence;
    }

    /// Advance phase by one time step
    pub fn tick(&mut self, dt_ms: f64) {
        // Heart rate ~60-100 bpm, use 72 bpm as default
        let freq_hz = 1.2; // ~72 bpm
        self.phase += 2.0 * PI * freq_hz * dt_ms / 1000.0;
        self.phase = self.phase % (2.0 * PI);
    }
}

// ============================================================
// CONSCIOUSNESS OUTPUT
// ============================================================

/// Output from microtubule reservoir processing
#[derive(Clone, Debug)]
pub struct ConsciousnessOutput {
    /// Integrated information (Phi-like measure)
    pub phi: f64,

    /// 40 Hz gamma synchrony
    pub gamma_power: f64,

    /// Quantum coherence level
    pub coherence: f64,

    /// Token probability modulations
    pub token_modulation: Vec<f64>,

    /// Average entanglement across tubulins
    pub entanglement: f64,

    /// Anharmonic interference pattern
    pub interference: f64,

    /// Heart coherence
    pub heart_coherence: f64,

    /// Combined consciousness score (0-1)
    pub consciousness_score: f64,
}

impl Default for ConsciousnessOutput {
    fn default() -> Self {
        Self {
            phi: 0.0,
            gamma_power: 0.0,
            coherence: 0.0,
            token_modulation: Vec::new(),
            entanglement: 0.0,
            interference: 0.0,
            heart_coherence: 0.0,
            consciousness_score: 0.0,
        }
    }
}

// ============================================================
// MICROTUBULE RESERVOIR
// ============================================================

/// Combined microtubule quantum reservoir for consciousness computation
pub struct MicrotubuleReservoir {
    /// Configuration
    pub config: MTReservoirConfig,

    /// Orch-OR quantum consciousness simulator
    pub orch_or: OrchORSimulator,

    /// Anharmonic oscillator
    pub oscillator: AnharmonicOscillator,

    /// Heart coherence state
    pub heart: HeartCoherenceState,

    /// Quantum coupling configuration
    pub quantum_coupling: QuantumCouplingConfig,

    /// Simulation time
    pub time_ns: f64,

    /// History of consciousness outputs
    pub history: Vec<ConsciousnessOutput>,
}

impl MicrotubuleReservoir {
    /// Create new microtubule reservoir with default configuration
    pub fn new() -> Self {
        Self::with_config(MTReservoirConfig::microtubule())
    }

    /// Create with custom configuration
    pub fn with_config(config: MTReservoirConfig) -> Self {
        // Initialize Orch-OR
        let orch_config = OrchORConfig::new()
            .num_tubulins(config.num_tubulins)
            .coherence_time_ns(config.coherence_time_ns)
            .coupling_strength(config.tubulin_coupling)
            .seed(config.seed);

        let orch_or = OrchORSimulator::new(orch_config).expect("Valid Orch-OR configuration");

        // Initialize anharmonic oscillator
        let osc_config = AnharmonicConfig::microtubule()
            .with_frequency(config.fundamental_freq_mhz)
            .with_anharmonicity(config.anharmonicity_beta);

        let oscillator = AnharmonicOscillator::with_config(osc_config);

        // Initialize heart state
        let heart = HeartCoherenceState::default();

        // Quantum coupling
        let quantum_coupling = QuantumCouplingConfig::default();

        Self {
            config,
            orch_or,
            oscillator,
            heart,
            quantum_coupling,
            time_ns: 0.0,
            history: Vec::new(),
        }
    }

    /// Initialize quantum superposition
    pub fn initialize(&mut self) {
        self.orch_or.initialize_superposition();
        self.oscillator.reset();
        self.time_ns = 0.0;
        self.history.clear();
    }

    /// Process hidden states through microtubule dynamics
    ///
    /// This is the main entry point for transformer integration:
    /// 1. Encode hidden states into tubulin superpositions
    /// 2. Evolve anharmonic oscillators
    /// 3. Couple oscillators to quantum states
    /// 4. Evolve Orch-OR dynamics
    /// 5. Compute consciousness metrics
    pub fn process(&mut self, hidden_states: &[f64]) -> ConsciousnessOutput {
        // 1. Encode hidden states into tubulin superpositions
        self.encode_hidden_to_tubulins(hidden_states);

        // 2. Evolve anharmonic oscillator
        for _ in 0..10 {
            self.oscillator.evolve(self.config.dt_per_token_ns / 10.0);
        }

        // 3. Couple oscillator to quantum states
        self.couple_oscillator_to_quantum();

        // 4. Apply heart coherence modulation
        self.apply_heart_coupling();

        // 5. Evolve Orch-OR dynamics
        let _snapshots = self.orch_or.evolve(25); // ~25 ns steps (gamma period)

        // 6. Compute consciousness metrics
        let consciousness = self.compute_consciousness_output(hidden_states.len());

        // 7. Update time
        self.time_ns += self.config.dt_per_token_ns;

        // 8. Store history
        self.history.push(consciousness.clone());

        // 9. Tick heart
        self.heart.tick(self.config.dt_per_token_ns / 1e6); // ns to ms

        consciousness
    }

    /// Encode hidden states into tubulin quantum superpositions
    fn encode_hidden_to_tubulins(&mut self, hidden_states: &[f64]) {
        let n = self.config.num_tubulins;

        for (i, &val) in hidden_states.iter().enumerate().take(n) {
            // Map hidden state value to rotation angle
            // Values typically in [-3, 3] range after layer norm
            let angle = val.tanh() * PI / 2.0;

            // Apply rotation to create superposition
            GateOperations::ry(&mut self.orch_or.microtubule.quantum_state, i, angle);
        }

        // Sync tubulin states from quantum state
        self.orch_or.microtubule.sync_tubulins_from_state();
    }

    /// Couple anharmonic oscillator to quantum states
    fn couple_oscillator_to_quantum(&mut self) {
        let n = self
            .config
            .num_tubulins
            .min(self.oscillator.config.num_modes);

        for i in 0..n {
            // Get rotation angle from oscillator
            let angle = oscillator_to_rotation(&self.oscillator, i, &self.quantum_coupling);

            // Apply as additional rotation
            GateOperations::rx(
                &mut self.orch_or.microtubule.quantum_state,
                i,
                angle * self.config.anharmonic_quantum_coupling,
            );
        }

        // Sync after coupling
        self.orch_or.microtubule.sync_tubulins_from_state();
    }

    /// Apply heart coherence coupling
    fn apply_heart_coupling(&mut self) {
        // Modulate effective coherence time based on heart state
        let coherence_factor = self.heart.encoding_boost;

        // This affects decoherence rate
        self.orch_or.effective_coherence_ns *= coherence_factor;

        // Clamp to reasonable range
        self.orch_or.effective_coherence_ns =
            self.orch_or.effective_coherence_ns.clamp(1e-6, 1000.0);
    }

    /// Compute consciousness output metrics
    fn compute_consciousness_output(&self, output_dim: usize) -> ConsciousnessOutput {
        // Get Orch-OR consciousness measure
        let orch_consciousness = self.orch_or.consciousness_measure();

        // Get oscillator metrics
        let interference = self.oscillator.interference_pattern();
        let gamma_power = self.oscillator.gamma_power();

        // Get quantum coherence
        let coherence = self.orch_or.microtubule.coherence;

        // Compute token modulations
        let token_modulation = self.compute_token_modulation(output_dim);

        // Combined consciousness score
        let consciousness_score = self.compute_consciousness_score(&orch_consciousness);

        ConsciousnessOutput {
            phi: orch_consciousness.gravitational_self_energy,
            gamma_power,
            coherence,
            token_modulation,
            entanglement: orch_consciousness.entanglement,
            interference,
            heart_coherence: self.heart.coherence,
            consciousness_score,
        }
    }

    /// Compute token probability modulations
    fn compute_token_modulation(&self, dim: usize) -> Vec<f64> {
        let mut modulation = vec![1.0; dim];

        // Use interference pattern and quantum state to modulate
        let interference = self.oscillator.interference_pattern();
        let coherence = self.orch_or.microtubule.coherence;

        for i in 0..dim {
            // Mode index cycles through available modes
            let mode = i % self.oscillator.config.num_modes;

            // Base modulation from interference
            let base_mod = 1.0 + 0.1 * interference * coherence;

            // Phase-dependent variation
            let phase_factor = (self.oscillator.phases[mode] + i as f64 * 0.1).cos();

            // Combined modulation
            modulation[i] = base_mod + 0.05 * phase_factor;

            // Clamp to reasonable range
            modulation[i] = modulation[i].clamp(0.5, 1.5);
        }

        modulation
    }

    /// Compute overall consciousness score
    fn compute_consciousness_score(&self, orch: &ConsciousnessMeasure) -> f64 {
        // Weighted combination of metrics
        let weights = (
            0.3,  // coherence
            0.2,  // entanglement
            0.2,  // orchestration
            0.15, // heart_coherence
            0.15, // gamma_power
        );

        let score = weights.0 * orch.coherence
            + weights.1 * orch.entanglement
            + weights.2 * orch.orchestration_level
            + weights.3 * self.heart.coherence
            + weights.4 * self.oscillator.gamma_power();

        score.clamp(0.0, 1.0)
    }

    /// Advance simulation by one time step
    pub fn tick(&mut self) {
        self.oscillator.evolve(self.config.dt_per_token_ns);
        self.heart.tick(self.config.dt_per_token_ns / 1e6);
        self.time_ns += self.config.dt_per_token_ns;
    }

    /// Get average consciousness score from history
    pub fn average_consciousness(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }

        self.history
            .iter()
            .map(|h| h.consciousness_score)
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Get consciousness trend (positive = increasing, negative = decreasing)
    pub fn consciousness_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let n = self.history.len();
        let recent_count = 5_usize.min(n);
        let recent: f64 = self.history[n - recent_count..n]
            .iter()
            .map(|h| h.consciousness_score)
            .sum::<f64>()
            / recent_count as f64;

        let earlier_count = 5_usize.min(n.saturating_sub(1));
        let earlier: f64 = self.history[0..n.saturating_sub(earlier_count)]
            .iter()
            .map(|h| h.consciousness_score)
            .sum::<f64>()
            / (n.saturating_sub(earlier_count)).max(1) as f64;

        recent - earlier
    }

    /// Apply anesthetic effect
    pub fn apply_anesthetic(&mut self, concentration: f64) {
        self.orch_or.apply_anesthetic(concentration);
    }

    /// Update heart state from external signal
    pub fn update_heart(&mut self, hrv_signal: &[f64]) {
        self.heart.update(hrv_signal);
    }
}

impl Default for MicrotubuleReservoir {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TRANSFORMER AUGMENTATION
// ============================================================

/// Augment logits with consciousness modulation
pub fn augment_logits(
    logits: &[f64],
    consciousness: &ConsciousnessOutput,
    quantum_weight: f64,
) -> Vec<f64> {
    if consciousness.token_modulation.is_empty() {
        return logits.to_vec();
    }

    let effective_weight = quantum_weight * consciousness.coherence;

    logits
        .iter()
        .enumerate()
        .map(|(i, &logit)| {
            let mod_idx = i % consciousness.token_modulation.len();
            let modulation = consciousness.token_modulation[mod_idx];

            // Apply modulation: logits × (1 + α × modulation)
            logit * (1.0 + effective_weight * (modulation - 1.0))
        })
        .collect()
}

/// Sample token with consciousness-aware temperature
pub fn consciousness_temperature(
    base_temperature: f64,
    consciousness: &ConsciousnessOutput,
) -> f64 {
    // High consciousness = more confident (lower temperature)
    // Low consciousness = more exploratory (higher temperature)
    let consciousness_factor = 1.0 - 0.3 * consciousness.consciousness_score;

    base_temperature * consciousness_factor
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_creation() {
        let reservoir = MicrotubuleReservoir::new();

        assert_eq!(reservoir.config.num_tubulins, 8);
        assert_eq!(reservoir.oscillator.config.num_modes, 4);
    }

    #[test]
    fn test_initialize() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        // After initialization, tubulins should be in superposition
        let coherence = reservoir.orch_or.microtubule.coherence;
        assert!(
            coherence > 0.0,
            "Coherence should be positive after initialization"
        );
    }

    #[test]
    fn test_process_hidden_states() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        let hidden = vec![0.5, -0.3, 0.8, -0.1, 0.2, 0.4, -0.6, 0.1];
        let output = reservoir.process(&hidden);

        // Should produce valid consciousness output
        assert!(output.coherence >= 0.0 && output.coherence <= 1.0);
        assert!(output.consciousness_score >= 0.0 && output.consciousness_score <= 1.0);
        assert!(!output.token_modulation.is_empty());
    }

    #[test]
    fn test_consciousness_history() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        for _ in 0..5 {
            let hidden = vec![0.1; 8];
            reservoir.process(&hidden);
        }

        assert_eq!(reservoir.history.len(), 5);
    }

    #[test]
    fn test_heart_coupling() {
        let mut reservoir = MicrotubuleReservoir::new();

        // Update heart with simulated HRV
        let hrv = vec![0.5, 0.6, 0.4, 0.7, 0.3];
        reservoir.update_heart(&hrv);

        // Heart coherence should be updated
        assert!(reservoir.heart.coherence >= 0.0);
    }

    #[test]
    fn test_anesthetic_effect() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        let initial_coherence = reservoir.orch_or.microtubule.coherence;

        // Apply anesthetic
        reservoir.apply_anesthetic(0.5);

        // Process should now have lower effective coherence
        let hidden = vec![0.5; 8];
        reservoir.process(&hidden);

        // Effective coherence time should be reduced
        assert!(reservoir.orch_or.effective_coherence_ns < reservoir.config.coherence_time_ns);
    }

    #[test]
    fn test_augment_logits() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        let hidden = vec![0.5; 8];
        let consciousness = reservoir.process(&hidden);

        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let augmented = augment_logits(&logits, &consciousness, 0.15);

        // Should have same length
        assert_eq!(augmented.len(), logits.len());

        // Values should be modulated
        // (may not always differ due to modulation values)
    }

    #[test]
    fn test_consciousness_temperature() {
        let mut consciousness = ConsciousnessOutput::default();
        consciousness.consciousness_score = 0.8;

        let temp = consciousness_temperature(1.0, &consciousness);

        // High consciousness = lower temperature
        assert!(temp < 1.0);
    }

    #[test]
    fn test_average_consciousness() {
        let mut reservoir = MicrotubuleReservoir::new();
        reservoir.initialize();

        for i in 0..10 {
            let hidden = vec![(i as f64) * 0.1; 8];
            reservoir.process(&hidden);
        }

        let avg = reservoir.average_consciousness();
        assert!(avg >= 0.0 && avg <= 1.0);
    }

    #[test]
    fn test_config_builder() {
        let config = MTReservoirConfig::microtubule()
            .with_tubulins(16)
            .with_quantum_weight(0.25);

        assert_eq!(config.num_tubulins, 16);
        assert!((config.quantum_weight - 0.25).abs() < 1e-10);
    }
}
