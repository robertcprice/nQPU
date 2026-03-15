//! Anharmonic Oscillations for Microtubule Quantum Consciousness
//!
//! **CRITICAL COMPONENT**: This implements the anharmonic vibrations that
//! Stuart Hameroff identifies as essential to consciousness:
//!
//! > "Consciousness depends on anharmonic vibrations of microtubules inside
//! > neurons, similar to certain kinds of Indian music, but unlike Western
//! > music which is harmonic."
//!
//! # Key Concepts
//!
//! **Harmonic vs Anharmonic**:
//! - Harmonic: Frequencies are integer multiples (f, 2f, 3f, ...)
//! - Anharmonic: Non-integer ratios create complex interference patterns
//!
//! # Physics
//!
//! Anharmonic frequency formula:
//! ```text
//! ω_n = n × ω_0 - β × n² × ω_0
//!
//! Where:
//! - ω_0 = fundamental frequency (~40 MHz for microtubules)
//! - n = mode number
//! - β = anharmonicity parameter (~0.006)
//! ```
//!
//! # Interference & Beats
//!
//! Anharmonic modes create evolving interference patterns that may
//! produce EEG-frequency "beats" from MHz-scale vibrations.

use std::f64::consts::PI;

// ============================================================
// CONFIGURATION
// ============================================================

/// Default fundamental frequency for microtubules (MHz)
pub const MT_FUNDAMENTAL_MHZ: f64 = 40.0;

/// Default anharmonicity parameter from experimental data
pub const MT_ANHARMONICITY_BETA: f64 = 0.006;

/// Default number of vibration modes
pub const MT_NUM_MODES: usize = 4;

// ============================================================
// ANHARMONIC OSCILLATOR
// ============================================================

/// Configuration for anharmonic oscillator
#[derive(Clone, Debug)]
pub struct AnharmonicConfig {
    /// Number of vibration modes
    pub num_modes: usize,

    /// Fundamental frequency (MHz)
    pub omega_0: f64,

    /// Anharmonicity parameters for each mode (β values)
    pub beta: Vec<f64>,

    /// Initial phases for each mode (radians)
    pub initial_phases: Vec<f64>,

    /// Initial amplitudes for each mode
    pub initial_amplitudes: Vec<f64>,

    /// Inter-mode coupling matrix
    pub coupling: Vec<Vec<f64>>,

    /// Damping rates for each mode
    pub damping: Vec<f64>,
}

impl Default for AnharmonicConfig {
    fn default() -> Self {
        let num_modes = MT_NUM_MODES;

        // Anharmonicity from experimental data (Bandyopadhyay)
        let beta = vec![0.0, 0.006, 0.006, 0.007];

        // Initial phases (can be randomized)
        let initial_phases = vec![0.0; num_modes];

        // Initial amplitudes (higher modes typically weaker)
        let initial_amplitudes = vec![1.0, 0.8, 0.6, 0.4];

        // Coupling matrix (nearest-neighbor dominant, like protofilaments)
        let coupling = vec![
            vec![0.0, 0.1, 0.02, 0.01],
            vec![0.1, 0.0, 0.1, 0.02],
            vec![0.02, 0.1, 0.0, 0.1],
            vec![0.01, 0.02, 0.1, 0.0],
        ];

        // Damping (higher modes damp faster)
        let damping = vec![0.01, 0.02, 0.03, 0.04];

        Self {
            num_modes,
            omega_0: MT_FUNDAMENTAL_MHZ,
            beta,
            initial_phases,
            initial_amplitudes,
            coupling,
            damping,
        }
    }
}

impl AnharmonicConfig {
    /// Create microtubule-specific configuration
    pub fn microtubule() -> Self {
        Self::default()
    }

    /// Create with custom fundamental frequency
    pub fn with_frequency(mut self, omega_0: f64) -> Self {
        self.omega_0 = omega_0;
        self
    }

    /// Create with custom anharmonicity
    pub fn with_anharmonicity(mut self, beta: f64) -> Self {
        self.beta = vec![0.0, beta, beta, beta * 1.1];
        self
    }

    /// Create harmonic oscillator (β = 0) for comparison
    pub fn harmonic() -> Self {
        Self {
            beta: vec![0.0; MT_NUM_MODES],
            ..Self::default()
        }
    }
}

/// Anharmonic oscillator for microtubule vibrations
///
/// Models the non-harmonic oscillations observed in microtubules,
/// which create complex interference patterns essential for
/// consciousness according to Orch-OR theory.
#[derive(Clone, Debug)]
pub struct AnharmonicOscillator {
    /// Configuration
    pub config: AnharmonicConfig,

    /// Current phase of each mode (radians)
    pub phases: Vec<f64>,

    /// Current amplitude of each mode
    pub amplitudes: Vec<f64>,

    /// Simulation time (arbitrary units, typically nanoseconds)
    pub time: f64,

    /// History of interference patterns for analysis
    interference_history: Vec<f64>,
}

impl AnharmonicOscillator {
    /// Create new anharmonic oscillator with default microtubule parameters
    pub fn new() -> Self {
        Self::with_config(AnharmonicConfig::microtubule())
    }

    /// Create with custom configuration
    pub fn with_config(config: AnharmonicConfig) -> Self {
        let phases = config.initial_phases.clone();
        let amplitudes = config.initial_amplitudes.clone();

        Self {
            config,
            phases,
            amplitudes,
            time: 0.0,
            interference_history: Vec::new(),
        }
    }

    /// Create harmonic oscillator for comparison experiments
    pub fn harmonic() -> Self {
        Self::with_config(AnharmonicConfig::harmonic())
    }

    /// Get anharmonic frequency for mode n (in MHz)
    ///
    /// Formula: ω_n = n × ω_0 × (1 - β_n)
    pub fn mode_frequency(&self, mode: usize) -> f64 {
        if mode >= self.config.num_modes {
            return 0.0;
        }
        let n = mode as f64 + 1.0;
        n * self.config.omega_0 * (1.0 - self.config.beta[mode])
    }

    /// Get all mode frequencies
    pub fn all_frequencies(&self) -> Vec<f64> {
        (0..self.config.num_modes)
            .map(|m| self.mode_frequency(m))
            .collect()
    }

    /// Evolve oscillator by time step dt (in nanoseconds)
    ///
    /// This updates all phases and amplitudes according to:
    /// 1. Base anharmonic frequency evolution
    /// 2. Non-linear inter-mode coupling
    /// 3. Damping
    pub fn evolve(&mut self, dt_ns: f64) {
        // Convert dt to appropriate units for MHz frequencies
        // 1 MHz = 10^6 cycles/s, 1 ns = 10^-9 s
        // So 1 MHz × 1 ns = 10^-3 cycles = 10^-3 × 2π radians
        let dt_mhz_cycles = dt_ns * 1e-9 * self.config.omega_0 * 1e6;
        let dt_phase = dt_mhz_cycles * 2.0 * PI;

        // Store old phases for coupling calculation
        let old_phases = self.phases.clone();

        for i in 0..self.config.num_modes {
            // Base anharmonic frequency evolution
            let omega_i = self.mode_frequency(i);
            let relative_freq = omega_i / self.config.omega_0;
            self.phases[i] += relative_freq * dt_phase;

            // Non-linear coupling effects (creates frequency mixing)
            for j in 0..self.config.num_modes {
                if i != j && self.config.coupling[i][j] > 0.0 {
                    let coupling_effect = self.config.coupling[i][j]
                        * self.amplitudes[j]
                        * (old_phases[j] - old_phases[i]).sin();
                    self.phases[i] += coupling_effect * dt_phase;
                }
            }

            // Apply damping
            let damp_factor = (-self.config.damping[i] * dt_ns * 1e-3).exp();
            self.amplitudes[i] *= damp_factor;

            // Ensure amplitude doesn't go negative
            self.amplitudes[i] = self.amplitudes[i].max(0.0);

            // Wrap phase to [0, 2π)
            self.phases[i] = self.phases[i] % (2.0 * PI);
            if self.phases[i] < 0.0 {
                self.phases[i] += 2.0 * PI;
            }
        }

        self.time += dt_ns;

        // Record interference pattern
        let interference = self.interference_pattern();
        self.interference_history.push(interference);
    }

    /// Compute interference pattern (sum of all modes)
    ///
    /// This is the superposition of all oscillation modes,
    /// which creates complex time-varying patterns.
    pub fn interference_pattern(&self) -> f64 {
        let mut sum = 0.0;
        let mut total_amp = 0.0;

        for i in 0..self.config.num_modes {
            sum += self.amplitudes[i] * self.phases[i].cos();
            total_amp += self.amplitudes[i];
        }

        if total_amp > 0.0 {
            sum / total_amp
        } else {
            0.0
        }
    }

    /// Compute quadrature component (for complex interference)
    pub fn quadrature_pattern(&self) -> f64 {
        let mut sum = 0.0;
        let mut total_amp = 0.0;

        for i in 0..self.config.num_modes {
            sum += self.amplitudes[i] * self.phases[i].sin();
            total_amp += self.amplitudes[i];
        }

        if total_amp > 0.0 {
            sum / total_amp
        } else {
            0.0
        }
    }

    /// Compute instantaneous power
    pub fn instantaneous_power(&self) -> f64 {
        let i = self.interference_pattern();
        let q = self.quadrature_pattern();
        (i * i + q * q).sqrt()
    }

    /// Compute beat frequencies between all mode pairs (in MHz)
    ///
    /// Beats arise from interference between modes with different frequencies
    pub fn beat_frequencies(&self) -> Vec<f64> {
        let mut beats = Vec::new();
        for i in 0..self.config.num_modes {
            for j in (i + 1)..self.config.num_modes {
                let f_i = self.mode_frequency(i);
                let f_j = self.mode_frequency(j);
                beats.push((f_i - f_j).abs());
            }
        }
        beats
    }

    /// Estimate gamma-band power (~40 Hz) emerging from MHz interference
    ///
    /// This is a simplified model - the real mechanism involves
    /// non-linear mixing and neural entrainment
    pub fn gamma_power(&self) -> f64 {
        let beats = self.beat_frequencies();

        // Look for beats near 40 Hz = 0.04 MHz
        let target = 0.04; // 40 Hz in MHz
        let tolerance = 0.02; // ±20 Hz

        let mut gamma_power = 0.0;
        for &beat in &beats {
            if (beat - target).abs() < tolerance {
                // Contribution weighted by amplitudes of involved modes
                gamma_power += 1.0 / (1.0 + ((beat - target) / tolerance).powi(2));
            }
        }

        // Normalize
        gamma_power.min(1.0)
    }

    /// Compute modulation signal for quantum operations
    ///
    /// Returns a value in [-1, 1] based on current interference state
    pub fn quantum_modulation(&self) -> f64 {
        self.interference_pattern()
    }

    /// Inject energy into a specific mode
    pub fn excite_mode(&mut self, mode: usize, energy: f64) {
        if mode < self.config.num_modes {
            self.amplitudes[mode] = (self.amplitudes[mode] + energy).min(2.0);
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        self.phases = self.config.initial_phases.clone();
        self.amplitudes = self.config.initial_amplitudes.clone();
        self.time = 0.0;
        self.interference_history.clear();
    }

    /// Get interference history
    pub fn history(&self) -> &[f64] {
        &self.interference_history
    }

    /// Compute phase coherence between modes
    ///
    /// High coherence = modes are phase-locked
    /// Low coherence = modes are desynchronized
    pub fn phase_coherence(&self) -> f64 {
        if self.config.num_modes < 2 {
            return 1.0;
        }

        // Compute Kuramoto-like order parameter
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for i in 0..self.config.num_modes {
            real_sum += self.amplitudes[i] * self.phases[i].cos();
            imag_sum += self.amplitudes[i] * self.phases[i].sin();
        }

        let total_amp: f64 = self.amplitudes.iter().sum();
        if total_amp > 0.0 {
            (real_sum * real_sum + imag_sum * imag_sum).sqrt() / total_amp
        } else {
            0.0
        }
    }
}

impl Default for AnharmonicOscillator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// COUPLING TO QUANTUM STATES
// ============================================================

/// Coupling configuration between oscillator and quantum states
#[derive(Clone, Debug)]
pub struct QuantumCouplingConfig {
    /// Base coupling strength
    pub coupling_strength: f64,

    /// How much phase affects tunneling rate
    pub phase_coupling: f64,

    /// How much amplitude affects coherence
    pub amplitude_coupling: f64,

    /// Minimum coherence factor
    pub min_coherence_factor: f64,
}

impl Default for QuantumCouplingConfig {
    fn default() -> Self {
        Self {
            coupling_strength: 0.1,
            phase_coupling: 0.5,
            amplitude_coupling: 0.3,
            min_coherence_factor: 0.5,
        }
    }
}

/// Compute rotation angle for quantum gate based on oscillator state
pub fn oscillator_to_rotation(
    oscillator: &AnharmonicOscillator,
    mode: usize,
    config: &QuantumCouplingConfig,
) -> f64 {
    if mode >= oscillator.config.num_modes {
        return 0.0;
    }

    let phase = oscillator.phases[mode];
    let amplitude = oscillator.amplitudes[mode];

    // Phase modulates rotation angle
    // Amplitude modulates strength
    config.coupling_strength * phase.cos() * amplitude * config.phase_coupling * PI
}

/// Compute coherence modulation factor from oscillator state
pub fn oscillator_coherence_factor(
    oscillator: &AnharmonicOscillator,
    config: &QuantumCouplingConfig,
) -> f64 {
    let interference = oscillator.interference_pattern().abs();

    // High interference = more "active" = slightly reduced coherence
    // Low interference = more "quiet" = increased coherence
    let factor = 1.0 - config.amplitude_coupling * interference;

    factor.max(config.min_coherence_factor)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-6;

    #[test]
    fn test_default_config() {
        let config = AnharmonicConfig::microtubule();

        assert_eq!(config.num_modes, 4);
        assert!((config.omega_0 - 40.0).abs() < TOLERANCE);
        assert!((config.beta[1] - 0.006).abs() < TOLERANCE);
    }

    #[test]
    fn test_anharmonic_frequencies() {
        let osc = AnharmonicOscillator::new();

        // Fundamental should be 40 MHz
        let f0 = osc.mode_frequency(0);
        assert!((f0 - 40.0).abs() < 0.1, "Expected 40 MHz, got {}", f0);

        // Mode 1 should be anharmonic (not exactly 80 MHz)
        let f1 = osc.mode_frequency(1);
        assert!(f1 < 80.0, "Mode 1 should be < 80 MHz, got {}", f1);
        assert!(f1 > 75.0, "Mode 1 should be > 75 MHz, got {}", f1);

        // Verify anharmonic compression
        assert!(f1 < 2.0 * f0, "Anharmonic should compress frequencies");
    }

    #[test]
    fn test_harmonic_vs_anharmonic() {
        let anharmonic = AnharmonicOscillator::new();
        let harmonic = AnharmonicOscillator::harmonic();

        // Harmonic should have exact integer ratios
        let h_f0 = harmonic.mode_frequency(0);
        let h_f1 = harmonic.mode_frequency(1);
        assert!((h_f1 - 2.0 * h_f0).abs() < TOLERANCE);

        // Anharmonic should not
        let a_f0 = anharmonic.mode_frequency(0);
        let a_f1 = anharmonic.mode_frequency(1);
        assert!((a_f1 - 2.0 * a_f0).abs() > 0.1);
    }

    #[test]
    fn test_beat_frequencies() {
        let osc = AnharmonicOscillator::new();
        let beats = osc.beat_frequencies();

        // Should have n*(n-1)/2 beats for n modes
        assert_eq!(beats.len(), 6); // 4 modes = 6 pairs

        // All beats should be positive
        for &beat in &beats {
            assert!(beat > 0.0);
        }
    }

    #[test]
    fn test_evolution() {
        let mut osc = AnharmonicOscillator::new();

        let initial_phase = osc.phases[0];

        // Evolve for 1000 ns (1 μs)
        for _ in 0..1000 {
            osc.evolve(1.0);
        }

        // Phase should have changed
        assert!(
            (osc.phases[0] - initial_phase).abs() > 0.01,
            "Phase should evolve"
        );

        // Time should be tracked
        assert!((osc.time - 1000.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_interference_pattern() {
        let osc = AnharmonicOscillator::new();
        let pattern = osc.interference_pattern();

        // Should be in [-1, 1] range
        assert!(pattern >= -1.0 && pattern <= 1.0);
    }

    #[test]
    fn test_damping() {
        let mut osc = AnharmonicOscillator::new();

        let initial_amp = osc.amplitudes[1];

        // Evolve for many steps
        for _ in 0..10000 {
            osc.evolve(1.0);
        }

        // Amplitude should decrease due to damping
        assert!(osc.amplitudes[1] < initial_amp);
    }

    #[test]
    fn test_excite_mode() {
        let mut osc = AnharmonicOscillator::new();

        osc.excite_mode(0, 0.5);

        // Amplitude should increase
        assert!(osc.amplitudes[0] > 1.0);

        // But not exceed maximum
        osc.excite_mode(0, 10.0);
        assert!(osc.amplitudes[0] <= 2.0);
    }

    #[test]
    fn test_reset() {
        let mut osc = AnharmonicOscillator::new();

        // Evolve
        for _ in 0..100 {
            osc.evolve(1.0);
        }

        // Reset
        osc.reset();

        // Should return to initial state
        assert!((osc.phases[0] - 0.0).abs() < TOLERANCE);
        assert!((osc.amplitudes[0] - 1.0).abs() < TOLERANCE);
        assert!((osc.time - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_phase_coherence() {
        let osc = AnharmonicOscillator::new();
        let coherence = osc.phase_coherence();

        // Should be in [0, 1]
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_gamma_power() {
        let osc = AnharmonicOscillator::new();
        let gamma = osc.gamma_power();

        // Should be in [0, 1]
        assert!(gamma >= 0.0 && gamma <= 1.0);
    }

    #[test]
    fn test_quantum_coupling() {
        let osc = AnharmonicOscillator::new();
        let config = QuantumCouplingConfig::default();

        let rotation = oscillator_to_rotation(&osc, 0, &config);

        // Rotation should be bounded
        assert!(rotation.abs() <= PI);
    }

    #[test]
    fn test_coherence_factor() {
        let osc = AnharmonicOscillator::new();
        let config = QuantumCouplingConfig::default();

        let factor = oscillator_coherence_factor(&osc, &config);

        // Should be in valid range
        assert!(factor >= config.min_coherence_factor);
        assert!(factor <= 1.0);
    }

    #[test]
    fn test_interference_history() {
        let mut osc = AnharmonicOscillator::new();

        for _ in 0..10 {
            osc.evolve(1.0);
        }

        // Should have recorded history
        assert_eq!(osc.history().len(), 10);
    }
}
