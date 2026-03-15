//! Microtubule-Inspired Feature Augmentor
//!
//! This module provides an opt-in bridge between:
//! 1) Orch-OR microtubule dynamics (`orch_or`), and
//! 2) transformer-style token feature pipelines.
//!
//! It is intended as an experimental inductive bias, not as evidence that
//! model outputs are conscious.

use crate::orch_or::{ConsciousnessMeasure, OrchORConfig, OrchORError, OrchORSimulator};

/// Configuration for the microtubule-inspired augmentor.
#[derive(Clone, Debug)]
pub struct MicrotubuleAugmentorConfig {
    /// Orch-OR simulator configuration used as the internal state machine.
    pub orch_or: OrchORConfig,
    /// Number of Orch-OR evolution steps per token.
    pub micro_steps_per_token: usize,
    /// How strongly augmentor signals modulate token features [0, 1].
    pub blend_strength: f64,
    /// EMA decay for the gate signal [0, 1).
    pub feedback_decay: f64,
    /// Gate bias before sigmoid.
    pub gate_bias: f64,
    /// Weight for coherence in gate computation.
    pub coherence_weight: f64,
    /// Weight for entanglement in gate computation.
    pub entanglement_weight: f64,
    /// Weight for orchestration in gate computation.
    pub orchestration_weight: f64,
    /// Weight for reduction-event rate in gate computation.
    pub reduction_weight: f64,
    /// Weight for anesthetic suppression in gate computation.
    pub anesthetic_weight: f64,
    /// Small oscillatory term to mimic rhythmic modulation.
    pub oscillation_strength: f64,
    /// Phase advance per token for oscillatory modulation.
    pub phase_step: f64,
    /// Enable initial Hadamard superposition of tubulins.
    pub initialize_superposition: bool,
}

impl Default for MicrotubuleAugmentorConfig {
    fn default() -> Self {
        Self {
            orch_or: OrchORConfig::default(),
            micro_steps_per_token: 2,
            blend_strength: 0.15,
            feedback_decay: 0.85,
            gate_bias: -0.35,
            coherence_weight: 1.20,
            entanglement_weight: 0.85,
            orchestration_weight: 0.65,
            reduction_weight: 0.50,
            anesthetic_weight: 1.20,
            oscillation_strength: 0.15,
            phase_step: 0.35,
            initialize_superposition: true,
        }
    }
}

impl MicrotubuleAugmentorConfig {
    /// Create a config with defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Signal emitted by the microtubule state machine for one token step.
#[derive(Clone, Debug)]
pub struct MicrotubuleSignal {
    /// Mean coherence from Orch-OR consciousness metrics.
    pub coherence: f64,
    /// Mean pairwise entanglement.
    pub entanglement: f64,
    /// Orchestration level from Orch-OR metrics.
    pub orchestration: f64,
    /// Reduction events per evolution step in this token update.
    pub reduction_rate: f64,
    /// Raw logistic gate before EMA smoothing.
    pub raw_gate: f64,
    /// Smoothed gate used for feature modulation.
    pub gate: f64,
}

/// Stateful augmentor driven by Orch-OR dynamics.
pub struct MicrotubuleAugmentor {
    /// Runtime configuration.
    pub config: MicrotubuleAugmentorConfig,
    /// Underlying Orch-OR simulator.
    pub simulator: OrchORSimulator,
    token_index: usize,
    ema_gate: f64,
    seen_reductions: usize,
    last_signal: Option<MicrotubuleSignal>,
}

impl MicrotubuleAugmentor {
    /// Construct a new augmentor.
    pub fn new(config: MicrotubuleAugmentorConfig) -> Result<Self, OrchORError> {
        let mut simulator = OrchORSimulator::new(config.orch_or.clone())?;
        if config.initialize_superposition {
            simulator.initialize_superposition();
        }

        Ok(Self {
            config,
            simulator,
            token_index: 0,
            ema_gate: 0.5,
            seen_reductions: 0,
            last_signal: None,
        })
    }

    /// Apply an anesthetic concentration to the underlying Orch-OR simulator.
    pub fn set_anesthetic(&mut self, concentration: f64) {
        self.simulator.apply_anesthetic(concentration);
    }

    /// Return the most recent signal if one has been computed.
    pub fn last_signal(&self) -> Option<&MicrotubuleSignal> {
        self.last_signal.as_ref()
    }

    /// Advance the internal state machine and return the new control signal.
    pub fn step_signal(&mut self) -> MicrotubuleSignal {
        let steps = self.config.micro_steps_per_token.max(1);
        let snapshots = self.simulator.evolve(steps);

        let measure = snapshots
            .last()
            .map(|s| s.consciousness.clone())
            .unwrap_or_else(|| self.simulator.consciousness_measure());

        let total_reductions = self.simulator.reduction_history.len();
        let new_reductions = total_reductions.saturating_sub(self.seen_reductions);
        self.seen_reductions = total_reductions;
        let reduction_rate = new_reductions as f64 / steps as f64;

        let raw_gate = self.raw_gate(&measure, reduction_rate);
        let decay = self.config.feedback_decay.clamp(0.0, 0.999);
        self.ema_gate = decay * self.ema_gate + (1.0 - decay) * raw_gate;

        let signal = MicrotubuleSignal {
            coherence: measure.coherence,
            entanglement: measure.entanglement,
            orchestration: measure.orchestration_level,
            reduction_rate,
            raw_gate,
            gate: self.ema_gate.clamp(0.0, 1.0),
        };

        self.last_signal = Some(signal.clone());
        signal
    }

    /// Modulate a single token feature vector.
    pub fn augment_token(&mut self, token: &[f64]) -> (Vec<f64>, MicrotubuleSignal) {
        let signal = self.step_signal();
        let blend = self.config.blend_strength.clamp(0.0, 1.0);

        let phase = (self.token_index as f64 * self.config.phase_step).sin();
        let oscillatory = blend * self.config.oscillation_strength * phase * signal.orchestration;

        // Gate in [0,1] -> scale around 1.0 with bounded range from blend_strength.
        let base_scale = 1.0 + blend * (2.0 * signal.gate - 1.0);
        let global_scale = (base_scale + oscillatory).max(0.0);

        let reduction_damp = 1.0 - 0.4 * blend * signal.reduction_rate.min(1.0);
        let offset = blend * 0.02 * (signal.coherence - signal.entanglement);

        let mut out = Vec::with_capacity(token.len());
        for (i, &x) in token.iter().enumerate() {
            let local_scale = if i % 2 == 0 {
                global_scale
            } else {
                global_scale * reduction_damp
            };
            out.push(x * local_scale + offset);
        }

        self.token_index = self.token_index.saturating_add(1);
        (out, signal)
    }

    /// Modulate a sequence of token feature vectors.
    pub fn augment_sequence(
        &mut self,
        tokens: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<MicrotubuleSignal>) {
        let mut output = Vec::with_capacity(tokens.len());
        let mut signals = Vec::with_capacity(tokens.len());

        for token in tokens {
            let (aug, signal) = self.augment_token(token);
            output.push(aug);
            signals.push(signal);
        }

        (output, signals)
    }

    fn raw_gate(&self, measure: &ConsciousnessMeasure, reduction_rate: f64) -> f64 {
        let z = self.config.gate_bias
            + self.config.coherence_weight * measure.coherence
            + self.config.entanglement_weight * measure.entanglement
            + self.config.orchestration_weight * measure.orchestration_level
            + self.config.reduction_weight * reduction_rate
            - self.config.anesthetic_weight * measure.anesthetic_suppression;

        sigmoid(z).clamp(0.0, 1.0)
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_orch_or(seed: u64, anesthetic: f64) -> OrchORConfig {
        OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(25.0)
            .temperature_kelvin(310.0)
            .coupling_strength(0.02)
            .anesthetic_concentration(anesthetic)
            .seed(seed)
    }

    #[test]
    fn test_augment_sequence_shape_and_signal_range() {
        let cfg = MicrotubuleAugmentorConfig {
            orch_or: test_orch_or(7, 0.0),
            ..MicrotubuleAugmentorConfig::default()
        };
        let mut aug = MicrotubuleAugmentor::new(cfg).unwrap();

        let tokens = vec![
            vec![0.1, 0.2, -0.3, 0.4],
            vec![0.3, -0.1, 0.5, -0.2],
            vec![0.7, 0.1, 0.0, -0.4],
        ];

        let (out, signals) = aug.augment_sequence(&tokens);
        assert_eq!(out.len(), tokens.len());
        assert_eq!(signals.len(), tokens.len());
        for (o, i) in out.iter().zip(tokens.iter()) {
            assert_eq!(o.len(), i.len());
        }
        for s in signals {
            assert!((0.0..=1.0).contains(&s.raw_gate));
            assert!((0.0..=1.0).contains(&s.gate));
        }
    }

    #[test]
    fn test_blend_zero_is_identity() {
        let cfg = MicrotubuleAugmentorConfig {
            orch_or: test_orch_or(11, 0.0),
            blend_strength: 0.0,
            ..MicrotubuleAugmentorConfig::default()
        };
        let mut aug = MicrotubuleAugmentor::new(cfg).unwrap();

        let token = vec![0.25, -0.5, 0.75, -1.0];
        let (out, _signal) = aug.augment_token(&token);

        for (a, b) in out.iter().zip(token.iter()) {
            assert!(
                (a - b).abs() < 1e-12,
                "expected identity, got {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_anesthetic_reduces_modulation_strength() {
        let cfg_clear = MicrotubuleAugmentorConfig {
            orch_or: test_orch_or(3, 0.0),
            ..MicrotubuleAugmentorConfig::default()
        };
        let cfg_suppressed = MicrotubuleAugmentorConfig {
            orch_or: test_orch_or(3, 0.9),
            ..MicrotubuleAugmentorConfig::default()
        };

        let mut clear = MicrotubuleAugmentor::new(cfg_clear).unwrap();
        let mut suppressed = MicrotubuleAugmentor::new(cfg_suppressed).unwrap();

        let token = vec![1.0, 1.0, 1.0, 1.0];
        let mut clear_energy = 0.0;
        let mut suppressed_energy = 0.0;

        for _ in 0..12 {
            let (a, _) = clear.augment_token(&token);
            let (b, _) = suppressed.augment_token(&token);
            clear_energy += a.iter().map(|x| x.abs()).sum::<f64>() / a.len() as f64;
            suppressed_energy += b.iter().map(|x| x.abs()).sum::<f64>() / b.len() as f64;
        }

        assert!(
            clear_energy > suppressed_energy,
            "expected lower modulation with anesthetic: clear={} suppressed={}",
            clear_energy,
            suppressed_energy
        );
    }
}
