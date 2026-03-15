//! Heisenberg-limited Quantum Phase Estimation.
//!
//! Uses adaptive Bayesian phase updates with exponentially growing evolution
//! times to approach O(1/T) precision scaling.

use std::f64::consts::PI;

/// Configuration for adaptive Heisenberg-limited phase estimation.
#[derive(Clone, Debug)]
pub struct HeisenbergQpeConfig {
    pub rounds: usize,
    pub shots_per_round: usize,
    pub posterior_grid_size: usize,
}

impl Default for HeisenbergQpeConfig {
    fn default() -> Self {
        Self {
            rounds: 10,
            shots_per_round: 64,
            posterior_grid_size: 2048,
        }
    }
}

/// Measurement oracle for controlled-U^k interferometry.
pub trait PhaseOracle {
    /// Returns number of `1` outcomes out of `shots` for a Ramsey-like experiment
    /// with evolution power `k` and control phase offset `beta`.
    fn sample_ones(&self, k: u64, beta: f64, shots: usize) -> usize;
}

/// Ideal single-phase oracle with optional symmetric readout error.
#[derive(Clone, Debug)]
pub struct IdealPhaseOracle {
    /// True phase in turns: phi in [0, 1), where eigenvalue is exp(i 2π phi).
    pub phase: f64,
    /// Symmetric readout error probability.
    pub readout_error: f64,
}

impl PhaseOracle for IdealPhaseOracle {
    fn sample_ones(&self, k: u64, beta: f64, shots: usize) -> usize {
        let mut ones = 0usize;
        let p1 = noisy_prob_one(self.phase, k, beta, self.readout_error);
        for _ in 0..shots {
            if rand::random::<f64>() < p1 {
                ones += 1;
            }
        }
        ones
    }
}

/// Estimation result with posterior diagnostics.
#[derive(Clone, Debug)]
pub struct HeisenbergQpeResult {
    pub phase_estimate: f64,
    pub circular_std: f64,
    pub heisenberg_bound: f64,
    pub total_query_time: u64,
    pub grid: Vec<f64>,
    pub posterior: Vec<f64>,
}

/// Adaptive Bayesian Heisenberg-limited QPE.
pub fn estimate_phase_heisenberg<O: PhaseOracle>(
    oracle: &O,
    config: &HeisenbergQpeConfig,
) -> HeisenbergQpeResult {
    let n = config.posterior_grid_size.max(64);
    let mut grid = Vec::with_capacity(n);
    for i in 0..n {
        grid.push(i as f64 / n as f64);
    }

    let mut posterior = vec![1.0 / n as f64; n];
    let mut total_query_time: u64 = 0;

    for round in 0..config.rounds {
        let k = 1u64 << round;
        total_query_time += k * config.shots_per_round as u64;

        let mean_phase = circular_mean(&grid, &posterior);
        let beta = -2.0 * PI * (k as f64) * mean_phase;
        let ones = oracle.sample_ones(k, beta, config.shots_per_round);
        let zeros = config.shots_per_round.saturating_sub(ones);

        for (i, p) in posterior.iter_mut().enumerate() {
            let p1 = ideal_prob_one(grid[i], k, beta).clamp(1e-12, 1.0 - 1e-12);
            let likelihood = p1.powi(ones as i32) * (1.0 - p1).powi(zeros as i32);
            *p *= likelihood;
        }

        normalize(&mut posterior);
    }

    let phase_estimate = circular_mean(&grid, &posterior);
    let circular_std = circular_std(&grid, &posterior);
    let heisenberg_bound = 1.0 / (2.0 * PI * (total_query_time as f64).max(1.0));

    HeisenbergQpeResult {
        phase_estimate,
        circular_std,
        heisenberg_bound,
        total_query_time,
        grid,
        posterior,
    }
}

#[inline]
fn ideal_prob_one(phase: f64, k: u64, beta: f64) -> f64 {
    let x = 2.0 * PI * (k as f64) * phase + beta;
    0.5 * (1.0 - x.cos())
}

#[inline]
fn noisy_prob_one(phase: f64, k: u64, beta: f64, readout_error: f64) -> f64 {
    let p = ideal_prob_one(phase, k, beta);
    p * (1.0 - readout_error) + (1.0 - p) * readout_error
}

fn normalize(dist: &mut [f64]) {
    let s: f64 = dist.iter().sum();
    if s <= 0.0 {
        let v = 1.0 / dist.len().max(1) as f64;
        for x in dist {
            *x = v;
        }
        return;
    }
    for x in dist {
        *x /= s;
    }
}

fn circular_mean(grid: &[f64], posterior: &[f64]) -> f64 {
    let mut x = 0.0;
    let mut y = 0.0;
    for (phi, p) in grid.iter().zip(posterior.iter()) {
        let a = 2.0 * PI * phi;
        x += p * a.cos();
        y += p * a.sin();
    }
    let mut angle = y.atan2(x) / (2.0 * PI);
    if angle < 0.0 {
        angle += 1.0;
    }
    angle
}

fn circular_std(grid: &[f64], posterior: &[f64]) -> f64 {
    let mut x = 0.0;
    let mut y = 0.0;
    for (phi, p) in grid.iter().zip(posterior.iter()) {
        let a = 2.0 * PI * phi;
        x += p * a.cos();
        y += p * a.sin();
    }
    let r = (x * x + y * y).sqrt().clamp(1e-12, 1.0);
    (-2.0 * r.ln()).sqrt() / (2.0 * PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DeterministicOracle {
        phase: f64,
    }

    impl PhaseOracle for DeterministicOracle {
        fn sample_ones(&self, k: u64, beta: f64, shots: usize) -> usize {
            let p1 = ideal_prob_one(self.phase, k, beta);
            (p1 * shots as f64).round() as usize
        }
    }

    struct DeterministicNoisyOracle {
        phase: f64,
        readout_error: f64,
    }

    impl PhaseOracle for DeterministicNoisyOracle {
        fn sample_ones(&self, k: u64, beta: f64, shots: usize) -> usize {
            let p1 = noisy_prob_one(self.phase, k, beta, self.readout_error);
            (p1 * shots as f64).round() as usize
        }
    }

    fn wrapped_error(a: f64, b: f64) -> f64 {
        let mut d = (a - b).abs();
        if d > 0.5 {
            d = 1.0 - d;
        }
        d
    }

    #[test]
    fn test_heisenberg_phase_estimation_deterministic() {
        let oracle = DeterministicOracle { phase: 0.318_3 };
        let cfg = HeisenbergQpeConfig {
            rounds: 11,
            shots_per_round: 96,
            posterior_grid_size: 4096,
        };

        let res = estimate_phase_heisenberg(&oracle, &cfg);
        assert!(wrapped_error(res.phase_estimate, oracle.phase) < 0.02);
        assert!(res.circular_std < 0.1);
        assert!(res.total_query_time > 0);
    }

    #[test]
    fn test_ideal_oracle_with_noise() {
        let oracle = DeterministicNoisyOracle {
            phase: 0.125,
            readout_error: 0.02,
        };
        let cfg = HeisenbergQpeConfig {
            rounds: 9,
            shots_per_round: 192,
            posterior_grid_size: 2048,
        };

        let res = estimate_phase_heisenberg(&oracle, &cfg);
        assert!((0.0..1.0).contains(&res.phase_estimate));
        assert!(res.circular_std.is_finite());
        assert!(res.heisenberg_bound > 0.0);
    }
}
