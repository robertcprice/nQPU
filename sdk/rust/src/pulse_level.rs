//! Pulse-level simulation and optimization.

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

pub type C64 = Complex64;

#[derive(Clone, Debug)]
pub enum PulseShape {
    Square,
    Gaussian { sigma: f64 },
    Drag { sigma: f64, beta: f64 },
}

#[derive(Clone, Debug)]
pub struct Pulse {
    pub channel: usize,
    pub t0: f64,
    pub duration: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub detuning: f64,
    pub shape: PulseShape,
}

#[derive(Clone, Debug)]
pub struct PulseHamiltonian {
    pub drift: Array2<C64>,
    pub controls: Vec<Array2<C64>>,
}

#[derive(Clone, Debug)]
pub struct PulseSimulator {
    pub hamiltonian: PulseHamiltonian,
    pub pulses: Vec<Pulse>,
    pub dt: f64,
    pub total_time: f64,
}

#[derive(Clone, Debug)]
pub struct GrapeConfig {
    pub iterations: usize,
    pub learning_rate: f64,
    pub epsilon: f64,
}

impl Default for GrapeConfig {
    fn default() -> Self {
        Self {
            iterations: 32,
            learning_rate: 0.2,
            epsilon: 1e-3,
        }
    }
}

impl PulseSimulator {
    pub fn new(hamiltonian: PulseHamiltonian, dt: f64, total_time: f64) -> Self {
        Self {
            hamiltonian,
            pulses: Vec::new(),
            dt,
            total_time,
        }
    }

    pub fn add_pulse(&mut self, pulse: Pulse) {
        self.pulses.push(pulse);
    }

    pub fn simulate_state(&self, initial: &Array1<C64>) -> Array1<C64> {
        let mut psi = initial.clone();
        let n_steps = (self.total_time / self.dt).ceil() as usize;

        for step in 0..n_steps {
            let t = step as f64 * self.dt;
            let h = self.h_total(t);

            // First-order unitary update: |psi'> = (I - i H dt)|psi>
            let mut k = h.dot(&psi);
            for x in &mut k {
                *x *= C64::new(0.0, -self.dt);
            }
            psi = &psi + &k;
            renormalize(&mut psi);
        }

        psi
    }

    /// Floquet effective Hamiltonian by temporal averaging over one period.
    pub fn floquet_effective_hamiltonian(&self, period: f64, samples: usize) -> Array2<C64> {
        let dim = self.hamiltonian.drift.shape()[0];
        let mut h_eff = Array2::zeros((dim, dim));

        let n = samples.max(1);
        for i in 0..n {
            let t = period * (i as f64) / (n as f64);
            h_eff = h_eff + self.h_total(t);
        }

        h_eff.mapv(|x: C64| x / n as f64)
    }

    /// Optimize selected pulse amplitudes with finite-difference GRAPE.
    pub fn optimize_grape(
        &mut self,
        pulse_indices: &[usize],
        initial_state: &Array1<C64>,
        target_state: &Array1<C64>,
        cfg: &GrapeConfig,
    ) {
        for _ in 0..cfg.iterations {
            let base_final = self.simulate_state(initial_state);
            let base_obj = state_fidelity(&base_final, target_state);

            for &idx in pulse_indices {
                if idx >= self.pulses.len() {
                    continue;
                }

                self.pulses[idx].amplitude += cfg.epsilon;
                let up = state_fidelity(&self.simulate_state(initial_state), target_state);

                self.pulses[idx].amplitude -= 2.0 * cfg.epsilon;
                let down = state_fidelity(&self.simulate_state(initial_state), target_state);

                self.pulses[idx].amplitude += cfg.epsilon;

                let grad = (up - down) / (2.0 * cfg.epsilon);
                self.pulses[idx].amplitude += cfg.learning_rate * grad;
            }

            let new_obj = state_fidelity(&self.simulate_state(initial_state), target_state);
            if (new_obj - base_obj).abs() < 1e-8 {
                break;
            }
        }
    }

    fn h_total(&self, t: f64) -> Array2<C64> {
        let mut h = self.hamiltonian.drift.clone();

        for p in &self.pulses {
            let env = envelope(p, t);
            if env.abs() < 1e-14 {
                continue;
            }

            let phase = p.phase + 2.0 * PI * p.detuning * (t - p.t0);
            let coeff = C64::new(env * phase.cos(), env * phase.sin());

            if let Some(ctrl) = self.hamiltonian.controls.get(p.channel) {
                h = h + ctrl.mapv(|x| coeff * x);
            }
        }

        h
    }
}

pub fn state_fidelity(a: &Array1<C64>, b: &Array1<C64>) -> f64 {
    let mut inner = C64::new(0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        inner += x.conj() * y;
    }
    inner.norm_sqr()
}

fn envelope(p: &Pulse, t: f64) -> f64 {
    if t < p.t0 || t > p.t0 + p.duration {
        return 0.0;
    }

    let tau = t - p.t0;
    let mid = p.duration * 0.5;

    match p.shape {
        PulseShape::Square => p.amplitude,
        PulseShape::Gaussian { sigma } => {
            let x = (tau - mid) / sigma.max(1e-9);
            p.amplitude * (-0.5 * x * x).exp()
        }
        PulseShape::Drag { sigma, beta } => {
            let x = (tau - mid) / sigma.max(1e-9);
            let gauss = (-0.5 * x * x).exp();
            let drag = -beta * x * gauss;
            p.amplitude * (gauss + drag)
        }
    }
}

fn renormalize(v: &mut Array1<C64>) {
    let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pauli_x_half() -> Array2<C64> {
        let mut x = Array2::zeros((2, 2));
        x[[0, 1]] = C64::new(0.5, 0.0);
        x[[1, 0]] = C64::new(0.5, 0.0);
        x
    }

    #[test]
    fn test_pi_pulse_flips_qubit() {
        let ham = PulseHamiltonian {
            drift: Array2::zeros((2, 2)),
            controls: vec![pauli_x_half()],
        };
        let mut sim = PulseSimulator::new(ham, 1e-3, 1.0);
        sim.add_pulse(Pulse {
            channel: 0,
            t0: 0.0,
            duration: 1.0,
            amplitude: PI,
            phase: 0.0,
            detuning: 0.0,
            shape: PulseShape::Square,
        });

        let mut psi0 = Array1::zeros(2);
        psi0[0] = C64::new(1.0, 0.0);

        let final_state = sim.simulate_state(&psi0);
        let p1 = final_state[1].norm_sqr();
        assert!(p1 > 0.95);
    }

    #[test]
    fn test_grape_improves_fidelity() {
        let ham = PulseHamiltonian {
            drift: Array2::zeros((2, 2)),
            controls: vec![pauli_x_half()],
        };
        let mut sim = PulseSimulator::new(ham, 2e-3, 1.0);
        sim.add_pulse(Pulse {
            channel: 0,
            t0: 0.0,
            duration: 1.0,
            amplitude: 1.0,
            phase: 0.0,
            detuning: 0.0,
            shape: PulseShape::Square,
        });

        let mut psi0 = Array1::zeros(2);
        psi0[0] = C64::new(1.0, 0.0);
        let mut target = Array1::zeros(2);
        target[1] = C64::new(1.0, 0.0);

        let before = state_fidelity(&sim.simulate_state(&psi0), &target);
        sim.optimize_grape(&[0], &psi0, &target, &GrapeConfig::default());
        let after = state_fidelity(&sim.simulate_state(&psi0), &target);

        assert!(after >= before);
    }

    #[test]
    fn test_floquet_effective_hamiltonian_shape() {
        let ham = PulseHamiltonian {
            drift: Array2::zeros((2, 2)),
            controls: vec![pauli_x_half()],
        };
        let sim = PulseSimulator::new(ham, 1e-3, 0.1);
        let h = sim.floquet_effective_hamiltonian(0.1, 16);
        assert_eq!(h.shape(), &[2, 2]);
    }
}
