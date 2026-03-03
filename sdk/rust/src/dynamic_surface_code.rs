//! Dynamic surface code simulation with a lightweight RL decoder.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SyndromeSnapshot {
    pub defects: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderAction {
    NoOp,
    ApplyX,
    ApplyZ,
}

#[derive(Clone, Debug)]
pub struct CycleReport {
    pub measured_defects: usize,
    pub action: DecoderAction,
    pub reward: f64,
    pub logical_error_rate: f64,
    pub code_distance: usize,
}

#[derive(Clone, Debug)]
pub struct DynamicSurfaceCode {
    pub distance: usize,
    pub min_distance: usize,
    pub max_distance: usize,
    pub physical_error_rate: f64,
    pub target_logical_error: f64,
    pub last_logical_errors: Vec<f64>,
}

impl DynamicSurfaceCode {
    pub fn new(distance: usize) -> Self {
        Self {
            distance,
            min_distance: 3,
            max_distance: 31,
            physical_error_rate: 1e-3,
            target_logical_error: 1e-4,
            last_logical_errors: Vec::new(),
        }
    }

    pub fn with_error_rate(mut self, p: f64) -> Self {
        self.physical_error_rate = p;
        self
    }

    pub fn sample_syndrome(&self) -> SyndromeSnapshot {
        let checks_per_side = self.distance.saturating_sub(1).max(1);
        let total_checks = checks_per_side * checks_per_side;

        let mut defects = Vec::new();
        for _ in 0..total_checks {
            if rand::random::<f64>() < self.physical_error_rate {
                let x = rand::random::<usize>() % checks_per_side;
                let y = rand::random::<usize>() % checks_per_side;
                defects.push((x, y));
            }
        }

        SyndromeSnapshot { defects }
    }

    pub fn logical_error_proxy(&self, syndrome: &SyndromeSnapshot, action: DecoderAction) -> f64 {
        let d = self.distance as f64;
        let base = self.physical_error_rate.powf((d + 1.0) * 0.5);
        let defect_penalty =
            syndrome.defects.len() as f64 / (self.distance * self.distance).max(1) as f64;
        let action_gain = match action {
            DecoderAction::NoOp => 1.25,
            DecoderAction::ApplyX => 0.85,
            DecoderAction::ApplyZ => 0.85,
        };
        (base * (1.0 + defect_penalty) * action_gain).min(1.0)
    }

    pub fn adapt_distance(&mut self, current_logical_error: f64) {
        self.last_logical_errors.push(current_logical_error);
        if self.last_logical_errors.len() > 32 {
            self.last_logical_errors.remove(0);
        }

        let avg = self.last_logical_errors.iter().sum::<f64>()
            / self.last_logical_errors.len().max(1) as f64;

        if avg > self.target_logical_error * 1.2 && self.distance < self.max_distance {
            self.distance += 2; // maintain odd code distance
        } else if avg < self.target_logical_error * 0.2 && self.distance > self.min_distance {
            self.distance = self.distance.saturating_sub(2).max(self.min_distance);
        }
    }

    pub fn run_cycle(&mut self, decoder: &mut RlDecoder) -> CycleReport {
        let syndrome = self.sample_syndrome();
        let state = DecoderState::from_syndrome(&syndrome, self.distance);
        let action = decoder.select_action(state);

        let logical_error = self.logical_error_proxy(&syndrome, action);
        self.adapt_distance(logical_error);

        let reward = 1.0 - 20.0 * logical_error;
        let next_state = DecoderState::from_syndrome(&self.sample_syndrome(), self.distance);
        decoder.update(state, action, reward, next_state);

        CycleReport {
            measured_defects: syndrome.defects.len(),
            action,
            reward,
            logical_error_rate: logical_error,
            code_distance: self.distance,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct DecoderState {
    defect_bin: u8,
    distance_bin: u8,
}

impl DecoderState {
    fn from_syndrome(s: &SyndromeSnapshot, distance: usize) -> Self {
        let defect_bin = if s.defects.len() == 0 {
            0
        } else if s.defects.len() <= 2 {
            1
        } else if s.defects.len() <= 5 {
            2
        } else {
            3
        };

        let distance_bin = if distance <= 5 {
            0
        } else if distance <= 11 {
            1
        } else {
            2
        };

        Self {
            defect_bin,
            distance_bin,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RlDecoder {
    pub q_table: HashMap<DecoderState, [f64; 3]>,
    pub alpha: f64,
    pub gamma: f64,
    pub epsilon: f64,
}

impl Default for RlDecoder {
    fn default() -> Self {
        Self {
            q_table: HashMap::new(),
            alpha: 0.15,
            gamma: 0.9,
            epsilon: 0.15,
        }
    }
}

impl RlDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn select_action(&mut self, state: DecoderState) -> DecoderAction {
        if rand::random::<f64>() < self.epsilon {
            return match rand::random::<u8>() % 3 {
                0 => DecoderAction::NoOp,
                1 => DecoderAction::ApplyX,
                _ => DecoderAction::ApplyZ,
            };
        }

        let q = self.q_table.entry(state).or_insert([0.0; 3]);
        let idx = argmax(q);
        action_from_idx(idx)
    }

    pub fn update(
        &mut self,
        state: DecoderState,
        action: DecoderAction,
        reward: f64,
        next_state: DecoderState,
    ) {
        let a = action_idx(action);

        let max_next = {
            let qn = self.q_table.entry(next_state).or_insert([0.0; 3]);
            qn.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };

        let q = self.q_table.entry(state).or_insert([0.0; 3]);
        let target = reward + self.gamma * max_next;
        q[a] = (1.0 - self.alpha) * q[a] + self.alpha * target;
    }
}

fn argmax(q: &[f64; 3]) -> usize {
    let mut idx = 0usize;
    let mut best = q[0];
    for (i, &v) in q.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            idx = i;
        }
    }
    idx
}

fn action_idx(a: DecoderAction) -> usize {
    match a {
        DecoderAction::NoOp => 0,
        DecoderAction::ApplyX => 1,
        DecoderAction::ApplyZ => 2,
    }
}

fn action_from_idx(i: usize) -> DecoderAction {
    match i {
        0 => DecoderAction::NoOp,
        1 => DecoderAction::ApplyX,
        _ => DecoderAction::ApplyZ,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_updates_q_table() {
        let mut code = DynamicSurfaceCode::new(5).with_error_rate(0.02);
        let mut decoder = RlDecoder::new();

        for _ in 0..10 {
            let _ = code.run_cycle(&mut decoder);
        }

        assert!(!decoder.q_table.is_empty());
    }

    #[test]
    fn test_distance_adapts_up_for_high_error() {
        let mut code = DynamicSurfaceCode::new(5).with_error_rate(0.1);
        code.target_logical_error = 1e-5;

        for _ in 0..8 {
            code.adapt_distance(5e-3);
        }

        assert!(code.distance >= 7);
    }

    #[test]
    fn test_cycle_report_fields() {
        let mut code = DynamicSurfaceCode::new(7);
        let mut decoder = RlDecoder::new();
        let rep = code.run_cycle(&mut decoder);

        assert!(rep.code_distance >= 3);
        assert!(rep.logical_error_rate >= 0.0);
    }
}
