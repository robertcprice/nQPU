use nqpu_metal::density_matrix::DensityMatrixSimulator;
use nqpu_metal::full_quantum_transformer::FullyQuantumTransformer;
use nqpu_metal::microtubule_augmentor::{MicrotubuleAugmentor, MicrotubuleAugmentorConfig};
use nqpu_metal::orch_or::OrchORConfig;
use std::env;
use std::time::Instant;

#[derive(Clone, Copy)]
struct AblationCondition {
    name: &'static str,
    blend_strength: f64,
    anesthetic: f64,
}

#[derive(Default)]
struct AblationResult {
    name: String,
    mean_entropy: f64,
    mean_top1_gap: f64,
    mean_inter_prompt_drift: f64,
    mean_gate: f64,
    mean_coherence: f64,
    mean_entanglement: f64,
    mean_orchestration: f64,
    mean_reduction_rate: f64,
    total_ms: f64,
}

fn main() {
    let mut num_prompts = 32usize;
    let mut seq_len = 8usize;
    let mut vocab_size = 64usize;

    parse_args(&mut num_prompts, &mut seq_len, &mut vocab_size);

    let prompts = build_prompts(num_prompts, seq_len, vocab_size);
    let conditions = [
        AblationCondition {
            name: "control_blend0",
            blend_strength: 0.0,
            anesthetic: 0.0,
        },
        AblationCondition {
            name: "baseline",
            blend_strength: 0.15,
            anesthetic: 0.0,
        },
        AblationCondition {
            name: "high_anesthetic",
            blend_strength: 0.15,
            anesthetic: 0.9,
        },
    ];

    println!("Microtubule Augmentor Ablation");
    println!("==============================");
    println!(
        "prompts={}, seq_len={}, vocab_size={}",
        num_prompts, seq_len, vocab_size
    );
    println!();

    let mut results = Vec::new();
    for c in conditions {
        let result = run_condition(c, &prompts, seq_len, vocab_size);
        results.push(result);
    }

    print_results(&results);
}

fn parse_args(num_prompts: &mut usize, seq_len: &mut usize, vocab_size: &mut usize) {
    let args: Vec<String> = env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompts" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    *num_prompts = v.max(1);
                }
                i += 2;
            }
            "--seq-len" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    *seq_len = v.max(2);
                }
                i += 2;
            }
            "--vocab" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    *vocab_size = v.max(8);
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
}

fn build_prompts(num_prompts: usize, seq_len: usize, vocab_size: usize) -> Vec<Vec<usize>> {
    let mut prompts = Vec::with_capacity(num_prompts);
    for p in 0..num_prompts {
        let mut prompt = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            // Deterministic pseudo-diverse token pattern.
            let token = (p * 17 + t * 13 + (p ^ t) * 7) % vocab_size;
            prompt.push(token);
        }
        prompts.push(prompt);
    }
    prompts
}

fn run_condition(
    condition: AblationCondition,
    prompts: &[Vec<usize>],
    seq_len: usize,
    vocab_size: usize,
) -> AblationResult {
    let mut transformer =
        FullyQuantumTransformer::new_with_model_dim(2, 4, seq_len, vocab_size, 64);

    let mut augmentor = MicrotubuleAugmentor::new(MicrotubuleAugmentorConfig {
        orch_or: OrchORConfig::new()
            .num_tubulins(4)
            .coherence_time_ns(25.0)
            .temperature_kelvin(310.0)
            .coupling_strength(0.02)
            .anesthetic_concentration(condition.anesthetic)
            .seed(42),
        blend_strength: condition.blend_strength,
        micro_steps_per_token: 1,
        ..MicrotubuleAugmentorConfig::default()
    })
    .expect("valid microtubule augmentor config");

    let start = Instant::now();

    let mut entropy_sum = 0.0;
    let mut gap_sum = 0.0;
    let mut drift_sum = 0.0;
    let mut drift_count = 0usize;

    let mut gate_sum = 0.0;
    let mut coherence_sum = 0.0;
    let mut entanglement_sum = 0.0;
    let mut orchestration_sum = 0.0;
    let mut reduction_sum = 0.0;
    let mut signal_count = 0usize;

    let mut prev_probs: Option<Vec<f64>> = None;

    for prompt in prompts {
        let mut sim = DensityMatrixSimulator::new(6);
        let (logits, signals) =
            transformer.forward_with_augmentor_trace(prompt, &mut sim, Some(&mut augmentor));

        let probs = safe_softmax(&logits);
        entropy_sum += normalized_entropy_from_probs(&probs);
        gap_sum += top1_gap(&probs);

        if let Some(prev) = &prev_probs {
            drift_sum += l2_distance(prev, &probs);
            drift_count += 1;
        }
        prev_probs = Some(probs);

        for s in signals {
            gate_sum += s.gate;
            coherence_sum += s.coherence;
            entanglement_sum += s.entanglement;
            orchestration_sum += s.orchestration;
            reduction_sum += s.reduction_rate;
            signal_count += 1;
        }
    }

    let n = prompts.len() as f64;
    let sig_n = (signal_count as f64).max(1.0);

    AblationResult {
        name: condition.name.to_string(),
        mean_entropy: entropy_sum / n,
        mean_top1_gap: gap_sum / n,
        mean_inter_prompt_drift: if drift_count > 0 {
            drift_sum / drift_count as f64
        } else {
            0.0
        },
        mean_gate: gate_sum / sig_n,
        mean_coherence: coherence_sum / sig_n,
        mean_entanglement: entanglement_sum / sig_n,
        mean_orchestration: orchestration_sum / sig_n,
        mean_reduction_rate: reduction_sum / sig_n,
        total_ms: start.elapsed().as_secs_f64() * 1000.0,
    }
}

fn normalized_entropy_from_probs(probs: &[f64]) -> f64 {
    let h = probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    let max_h = (probs.len() as f64).ln().max(1e-12);
    h / max_h
}

fn top1_gap(logits: &[f64]) -> f64 {
    if logits.len() < 2 {
        return 0.0;
    }
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for &x in logits {
        if x > best {
            second = best;
            best = x;
        } else if x > second {
            second = x;
        }
    }
    (best - second).max(0.0)
}

fn safe_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_v = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0;
    for &x in logits {
        let e = (x - max_v).exp();
        exps.push(e);
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    exps.into_iter().map(|e| e * inv).collect()
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 0..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    (acc / n as f64).sqrt()
}

fn print_results(results: &[AblationResult]) {
    println!(
        "{:<18} {:>8} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "condition",
        "entropy",
        "top1_gap",
        "drift",
        "gate",
        "coherence",
        "entangle",
        "orchestr",
        "reduction",
        "ms"
    );
    println!("{}", "-".repeat(114));

    for r in results {
        println!(
            "{:<18} {:>8.4} {:>10.4} {:>10.4} {:>8.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>9.1}",
            r.name,
            r.mean_entropy,
            r.mean_top1_gap,
            r.mean_inter_prompt_drift,
            r.mean_gate,
            r.mean_coherence,
            r.mean_entanglement,
            r.mean_orchestration,
            r.mean_reduction_rate,
            r.total_ms
        );
    }
    println!();
    println!("Interpretation notes:");
    println!("  - entropy: normalized output entropy (higher = flatter output distribution)");
    println!("  - top1_gap: confidence separation between top-1 and top-2 probabilities");
    println!("  - drift: L2 change in output probabilities across successive prompts");
    println!("  - gate/coherence/...: mean augmentor control signals");
}
