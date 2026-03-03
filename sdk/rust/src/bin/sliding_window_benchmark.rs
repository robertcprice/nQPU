//! Sliding Window QEC Decoder Benchmark
//!
//! Measures decode throughput of the sliding window decoder at multiple code
//! distances and error rates using both Greedy and UnionFind inner decoders.
//!
//! Generates realistic surface code syndrome data: `d^2 - 1` detectors per
//! round, 100 rounds, with each detector flipping independently at physical
//! error rate `p`.
//!
//! Usage:
//!     cargo run --release --bin sliding_window_benchmark

use std::hint::black_box;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nqpu_metal::sliding_window_decoder::{
    SlidingWindowDecoder, SyndromeRound, WindowInnerDecoder, WindowResult,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Code distances to benchmark.
const DISTANCES: &[usize] = &[3, 5, 7, 9, 11, 13, 15, 17, 19, 21];

/// Number of syndrome measurement rounds per trial.
const NUM_ROUNDS: usize = 100;

/// Physical error rate for the distance sweep.
const DEFAULT_ERROR_RATE: f64 = 0.001;

/// Error rates for the error-rate sweep (at d=5).
const ERROR_RATES: &[f64] = &[0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01];

/// RNG seed for reproducibility.
const RNG_SEED: u64 = 42;

/// Number of warmup iterations before timing.
const WARMUP_RUNS: usize = 1;

/// Number of timed iterations to average over.
const MEASURE_RUNS: usize = 3;

// ---------------------------------------------------------------------------
// Syndrome generation
// ---------------------------------------------------------------------------

/// Generate `num_rounds` syndrome rounds for a surface code with `num_detectors`
/// detectors, where each detector flips with probability `error_rate` per round.
fn generate_syndrome_data(
    num_detectors: usize,
    num_rounds: usize,
    error_rate: f64,
    rng: &mut StdRng,
) -> Vec<SyndromeRound> {
    (0..num_rounds)
        .map(|round_id| {
            let syndrome: Vec<bool> = (0..num_detectors)
                .map(|_| rng.gen_bool(error_rate.min(1.0)))
                .collect();
            SyndromeRound {
                round_id,
                syndrome,
                timestamp: round_id as f64 * 1e-6, // 1 us per round
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct DecoderBenchResult {
    decoder_name: String,
    code_distance: usize,
    num_detectors: usize,
    error_rate: f64,
    num_rounds: usize,
    window_size: usize,
    slide_step: usize,
    total_defects: usize,
    total_matches: usize,
    total_windows: usize,
    avg_window_time_us: f64,
    total_time_us: f64,
    throughput_rounds_per_sec: f64,
}

impl DecoderBenchResult {
    fn header_distance_sweep() {
        println!(
            "{:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>12} {:>14} {:>14}",
            "d",
            "det",
            "decoder",
            "windows",
            "defects",
            "matches",
            "avg_win_us",
            "total_us",
            "rounds/sec"
        );
        println!("{}", "-".repeat(104));
    }

    fn header_error_sweep() {
        println!(
            "{:>8} {:>8} {:>10} {:>10} {:>8} {:>12} {:>14} {:>14}",
            "p",
            "decoder",
            "windows",
            "defects",
            "matches",
            "avg_win_us",
            "total_us",
            "rounds/sec"
        );
        println!("{}", "-".repeat(98));
    }

    fn print_distance_row(&self) {
        println!(
            "{:>4} {:>6} {:>8} {:>10} {:>10} {:>8} {:>12.1} {:>14.1} {:>14.0}",
            self.code_distance,
            self.num_detectors,
            self.decoder_name,
            self.total_windows,
            self.total_defects,
            self.total_matches,
            self.avg_window_time_us,
            self.total_time_us,
            self.throughput_rounds_per_sec,
        );
    }

    fn print_error_row(&self) {
        println!(
            "{:>8.4} {:>8} {:>10} {:>10} {:>8} {:>12.1} {:>14.1} {:>14.0}",
            self.error_rate,
            self.decoder_name,
            self.total_windows,
            self.total_defects,
            self.total_matches,
            self.avg_window_time_us,
            self.total_time_us,
            self.throughput_rounds_per_sec,
        );
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Run the sliding window decoder on the given syndrome data and return metrics.
fn bench_decoder(
    code_distance: usize,
    decoder_type: WindowInnerDecoder,
    rounds: &[SyndromeRound],
    error_rate: f64,
) -> DecoderBenchResult {
    let num_detectors = rounds.first().map(|r| r.syndrome.len()).unwrap_or(0);
    let window_size = (2 * code_distance).max(2);
    let slide_step = code_distance.max(1);

    let decoder_name = match decoder_type {
        WindowInnerDecoder::Greedy => "Greedy".to_string(),
        WindowInnerDecoder::UnionFind => "UFind".to_string(),
    };

    // Warmup
    for _ in 0..WARMUP_RUNS {
        let mut dec =
            SlidingWindowDecoder::new(window_size, slide_step, code_distance, decoder_type);
        for r in rounds.iter() {
            dec.push_round(r.clone());
            if dec.ready() {
                let _ = black_box(dec.decode_window());
            }
        }
        let _ = black_box(dec.flush());
    }

    // Timed runs
    let mut best_total_us = f64::MAX;
    let mut best_results: Vec<WindowResult> = Vec::new();

    for _ in 0..MEASURE_RUNS {
        let mut dec =
            SlidingWindowDecoder::new(window_size, slide_step, code_distance, decoder_type);
        let mut window_results: Vec<WindowResult> = Vec::new();

        let start = Instant::now();

        for r in rounds.iter() {
            dec.push_round(r.clone());
            while dec.ready() {
                window_results.push(dec.decode_window());
            }
        }
        // Flush remaining
        window_results.extend(dec.flush());

        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        if elapsed_us < best_total_us {
            best_total_us = elapsed_us;
            best_results = window_results;
        }
    }

    let total_windows = best_results.len();
    let total_defects: usize = best_results.iter().map(|w| w.defects_in_window).sum();
    let total_matches: usize = best_results.iter().map(|w| w.matches_found).sum();
    let avg_window_time_us = if total_windows > 0 {
        best_results.iter().map(|w| w.decode_time_us).sum::<f64>() / total_windows as f64
    } else {
        0.0
    };

    let committed_rounds: usize = best_results
        .iter()
        .map(|w| w.committed_rounds.len())
        .sum();
    let throughput = if best_total_us > 0.0 {
        committed_rounds as f64 / (best_total_us / 1_000_000.0)
    } else {
        0.0
    };

    DecoderBenchResult {
        decoder_name,
        code_distance,
        num_detectors,
        error_rate,
        num_rounds: rounds.len(),
        window_size,
        slide_step,
        total_defects,
        total_matches,
        total_windows,
        avg_window_time_us,
        total_time_us: best_total_us,
        throughput_rounds_per_sec: throughput,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("================================================================");
    println!("  Sliding Window QEC Decoder Benchmark");
    println!("  nQPU-Metal — Surface Code Syndrome Decoding");
    println!("================================================================");
    println!();
    println!("Configuration:");
    println!("  Rounds per trial:   {}", NUM_ROUNDS);
    println!("  Default error rate: {}", DEFAULT_ERROR_RATE);
    println!("  Warmup runs:        {}", WARMUP_RUNS);
    println!("  Measured runs:      {}", MEASURE_RUNS);
    println!("  RNG seed:           {}", RNG_SEED);
    println!();

    // -----------------------------------------------------------------------
    // Part 1: Distance sweep at fixed error rate
    // -----------------------------------------------------------------------
    println!("================================================================");
    println!("  Part 1: Code Distance Sweep (p = {})", DEFAULT_ERROR_RATE);
    println!("  Window size = 2*d, slide step = d");
    println!("================================================================");
    println!();

    DecoderBenchResult::header_distance_sweep();

    let mut all_results: Vec<DecoderBenchResult> = Vec::new();

    for &d in DISTANCES {
        let num_detectors = d * d - 1;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let rounds = generate_syndrome_data(num_detectors, NUM_ROUNDS, DEFAULT_ERROR_RATE, &mut rng);

        for decoder_type in &[WindowInnerDecoder::Greedy, WindowInnerDecoder::UnionFind] {
            let result = bench_decoder(d, *decoder_type, &rounds, DEFAULT_ERROR_RATE);
            result.print_distance_row();
            all_results.push(result);
        }
    }

    println!();

    // Print speedup comparison
    println!("Distance Sweep — UnionFind vs Greedy Speedup:");
    println!("{:>4} {:>14} {:>14} {:>10}", "d", "Greedy r/s", "UFind r/s", "Speedup");
    println!("{}", "-".repeat(48));

    for chunk in all_results.chunks(2) {
        if chunk.len() == 2 {
            let greedy = &chunk[0];
            let ufind = &chunk[1];
            let speedup = if greedy.throughput_rounds_per_sec > 0.0 {
                ufind.throughput_rounds_per_sec / greedy.throughput_rounds_per_sec
            } else {
                0.0
            };
            println!(
                "{:>4} {:>14.0} {:>14.0} {:>9.2}x",
                greedy.code_distance,
                greedy.throughput_rounds_per_sec,
                ufind.throughput_rounds_per_sec,
                speedup,
            );
        }
    }

    println!();

    // -----------------------------------------------------------------------
    // Part 2: Error rate sweep at d=5
    // -----------------------------------------------------------------------
    let sweep_d = 5;
    let sweep_detectors = sweep_d * sweep_d - 1;

    println!("================================================================");
    println!(
        "  Part 2: Error Rate Sweep (d = {}, {} detectors)",
        sweep_d, sweep_detectors
    );
    println!("  Window size = {}, slide step = {}", 2 * sweep_d, sweep_d);
    println!("================================================================");
    println!();

    DecoderBenchResult::header_error_sweep();

    let mut error_results: Vec<DecoderBenchResult> = Vec::new();

    for &p in ERROR_RATES {
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let rounds = generate_syndrome_data(sweep_detectors, NUM_ROUNDS, p, &mut rng);

        for decoder_type in &[WindowInnerDecoder::Greedy, WindowInnerDecoder::UnionFind] {
            let result = bench_decoder(sweep_d, *decoder_type, &rounds, p);
            result.print_error_row();
            error_results.push(result);
        }
    }

    println!();

    // Print error rate speedup comparison
    println!("Error Rate Sweep — UnionFind vs Greedy Speedup:");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "p", "G defects", "G r/s", "UF r/s", "Speedup"
    );
    println!("{}", "-".repeat(54));

    for chunk in error_results.chunks(2) {
        if chunk.len() == 2 {
            let greedy = &chunk[0];
            let ufind = &chunk[1];
            let speedup = if greedy.throughput_rounds_per_sec > 0.0 {
                ufind.throughput_rounds_per_sec / greedy.throughput_rounds_per_sec
            } else {
                0.0
            };
            println!(
                "{:>8.4} {:>10} {:>10.0} {:>10.0} {:>9.2}x",
                greedy.error_rate,
                greedy.total_defects,
                greedy.throughput_rounds_per_sec,
                ufind.throughput_rounds_per_sec,
                speedup,
            );
        }
    }

    println!();
    println!("================================================================");
    println!("  Benchmark complete.");
    println!("================================================================");
}
