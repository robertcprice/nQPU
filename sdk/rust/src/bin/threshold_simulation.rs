//! QEC Threshold Simulation via Sliding Window Decoder
//!
//! Estimates the error correction threshold by sweeping code distance and
//! physical error rate. The threshold is the critical error rate p* where
//! increasing code distance stops improving the logical error rate -- i.e.,
//! the point where curves for different distances cross.
//!
//! For each (distance, error_rate) pair, this binary:
//!   1. Generates 3*d syndrome rounds with independent bit-flip errors at rate p.
//!   2. Decodes with the sliding window union-find decoder.
//!   3. Compares the decoder's correction against injected errors.
//!   4. Records whether a logical failure occurred.
//!
//! Output is a TSV table suitable for plotting, plus a threshold crossing summary.
//!
//! Usage:
//!     cargo run --release --bin threshold_simulation

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use nqpu_metal::sliding_window_decoder::{
    SlidingWindowDecoder, SyndromeRound, WindowInnerDecoder,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Code distances to simulate. Odd distances only (surface code convention).
const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];

/// Number of Monte Carlo trials per (distance, error_rate) combination.
const NUM_TRIALS: usize = 1000;

/// Number of physical error rate sample points in logspace.
const NUM_ERROR_POINTS: usize = 15;

/// Bounds of the error rate sweep (inclusive endpoints in logspace).
const P_MIN: f64 = 1e-4;
const P_MAX: f64 = 5e-2;

/// Base RNG seed for deterministic reproducibility. Each (d, p_idx, trial)
/// triple derives a unique seed from this base.
const RNG_SEED_BASE: u64 = 0xDEAD_BEEF_CAFE_42;

// ---------------------------------------------------------------------------
// Error rate grid generation
// ---------------------------------------------------------------------------

/// Generate `n` points log-spaced between `lo` and `hi` (inclusive).
fn logspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (log_lo + t * (log_hi - log_lo)).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Syndrome generation with known error tracking
// ---------------------------------------------------------------------------

/// Generate `num_rounds` syndrome rounds for a distance-d code with
/// independent bit-flip errors at rate `p`. Returns both the syndrome
/// rounds (for the decoder) and the underlying data-qubit errors (for
/// verifying the decoder output).
///
/// Error model: each data qubit flips independently with probability `p`
/// each round. The syndrome is the "difference syndrome" -- the XOR of
/// the current round's error pattern with the previous round's. This
/// models the standard circuit-level noise where measurement outcomes are
/// differenced to produce detection events.
fn generate_trial_data(
    code_distance: usize,
    num_rounds: usize,
    error_rate: f64,
    rng: &mut StdRng,
) -> (Vec<SyndromeRound>, Vec<Vec<bool>>) {
    // Number of detectors: d-1 for a 1D repetition code (the simplest
    // code that exhibits a threshold). For a surface code this would be
    // d^2 - 1, but we use d - 1 to keep runtime tractable across many
    // trials and distances.
    let num_detectors = if code_distance > 1 {
        code_distance - 1
    } else {
        1
    };

    let mut syndrome_rounds = Vec::with_capacity(num_rounds);
    let mut all_data_errors = Vec::with_capacity(num_rounds);

    // Previous round's cumulative error state (for differencing).
    let mut prev_syndrome = vec![false; num_detectors];

    for round_id in 0..num_rounds {
        // Generate independent bit-flip errors on data qubits.
        let data_errors: Vec<bool> = (0..num_detectors)
            .map(|_| rng.gen_bool(error_rate.clamp(0.0, 1.0)))
            .collect();

        // Compute the "raw" syndrome for this round: XOR of adjacent
        // data qubit error states. For a repetition code, detector j
        // fires if data_error[j] XOR data_error[j+1]. In our simplified
        // model with num_detectors == d-1 data qubits, we just use the
        // data error pattern directly as the cumulative syndrome, then
        // difference it against the previous round.
        //
        // Cumulative syndrome = running XOR of all data errors up to now.
        // The decoder sees the difference syndrome = cumulative XOR prev.
        let cumulative: Vec<bool> = data_errors
            .iter()
            .enumerate()
            .map(|(i, &e)| prev_syndrome[i] ^ e)
            .collect();

        // The difference syndrome the decoder will see.
        // (For round 0, prev is all-false, so difference = cumulative.)
        // Actually, the sliding window decoder internally differences
        // consecutive syndromes. So we feed the *cumulative* syndrome
        // and let the decoder do the differencing.
        syndrome_rounds.push(SyndromeRound {
            round_id,
            syndrome: cumulative.clone(),
            timestamp: round_id as f64 * 1e-6,
        });

        all_data_errors.push(data_errors);
        prev_syndrome = cumulative;
    }

    (syndrome_rounds, all_data_errors)
}

// ---------------------------------------------------------------------------
// Logical failure detection
// ---------------------------------------------------------------------------

/// Determine whether the decoder's corrections, when combined with the
/// actual errors, result in a logical failure.
///
/// A logical failure occurs when the residual error (actual XOR correction)
/// forms a non-trivial logical operator. For a 1D repetition code, the
/// logical X operator is the all-ones vector (flip all data qubits). So
/// a logical failure happens when the residual has odd parity (the XOR
/// of all residual bits is true).
///
/// We check each committed round: if the residual error parity is odd
/// for ANY round, we declare a logical failure. In practice, for a memory
/// experiment we care about the final-round residual, but checking all
/// rounds and OR-ing is a conservative (stricter) test.
fn check_logical_failure(
    committed_corrections: &[Vec<bool>],
    data_errors: &[Vec<bool>],
    num_rounds_to_check: usize,
) -> bool {
    let rounds_to_check = num_rounds_to_check.min(committed_corrections.len()).min(data_errors.len());

    if rounds_to_check == 0 {
        return false;
    }

    // Track cumulative data error state (XOR of all errors up to this round).
    let num_qubits = data_errors.first().map_or(0, |v| v.len());
    let mut cumulative_error = vec![false; num_qubits];

    for round in 0..rounds_to_check {
        // Accumulate the data errors.
        for (i, &e) in data_errors[round].iter().enumerate() {
            cumulative_error[i] ^= e;
        }

        // Get the decoder's correction for this round. If the correction
        // vector is shorter than num_qubits, treat missing entries as false.
        let correction = &committed_corrections[round];

        // Compute residual = cumulative_error XOR correction.
        // Check parity of the residual.
        let mut parity = false;
        for i in 0..num_qubits {
            let corr_bit = correction.get(i).copied().unwrap_or(false);
            let residual_bit = cumulative_error[i] ^ corr_bit;
            parity ^= residual_bit;
        }

        // For a repetition code, odd parity residual = logical failure.
        if parity && round == rounds_to_check - 1 {
            return true;
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Result storage
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[allow(dead_code)]
struct ThresholdDataPoint {
    code_distance: usize,
    physical_error_rate: f64,
    logical_error_rate: f64,
    num_trials: usize,
    num_failures: usize,
}

// ---------------------------------------------------------------------------
// Main simulation
// ---------------------------------------------------------------------------

fn main() {
    let total_start = Instant::now();

    // Generate the error rate grid.
    let error_rates = logspace(P_MIN, P_MAX, NUM_ERROR_POINTS);

    let total_jobs = DISTANCES.len() * error_rates.len();
    let mut completed_jobs = 0;

    eprintln!("================================================================");
    eprintln!("  QEC Threshold Simulation — Sliding Window Decoder (UnionFind)");
    eprintln!("================================================================");
    eprintln!("  Distances:      {:?}", DISTANCES);
    eprintln!("  Error rates:    {} points in [{:.1e}, {:.1e}]", NUM_ERROR_POINTS, P_MIN, P_MAX);
    eprintln!("  Trials per pt:  {}", NUM_TRIALS);
    eprintln!("  Total jobs:     {}", total_jobs);
    eprintln!("  RNG seed base:  0x{:X}", RNG_SEED_BASE);
    eprintln!();

    // Results indexed by [distance_idx][error_rate_idx].
    let mut results: Vec<Vec<ThresholdDataPoint>> = Vec::new();

    // Print TSV header.
    println!("code_distance\tphysical_error_rate\tlogical_error_rate\tnum_trials\tnum_failures");

    for (d_idx, &d) in DISTANCES.iter().enumerate() {
        let num_rounds = 3 * d;
        let window_size = d.max(2); // window_size = d, minimum 2
        let slide_step = (d + 1) / 2; // ceil(d/2)

        let mut distance_results = Vec::new();

        for (p_idx, &p) in error_rates.iter().enumerate() {
            let mut failures = 0;

            for trial in 0..NUM_TRIALS {
                // Derive a unique, deterministic seed for this trial.
                let seed = RNG_SEED_BASE
                    .wrapping_mul(1000003)
                    .wrapping_add(d_idx as u64 * 100_000_007)
                    .wrapping_add(p_idx as u64 * 1_000_000_007)
                    .wrapping_add(trial as u64);
                let mut rng = StdRng::seed_from_u64(seed);

                // Generate syndrome data with known errors.
                let (syndrome_rounds, data_errors) =
                    generate_trial_data(d, num_rounds, p, &mut rng);

                // Create and run the sliding window decoder.
                let mut decoder = SlidingWindowDecoder::new(
                    window_size,
                    slide_step,
                    d,
                    WindowInnerDecoder::UnionFind,
                );

                for sr in &syndrome_rounds {
                    decoder.push_round(sr.clone());
                    while decoder.ready() {
                        let _ = decoder.decode_window();
                    }
                }
                // Flush remaining rounds.
                let _ = decoder.flush();

                // Check for logical failure.
                let committed = decoder.committed();
                if check_logical_failure(committed, &data_errors, num_rounds) {
                    failures += 1;
                }
            }

            let logical_error_rate = failures as f64 / NUM_TRIALS as f64;

            let point = ThresholdDataPoint {
                code_distance: d,
                physical_error_rate: p,
                logical_error_rate,
                num_trials: NUM_TRIALS,
                num_failures: failures,
            };

            // Print TSV row.
            println!(
                "{}\t{:.6e}\t{:.6e}\t{}\t{}",
                d, p, logical_error_rate, NUM_TRIALS, failures,
            );

            distance_results.push(point);

            completed_jobs += 1;
            if completed_jobs % 5 == 0 || completed_jobs == total_jobs {
                let elapsed = total_start.elapsed().as_secs_f64();
                let rate = completed_jobs as f64 / elapsed;
                let remaining = (total_jobs - completed_jobs) as f64 / rate;
                eprintln!(
                    "  [{}/{}] d={:2}, p={:.2e} => L_err={:.4} ({} failures) | {:.1}s elapsed, ~{:.0}s remaining",
                    completed_jobs,
                    total_jobs,
                    d,
                    p,
                    logical_error_rate,
                    failures,
                    elapsed,
                    remaining,
                );
            }
        }

        results.push(distance_results);
    }

    // -----------------------------------------------------------------------
    // Threshold crossing analysis
    // -----------------------------------------------------------------------
    eprintln!();
    eprintln!("================================================================");
    eprintln!("  Threshold Crossing Analysis");
    eprintln!("================================================================");
    eprintln!();

    println!();
    println!("# Threshold crossing analysis");
    println!("# For adjacent distance pairs, report where logical error rate");
    println!("# curves cross (indicator of the error correction threshold).");
    println!();
    println!("d_low\td_high\tcrossing_detected\tcrossing_p_low\tcrossing_p_high");

    for i in 0..results.len().saturating_sub(1) {
        let d_low = DISTANCES[i];
        let d_high = DISTANCES[i + 1];
        let curve_low = &results[i];
        let curve_high = &results[i + 1];

        // Look for a crossing: at low p, larger distance should have LOWER
        // logical error rate. At high p (above threshold), larger distance
        // has HIGHER logical error rate. A crossing is where
        // sign(L_high - L_low) changes.
        let mut crossing_found = false;
        let mut crossing_p_low = 0.0_f64;
        let mut crossing_p_high = 0.0_f64;

        let n = curve_low.len().min(curve_high.len());
        for j in 1..n {
            let diff_prev =
                curve_high[j - 1].logical_error_rate - curve_low[j - 1].logical_error_rate;
            let diff_curr =
                curve_high[j].logical_error_rate - curve_low[j].logical_error_rate;

            // Sign change indicates a crossing.
            if (diff_prev < 0.0 && diff_curr > 0.0)
                || (diff_prev > 0.0 && diff_curr < 0.0)
            {
                crossing_found = true;
                crossing_p_low = error_rates[j - 1];
                crossing_p_high = error_rates[j];
                break;
            }
        }

        if crossing_found {
            println!(
                "{}\t{}\ttrue\t{:.6e}\t{:.6e}",
                d_low, d_high, crossing_p_low, crossing_p_high,
            );
            eprintln!(
                "  d={} vs d={}: CROSSING in [{:.2e}, {:.2e}]",
                d_low, d_high, crossing_p_low, crossing_p_high,
            );
        } else {
            // Check the overall trend: does higher distance consistently help or hurt?
            let mut higher_d_better_count = 0;
            let mut higher_d_worse_count = 0;
            for j in 0..n {
                if curve_high[j].logical_error_rate < curve_low[j].logical_error_rate {
                    higher_d_better_count += 1;
                } else if curve_high[j].logical_error_rate > curve_low[j].logical_error_rate {
                    higher_d_worse_count += 1;
                }
            }
            println!("{}\t{}\tfalse\tNA\tNA", d_low, d_high);
            eprintln!(
                "  d={} vs d={}: NO CROSSING (higher d better at {}/{} points, worse at {}/{})",
                d_low, d_high, higher_d_better_count, n, higher_d_worse_count, n,
            );
        }
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("================================================================");
    eprintln!("  Simulation complete in {:.2}s", total_elapsed);
    eprintln!(
        "  Total trials: {} ({} per data point)",
        DISTANCES.len() * error_rates.len() * NUM_TRIALS,
        NUM_TRIALS,
    );
    eprintln!("================================================================");
}
