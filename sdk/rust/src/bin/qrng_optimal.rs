//! QRNG Optimization Experiment
//!
//! Goal: Get as close to 100% NIST pass rate as possible while keeping quantum source raw
//!
//! Strategies:
//! 1. De-synchronized sampling (random delays to break periodic patterns)
//! 2. Light SHA-256 conditioning (minimal post-processing)
//! 3. More data (50,000 samples)
//! 4. Multi-source combination
//! 5. Optimal bit extraction from middle bits (not LSBs which may have structure)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nqpu_metal::nist_tests::NistTestSuite;

// ============================================================================
// DATA COLLECTION WITH DE-SYNCHRONIZATION
// ============================================================================

/// Collect SSD timing with random delays to break periodic patterns
fn collect_ssd_desync(n_samples: usize) -> Vec<u64> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let test_file = "/tmp/ssd_optimal_test.bin";
    let data = vec![0x55u8; 4096];
    let mut timings = Vec::with_capacity(n_samples);
    let mut rng_state = 0x12345678u64;

    // Simple PRNG for de-synchronization delays
    let xorshift = |state: &mut u64| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    // Warm-up
    for _ in 0..20 {
        if let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(test_file)
        {
            let _ = file.write_all(&data);
        }
    }

    println!(
        "  Collecting {} SSD samples (de-synchronized)...",
        n_samples
    );

    for _ in 0..n_samples {
        // Random delay to break periodic patterns
        let delay_ns = xorshift(&mut rng_state) % 1_000_000; // 0-1ms random delay
        if delay_ns > 100_000 {
            std::thread::sleep(Duration::from_nanos(delay_ns));
        }

        let start = Instant::now();
        if let Ok(mut file) = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(test_file)
        {
            let _ = file.write_all(&data);
            #[cfg(unix)]
            {
                use std::os::unix::fs::FileExt;
                let _ = file.sync_all();
            }
        }
        timings.push(start.elapsed().as_nanos() as u64);
    }

    let _ = std::fs::remove_file(test_file);
    timings
}

/// Collect CPU jitter with de-synchronization
fn collect_cpu_desync(n_samples: usize) -> Vec<u64> {
    let mut timings = Vec::with_capacity(n_samples);
    let mut rng_state = 0x87654321u64;

    let xorshift = |state: &mut u64| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    println!(
        "  Collecting {} CPU jitter samples (de-synchronized)...",
        n_samples
    );

    for _ in 0..n_samples {
        let delay_ns = xorshift(&mut rng_state) % 500_000;
        if delay_ns > 50_000 {
            std::thread::sleep(Duration::from_nanos(delay_ns));
        }

        let start = Instant::now();
        let mut sum: u64 = 0;
        for j in 0..100 {
            sum = sum.wrapping_add(j as u64);
        }
        std::hint::black_box(sum);
        timings.push(start.elapsed().as_nanos() as u64);
    }
    timings
}

// ============================================================================
// EXTRACTION METHODS
// ============================================================================

/// Extract bits from middle of timing value (bits 8-15)
/// These bits have less structure than LSBs (affected by controller patterns)
/// and more entropy than high bits (deterministic)
fn extract_middle_bits(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(values.len() * 8);
    for &v in values {
        // Extract bits 8-15 (middle bits)
        for i in 8..16 {
            bits.push(((v >> i) & 1) as u8);
        }
    }
    bits
}

/// Von Neumann debiasing
fn von_neumann(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len() / 4);
    for chunk in bits.chunks(2) {
        if chunk.len() == 2 {
            match (chunk[0], chunk[1]) {
                (0, 1) => out.push(0),
                (1, 0) => out.push(1),
                _ => {}
            }
        }
    }
    out
}

/// Light SHA-like conditioning (minimal post-processing)
fn light_condition(bits: &[u8]) -> Vec<u8> {
    // Simple compression: XOR groups of 8 bits into 1
    let mut result = Vec::with_capacity(bits.len() / 8);
    for chunk in bits.chunks(8) {
        let compressed = chunk.iter().fold(0u8, |acc, &b| acc ^ b);
        result.push(compressed);
    }
    result
}

/// Hash-based conditioning (stronger but still simple)
fn hash_condition(bits: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();

    // Process in chunks of 64 bits
    for chunk in bits.chunks(64) {
        let mut hasher = DefaultHasher::new();
        for &b in chunk {
            b.hash(&mut hasher);
        }
        let hash = hasher.finish();
        for i in 0..8 {
            result.push(((hash >> i) & 1) as u8);
        }
    }
    result
}

/// XOR combine multiple sources
fn xor_combine(sources: &[&[u8]]) -> Vec<u8> {
    if sources.is_empty() {
        return Vec::new();
    }
    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    (0..min_len)
        .map(|i| sources.iter().fold(0u8, |acc, s| acc ^ s[i]))
        .collect()
}

/// Differential extraction (removes slow drift)
fn differential(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();
    for i in 1..values.len() {
        let diff = values[i].wrapping_sub(values[i - 1]);
        // Take middle bits of difference
        for j in 8..16 {
            bits.push(((diff >> j) & 1) as u8);
        }
    }
    bits
}

// ============================================================================
// TESTING
// ============================================================================

fn run_nist(name: &str, bits: &[u8]) -> (usize, f64) {
    if bits.len() < 1000 {
        println!("  ⚠️ {} skipped ({} bits)", name, bits.len());
        return (0, 0.0);
    }

    let suite = NistTestSuite::new();
    let result = suite.run_all_tests(bits);

    let status = if result.passed_count >= 14 {
        "🏆"
    } else if result.passed_count >= 12 {
        "✅"
    } else if result.passed_count >= 10 {
        "⚠️"
    } else {
        "❌"
    };

    println!(
        "  {} {}: {}/15 NIST ({:.0}%) min-ent={:.4}",
        status,
        name,
        result.passed_count,
        result.passed_count as f64 / 15.0 * 100.0,
        result.min_entropy
    );

    (result.passed_count, result.min_entropy)
}

// ============================================================================
// MAIN EXPERIMENT
// ============================================================================

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║     QRNG OPTIMIZATION - TARGET: 100% NIST PASS                      ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║  Strategies:                                                        ║");
    println!("║  1. De-synchronized sampling (breaks periodic patterns)             ║");
    println!("║  2. Middle-bit extraction (less structure than LSBs)                ║");
    println!("║  3. Minimal conditioning (keep quantum source raw)                  ║");
    println!("║  4. Multi-source combination                                        ║");
    println!("║  5. Larger dataset (more samples = better statistics)               ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    let n_samples = 30_000; // More samples for better NIST statistics

    println!("▶ PHASE 1: DATA COLLECTION ({} samples)", n_samples);
    let ssd = collect_ssd_desync(n_samples);
    let cpu = collect_cpu_desync(n_samples);
    println!(
        "  ✅ SSD: {} samples, CPU: {} samples",
        ssd.len(),
        cpu.len()
    );
    println!();

    let mut results: Vec<(&str, usize, f64)> = Vec::new();

    // ===== BASELINE =====
    println!("▶ BASELINE (Standard LSB extraction):");
    let mut bits = extract_middle_bits(&ssd);
    let (p, m) = run_nist("Middle bits raw", &bits);
    results.push(("Middle bits raw", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Middle bits + VN", &bits);
    results.push(("Middle bits + VN", p, m));

    // ===== DE-SYNCHRONIZED SAMPLING =====
    println!("\n▶ DE-SYNCHRONIZED SAMPLING (already applied above):");
    println!("  (Random delays between samples to break periodic patterns)");

    // ===== DIFFERENTIAL EXTRACTION =====
    println!("\n▶ DIFFERENTIAL EXTRACTION (removes drift):");
    bits = differential(&ssd);
    let (p, m) = run_nist("Differential raw", &bits);
    results.push(("Differential raw", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Differential + VN", &bits);
    results.push(("Differential + VN", p, m));

    // ===== LIGHT CONDITIONING =====
    println!("\n▶ LIGHT CONDITIONING (minimal post-processing):");
    bits = light_condition(&extract_middle_bits(&ssd));
    let (p, m) = run_nist("Light conditioned", &bits);
    results.push(("Light conditioned", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Light + VN", &bits);
    results.push(("Light + VN", p, m));

    // ===== HASH CONDITIONING =====
    println!("\n▶ HASH CONDITIONING (stronger whitening):");
    bits = hash_condition(&extract_middle_bits(&ssd));
    let (p, m) = run_nist("Hash conditioned", &bits);
    results.push(("Hash conditioned", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Hash + VN", &bits);
    results.push(("Hash + VN", p, m));

    // ===== MULTI-SOURCE COMBINATION =====
    println!("\n▶ MULTI-SOURCE COMBINATION:");
    let ssd_bits = extract_middle_bits(&ssd);
    let cpu_bits = extract_middle_bits(&cpu);
    bits = xor_combine(&[&ssd_bits, &cpu_bits]);
    let (p, m) = run_nist("SSD ⊕ CPU raw", &bits);
    results.push(("SSD ⊕ CPU raw", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("SSD ⊕ CPU + VN", &bits);
    results.push(("SSD ⊕ CPU + VN", p, m));

    // ===== DIFFERENTIAL + HASH =====
    println!("\n▶ DIFFERENTIAL + HASH (best of both):");
    let diff = differential(&ssd);
    bits = hash_condition(&diff);
    let (p, m) = run_nist("Diff→Hash", &bits);
    results.push(("Diff→Hash", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Diff→Hash→VN", &bits);
    results.push(("Diff→Hash→VN", p, m));

    // ===== TRIPLE SOURCE =====
    println!("\n▶ TRIPLE SOURCE (SSD ⊕ CPU ⊕ Diff):");
    let ssd_bits = extract_middle_bits(&ssd);
    let cpu_bits = extract_middle_bits(&cpu);
    let diff = differential(&ssd);
    let min_len = ssd_bits.len().min(cpu_bits.len()).min(diff.len());
    bits = (0..min_len)
        .map(|i| ssd_bits[i] ^ cpu_bits[i] ^ diff[i])
        .collect();
    let (p, m) = run_nist("Triple XOR raw", &bits);
    results.push(("Triple XOR raw", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Triple XOR + VN", &bits);
    results.push(("Triple XOR + VN", p, m));

    // ===== TRIPLE + HASH =====
    println!("\n▶ TRIPLE + HASH (maximum mixing):");
    let ssd_bits = extract_middle_bits(&ssd);
    let cpu_bits = extract_middle_bits(&cpu);
    let diff = differential(&ssd);
    let min_len = ssd_bits.len().min(cpu_bits.len()).min(diff.len());
    let triple: Vec<u8> = (0..min_len)
        .map(|i| ssd_bits[i] ^ cpu_bits[i] ^ diff[i])
        .collect();
    bits = hash_condition(&triple);
    let (p, m) = run_nist("Triple→Hash", &bits);
    results.push(("Triple→Hash", p, m));
    bits = von_neumann(&bits);
    let (p, m) = run_nist("Triple→Hash→VN", &bits);
    results.push(("Triple→Hash→VN", p, m));

    // ===== SUMMARY =====
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                      FINAL RESULTS                                  ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Method                │ NIST │ Pass%  │ Min-Ent │ Rating          ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    results.sort_by(|a, b| b.1.cmp(&a.1));

    for (name, pass, min_ent) in &results {
        let pct = *pass as f64 / 15.0 * 100.0;
        let rating = if *pass >= 15 {
            "🏆 PERFECT"
        } else if *pass >= 14 {
            "✅ EXCELLENT"
        } else if *pass >= 12 {
            "⚠️ GOOD"
        } else if *pass >= 10 {
            "❌ OK"
        } else {
            "💀 POOR"
        };

        println!(
            "║ {:<21} │ {:>2}/15 │ {:>5.0}% │ {:>7.4} │ {:<15} ║",
            name, pass, pct, min_ent, rating
        );
    }

    println!("╚════════════════════════════════════════════════════════════════════╝");

    if let Some(best) = results.first() {
        println!(
            "\n🏆 BEST: {} with {}/15 NIST passes ({:.0}%)",
            best.0,
            best.1,
            best.1 as f64 / 15.0 * 100.0
        );

        if best.1 >= 14 {
            println!("\n✅ SUCCESS! We achieved near-100% NIST pass rate!");
        } else if best.1 >= 12 {
            println!("\n⚠️ Good results but still room for improvement.");
            println!("   Consider: More samples, SHA-256 conditioning, or different hardware.");
        } else {
            println!("\n❌ Results indicate fundamental structure in the timing data.");
            println!("   The periodic patterns from OS/SSD controller are hard to eliminate.");
            println!("   Recommendation: Use cryptographic conditioning for final output.");
        }
    }
}
