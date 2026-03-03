//! OpenEntropy Integration - Use OpenEntropy as a quantum RNG source
//!
//! This binary integrates OpenEntropy's 47 entropy sources with our
//! quantum randomness measurement system.
//!
//! USAGE:
//!   cargo run --release --bin openentropy_source
//!   cargo run --release --bin openentropy_source -- --raw  # Raw mode (no conditioning)
//!
//! OpenEntropy Sources (Grade A - highest min-entropy):
//!   - audio_pll_timing: Audio PLL clock jitter from thermal perturbation
//!   - counter_beat: Two-oscillator beat (CPU crystal vs audio PLL)
//!   - display_pll: Display PLL phase noise from pixel clock domain
//!   - pcie_pll: PCIe PHY PLL jitter from Thunderbolt/PCIe crossing
//!   - dvfs_race: Cross-core DVFS frequency race
//!   - thread_lifecycle: Thread create/join scheduling jitter
//!   - dispatch_queue: GCD dispatch queue scheduling jitter
//!   - gpu_divergence: GPU shader thread divergence entropy
//!
//! Physics Note:
//!   These sources tap into Johnson-Nyquist thermal noise, clock phase noise,
//!   and oscillator beat frequencies. While some components are quantum
//!   (thermal noise from electron motion), only Bell inequality tests can
//!   CERTIFY quantum randomness - and those require entangled photon pairs.

use std::process::Command;
use std::fs::File;
use std::io::{Read, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SAMPLES: usize = 50000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw_mode = args.contains(&"--raw".to_string());

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        OPENENTROPY QUANTUM RANDOMNESS SOURCE                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    if raw_mode {
        println!("║  Mode: RAW (no conditioning) - preserves quantum signature   ║");
    } else {
        println!("║  Mode: SHA-256 conditioned (cryptographic quality)          ║");
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Check if OpenEntropy is available
    let openentropy_path = std::env::var("OPENENTROPY_BIN").unwrap_or_else(|_| "openentropy".to_string());

    println!("Checking OpenEntropy availability...");
    let check = Command::new(&openentropy_path)
        .arg("--version")
        .output();

    if check.is_err() {
        println!("❌ OpenEntropy not found at: {}", &openentropy_path);
        println!("   Set OPENENTROPY_BIN environment variable to the openentropy binary path");
        println!("   or ensure 'openentropy' is in your PATH");
        return;
    } else {
        println!("✅ OpenEntropy available");
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("COLLECTING ENTROPY FROM OPENENTROPY");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // Collect entropy using OpenEntropy
    let conditioning = if raw_mode { "raw" } else { "sha256" };
    let start = Instant::now();

    let output = Command::new(&openentropy_path)
        .args([
            "stream",
            "--format", "raw",
            "--bytes", &format!("{}", SAMPLES / 8),  // Convert bits to bytes
            "--conditioning", conditioning,
        ])
        .output()
        .expect("Failed to run OpenEntropy");

    let elapsed = start.elapsed();
    let bytes = output.stdout;

    println!("Collected {} bytes in {:?}", bytes.len(), elapsed);
    println!("Throughput: {:.1} KB/s", bytes.len() as f64 / elapsed.as_secs_f64() / 1024.0);

    // Convert to bits for analysis
    let bits: Vec<u8> = bytes.iter()
        .flat_map(|&b| (0..8).map(move |i| ((b >> i) & 1) as u8))
        .collect();

    println!("Converted to {} bits", bits.len());
    println!();

    // Measure quality
    println!("═══════════════════════════════════════════════════════════════════");
    println!("QUALITY MEASUREMENT");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    let score = measure_quantum_score(&bits);
    println!("Statistical quality score: {:.1}%", score);

    // Shannon entropy
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let n = bits.len();
    let p = ones as f64 / n as f64;
    let shannon = if p > 0.0 && p < 1.0 {
        -(p * p.log2() + (1.0-p) * (1.0-p).log2())
    } else {
        0.0
    };
    println!("Shannon entropy: {:.4} / 1.0 bits/bit", shannon);

    // Min-entropy
    let p_max = ones.max(n - ones) as f64 / n as f64;
    let min_ent = if p_max > 0.0 { -p_max.log2() } else { 0.0 };
    println!("Min-entropy H∞:  {:.4} / 1.0 bits/bit", min_ent);

    // Run NIST-like tests
    println!();
    run_mini_nist(&bits);

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("QUANTUM CERTIFICATION STATUS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    if raw_mode {
        println!("✅ RAW MODE - Preserves original noise characteristics");
        println!();
        println!("⚠️  IMPORTANT: Statistical tests CANNOT prove quantum randomness!");
        println!();
        println!("   What CAN certify quantum randomness:");
        println!("   ┌─────────────────────────────────────────────────────────────┐");
        println!("   │ 1. Bell Inequality Tests (device-independent)             │");
        println!("   │    - Requires entangled photon pairs                      │");
        println!("   │    - Certifies through non-local correlations             │");
        println!("   │    - NIST CURBy beacon uses this method                   │");
        println!("   │                                                            │");
        println!("   │ 2. Physics Arguments (what we have)                       │");
        println!("   │    - Thermal noise: Johnson-Nyquist (quantum at electron  │");
        println!("   │      level, but classical at measurement scale)           │");
        println!("   │    - Oscillator beats: phase noise from thermal motion    │");
        println!("   │    - DRAM timing: quantum tunneling in transistors        │");
        println!("   └─────────────────────────────────────────────────────────────┘");
        println!();
        println!("   Consumer hardware entropy sources are BELIEVED to be quantum");
        println!("   based on physics, but CANNOT be CERTIFIED as quantum without");
        println!("   specialized equipment (Bell test apparatus).");
    } else {
        println!("⚠️  SHA-256 CONDITIONED OUTPUT");
        println!();
        println!("   SHA-256 is a DETERMINISTIC cryptographic function.");
        println!("   It makes ANY input (quantum or classical) look uniformly random.");
        println!("   The {:.1}% score reflects hash quality, not quantumness.", score);
    }

    // Save output
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let filename = format!("/tmp/openentropy_{}_{}.bin",
        if raw_mode { "raw" } else { "conditioned" },
        timestamp
    );
    if let Ok(mut file) = File::create(&filename) {
        let _ = file.write_all(&bytes);
        println!();
        println!("Saved {} bytes to: {}", bytes.len(), filename);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("BEST GRADE A SOURCES (by min-entropy H∞)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("  Source               H∞      Physics");
    println!("  ────────────────────────────────────────────────────────────────");
    println!("  audio_pll_timing    6.31    Audio PLL thermal noise");
    println!("  thread_lifecycle    6.22    Kernel scheduling jitter");
    println!("  dvfs_race           6.20    Cross-core voltage races");
    println!("  counter_beat        6.17    Two-oscillator beat (CPU vs Audio)");
    println!("  dispatch_queue      6.17    GCD scheduling jitter");
    println!("  hash_timing         6.18    Microarchitectural timing");
    println!("  gpu_divergence      6.15    GPU thread divergence");
    println!("  display_pll         6.15    Display PLL phase noise");
    println!("  compression_timing  6.11    Data-dependent timing");
    println!("  pcie_pll            6.09    PCIe PHY PLL jitter");
}

fn measure_quantum_score(bits: &[u8]) -> f64 {
    if bits.is_empty() { return 0.0; }

    let mut scores = Vec::new();

    // 1. Shannon entropy
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let n = bits.len();
    let p = ones as f64 / n as f64;
    if p > 0.0 && p < 1.0 {
        let entropy = -(p * p.log2() + (1.0-p) * (1.0-p).log2());
        scores.push(entropy);
    }

    // 2. Autocorrelation (lag-1)
    if bits.len() > 100 {
        let bits_f: Vec<f64> = bits.iter().map(|&b| b as f64 * 2.0 - 1.0).collect();
        let mean: f64 = bits_f.iter().sum::<f64>() / bits_f.len() as f64;
        let var: f64 = bits_f.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / bits_f.len() as f64;

        if var > 0.0 {
            let mut sum = 0.0;
            for i in 0..(bits_f.len() - 1) {
                sum += (bits_f[i] - mean) * (bits_f[i + 1] - mean);
            }
            let autocorr = (sum / (bits_f.len() - 1) as f64) / var;
            scores.push(1.0 - autocorr.abs().min(1.0));
        }
    }

    // 3. Runs test
    if bits.len() > 50 {
        let mut runs = 1;
        for i in 1..bits.len() {
            if bits[i] != bits[i-1] { runs += 1; }
        }
        let expected_runs = (bits.len() as f64 + 1.0) / 2.0;
        let runs_score = 1.0 - ((runs as f64 - expected_runs).abs() / expected_runs).min(1.0);
        scores.push(runs_score);
    }

    // 4. Bit independence (chi-square)
    if bits.len() > 100 {
        let mut transitions = [0usize; 4];
        for i in 0..(bits.len() - 1) {
            let idx = (bits[i] as usize) << 1 | bits[i + 1] as usize;
            transitions[idx] += 1;
        }
        let total = transitions.iter().sum::<usize>() as f64;
        let expected = total / 4.0;
        let chi_sq: f64 = transitions.iter()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum();
        scores.push(1.0 - (chi_sq / 10.0).min(1.0));
    }

    if scores.is_empty() {
        0.0
    } else {
        (scores.iter().sum::<f64>() / scores.len() as f64) * 100.0
    }
}

fn run_mini_nist(bits: &[u8]) {
    let n = bits.len();
    let alpha = 0.01;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              MINI NIST STATISTICAL TESTS                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    // Frequency test
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let s = (2.0 * ones as f64 - n as f64) / n as f64;
    let p_freq = erfc(s.abs() / 2.0_f64.sqrt());
    println!("║  Frequency:     p={:.4} {:>6}                              ║",
        p_freq, if p_freq >= alpha { "✅" } else { "❌" });

    // Runs test
    let mut runs = 1;
    for i in 1..bits.len() {
        if bits[i] != bits[i-1] { runs += 1; }
    }
    let expected = (bits.len() as f64 + 1.0) / 2.0;
    let p_runs = erfc(((runs as f64 - expected).abs() - 0.5) / (bits.len() as f64).sqrt());
    println!("║  Runs:          p={:.4} {:>6}                              ║",
        p_runs, if p_runs >= alpha { "✅" } else { "❌" });

    // Serial test (bit pairs)
    let mut pairs = [0usize; 4];
    for i in 0..(bits.len() - 1) {
        pairs[(bits[i] as usize) << 1 | bits[i+1] as usize] += 1;
    }
    let total = pairs.iter().sum::<usize>() as f64;
    let chi_sq: f64 = pairs.iter()
        .map(|&c| (c as f64 - total/4.0).powi(2) / (total/4.0))
        .sum();
    println!("║  Serial (pairs): χ²={:.2}                                  ║", chi_sq);

    println!("╚══════════════════════════════════════════════════════════════╝");
}

fn erfc(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    (1.0 - sign * y).max(0.0).min(2.0)
}
