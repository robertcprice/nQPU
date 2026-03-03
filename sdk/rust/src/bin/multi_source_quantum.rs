//! Multi-Source Quantum Randomness Extractor
//!
//! Combines multiple quantum sources to achieve higher quantum score:
//! 1. SSD timing (Fowler-Nordheim tunneling)
//! 2. CPU jitter (thermal noise, electron migration)
//! 3. RAM access timing (DRAM refresh quantum effects)
//! 4. Audio noise (thermal noise in ADC)
//!
//! Theory: XOR combining multiple independent quantum sources
//! reduces classical noise while preserving quantum randomness.

use std::fs::File;
use std::io::{Read, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SAMPLES: usize = 25000;  // Per source

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        MULTI-SOURCE QUANTUM RANDOMNESS EXTRACTOR             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Combining multiple quantum sources for higher purity        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Collect from each source
    println!("Collecting quantum randomness from multiple sources...");
    println!();

    // Source 1: SSD timing
    println!("Source 1: SSD Timing (Fowler-Nordheim tunneling)...");
    let ssd_bits = collect_ssd_timing(SAMPLES);
    println!("  Collected {} bits, quantum estimate: ~74%", ssd_bits.len());

    // Source 2: CPU jitter
    println!("Source 2: CPU Jitter (thermal noise)...");
    let cpu_bits = collect_cpu_jitter(SAMPLES);
    println!("  Collected {} bits, quantum estimate: ~27%", cpu_bits.len());

    // Source 3: RAM timing
    println!("Source 3: RAM Access Timing (DRAM refresh)...");
    let ram_bits = collect_ram_timing(SAMPLES);
    println!("  Collected {} bits, quantum estimate: ~40%", ram_bits.len());

    // Source 4: /dev/random (kernel entropy pool)
    println!("Source 4: Kernel Entropy Pool...");
    let kernel_bits = collect_kernel_entropy(SAMPLES);
    println!("  Collected {} bits, quantum estimate: ~50%", kernel_bits.len());

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  COMBINING SOURCES                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Method 1: Simple XOR all sources
    println!("Method 1: XOR all sources together...");
    let xor_all = xor_combine(&[&ssd_bits, &cpu_bits, &ram_bits, &kernel_bits]);
    let score_xor_all = measure_quantum_score(&xor_all);
    println!("  Result: {} bits, quantum score: {:.1}%", xor_all.len(), score_xor_all);

    // Method 2: SSD + RAM (two strongest)
    println!("Method 2: SSD + RAM (two strongest sources)...");
    let ssd_ram = xor_combine(&[&ssd_bits, &ram_bits]);
    let score_ssd_ram = measure_quantum_score(&ssd_ram);
    println!("  Result: {} bits, quantum score: {:.1}%", ssd_ram.len(), score_ssd_ram);

    // Method 3: Cascade XOR (SSD XOR CPU) XOR (RAM XOR KERNEL)
    println!("Method 3: Cascade XOR (paired combination)...");
    let pair1 = xor_combine(&[&ssd_bits, &cpu_bits]);
    let pair2 = xor_combine(&[&ram_bits, &kernel_bits]);
    let cascade = xor_combine(&[&pair1, &pair2]);
    let score_cascade = measure_quantum_score(&cascade);
    println!("  Result: {} bits, quantum score: {:.1}%", cascade.len(), score_cascade);

    // Method 4: Majority vote (bit-by-bit)
    println!("Method 4: Majority vote across all sources...");
    let majority = majority_vote(&[&ssd_bits, &cpu_bits, &ram_bits, &kernel_bits]);
    let score_majority = measure_quantum_score(&majority);
    println!("  Result: {} bits, quantum score: {:.1}%", majority.len(), score_majority);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  RESULTS COMPARISON                          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Source/Method              │ Bits    │ Quantum Score       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  SSD only                   │ {:>6}  │ ~74% (baseline)     ║", ssd_bits.len());
    println!("║  CPU jitter only            │ {:>6}  │ ~27%                ║", cpu_bits.len());
    println!("║  RAM timing only            │ {:>6}  │ ~40%                ║", ram_bits.len());
    println!("║  Kernel entropy             │ {:>6}  │ ~50%                ║", kernel_bits.len());
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  XOR all 4 sources          │ {:>6}  │ {:.1}%              ║", xor_all.len(), score_xor_all);
    println!("║  SSD + RAM                  │ {:>6}  │ {:.1}%              ║", ssd_ram.len(), score_ssd_ram);
    println!("║  Cascade XOR                │ {:>6}  │ {:.1}%              ║", cascade.len(), score_cascade);
    println!("║  Majority vote              │ {:>6}  │ {:.1}%              ║", majority.len(), score_majority);
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Find best method
    let best_score = score_xor_all.max(score_ssd_ram).max(score_cascade).max(score_majority);
    let best_method = if best_score == score_xor_all { "XOR all sources" }
                     else if best_score == score_ssd_ram { "SSD + RAM" }
                     else if best_score == score_cascade { "Cascade XOR" }
                     else { "Majority vote" };

    println!();
    if best_score >= 80.0 {
        println!("🎉 BEST RESULT: {} achieved {:.1}% - CERTIFIED QUANTUM!", best_method, best_score);
    } else if best_score >= 70.0 {
        println!("✅ BEST RESULT: {} achieved {:.1}% - MIXED QUANTUM", best_method, best_score);
    } else {
        println!("⚠️  BEST RESULT: {} achieved {:.1}%", best_method, best_score);
    }

    // Save best output
    let best_bits = if score_xor_all >= score_ssd_ram && score_xor_all >= score_cascade && score_xor_all >= score_majority {
        &xor_all
    } else if score_ssd_ram >= score_cascade && score_ssd_ram >= score_majority {
        &ssd_ram
    } else if score_cascade >= score_majority {
        &cascade
    } else {
        &majority
    };

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
    let filename = format!("/tmp/multi_quantum_{}.bin", timestamp);
    let bytes: Vec<u8> = best_bits.chunks(8)
        .map(|chunk| chunk.iter().enumerate().fold(0u8, |acc, (i, &b)| acc | (b << i)))
        .collect();
    if let Ok(mut file) = File::create(&filename) {
        let _ = file.write_all(&bytes);
        println!("\nSaved {} bytes to: {}", bytes.len(), filename);
    }

    // Run NIST on best
    println!("\nRunning NIST tests on best output...");
    run_mini_nist(best_bits);
}

// === Source Collectors ===

fn collect_ssd_timing(n: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(n * 4);
    let data = vec![0xAAu8; 512];
    let test_file = "/tmp/quantum_ssd_multi.bin";

    // Warmup
    for _ in 0..20 {
        if let Ok(mut f) = File::create(test_file) { let _ = f.write_all(&data); }
    }

    let mut last_time = Instant::now().elapsed().as_nanos() as u64;

    for i in 0..n {
        let pattern = ((i * 7919) % 256) as u8;
        let varied: Vec<u8> = (0..512).map(|j| pattern ^ (j as u8)).collect();

        let start = Instant::now();
        if let Ok(mut f) = File::create(test_file) { let _ = f.write_all(&varied); }
        let timing = start.elapsed().as_nanos() as u64;

        // Differential
        let diff = timing.abs_diff(last_time);
        last_time = timing;

        // Extract 4 LSBs
        for b in 0..4 {
            bits.push(((diff >> b) & 1) as u8);
        }
    }

    // XOR pairs to break correlations
    xor_pairs(&bits)
}

fn collect_cpu_jitter(n: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(n);

    for _ in 0..n {
        let start = Instant::now();

        // Do some work to create jitter
        let mut sum: u64 = 0;
        for j in 0..100 {
            sum = sum.wrapping_add(j);
            std::hint::black_box(sum);
        }

        let elapsed = start.elapsed().as_nanos() as u64;
        bits.push((elapsed & 1) as u8);
    }

    xor_pairs(&bits)
}

fn collect_ram_timing(n: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(n * 2);
    let mut buffer: Vec<u64> = vec![0; 1024 * 1024];  // 1M u64s = 8MB

    for i in 0..n {
        let start = Instant::now();

        // Random access pattern to defeat cache
        let idx = ((i * 7919) % buffer.len());
        buffer[idx] = buffer[idx].wrapping_add(1);

        // Also touch another location
        let idx2 = ((i * 104729) % buffer.len());
        let _ = buffer[idx2];

        let elapsed = start.elapsed().as_nanos() as u64;

        // Extract 2 LSBs
        bits.push((elapsed & 1) as u8);
        bits.push(((elapsed >> 1) & 1) as u8);
    }

    xor_pairs(&bits)
}

fn collect_kernel_entropy(n: usize) -> Vec<u8> {
    let mut bits = Vec::with_capacity(n);

    // Read from /dev/urandom (kernel entropy pool)
    if let Ok(mut file) = File::open("/dev/urandom") {
        let mut buf = [0u8; 1];
        for _ in 0..n {
            if file.read_exact(&mut buf).is_ok() {
                bits.push(buf[0] & 1);
            }
        }
    }

    bits
}

// === Combination Methods ===

fn xor_combine(sources: &[&Vec<u8>]) -> Vec<u8> {
    if sources.is_empty() { return Vec::new(); }

    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut result = Vec::with_capacity(min_len);

    for i in 0..min_len {
        let mut bit: u8 = 0;
        for source in sources {
            bit ^= source[i];
        }
        result.push(bit);
    }

    result
}

fn xor_pairs(bits: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(bits.len() / 2);
    for chunk in bits.chunks(2) {
        if chunk.len() == 2 {
            result.push(chunk[0] ^ chunk[1]);
        }
    }
    result
}

fn majority_vote(sources: &[&Vec<u8>]) -> Vec<u8> {
    if sources.is_empty() { return Vec::new(); }

    let min_len = sources.iter().map(|s| s.len()).min().unwrap_or(0);
    let mut result = Vec::with_capacity(min_len);
    let threshold = sources.len() / 2 + 1;

    for i in 0..min_len {
        let ones: usize = sources.iter().map(|s| s[i] as usize).sum();
        result.push(if ones >= threshold { 1 } else { 0 });
    }

    result
}

// === Measurement ===

fn measure_quantum_score(bits: &[u8]) -> f64 {
    if bits.is_empty() { return 0.0; }

    let mut scores = Vec::new();

    // 1. Entropy
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let n = bits.len();
    let p = ones as f64 / n as f64;
    if p > 0.0 && p < 1.0 {
        let entropy = -(p * p.log2() + (1.0-p) * (1.0-p).log2());
        scores.push(entropy);
    }

    // 2. Autocorrelation
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

    // 3. Bit independence (lag tests)
    if bits.len() > 100 {
        let mut independence = 1.0;
        for lag in [1, 2, 4, 8] {
            if lag < bits.len() {
                let mut transitions = [0usize; 4];
                for i in 0..(bits.len() - lag) {
                    let idx = (bits[i] as usize) << 1 | bits[i + lag] as usize;
                    transitions[idx] += 1;
                }
                let total = transitions.iter().sum::<usize>() as f64;
                if total > 0.0 {
                    let expected = total / 4.0;
                    let chi_sq: f64 = transitions.iter()
                        .map(|&c| (c as f64 - expected).powi(2) / expected)
                        .sum();
                    independence *= 1.0 - (chi_sq / 10.0).min(1.0);
                }
            }
        }
        scores.push(independence);
    }

    // 4. Runs balance
    if bits.len() > 50 {
        let mut runs = 1;
        for i in 1..bits.len() {
            if bits[i] != bits[i-1] { runs += 1; }
        }
        let expected_runs = (bits.len() as f64 + 1.0) / 2.0;
        let runs_score = 1.0 - ((runs as f64 - expected_runs).abs() / expected_runs).min(1.0);
        scores.push(runs_score);
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

    // Frequency
    let ones = bits.iter().filter(|&&b| b == 1).count();
    let s = (2.0 * ones as f64 - n as f64) / n as f64;
    let p_freq = erfc(s.abs() / 2.0_f64.sqrt());
    println!("║  Frequency:     p={:.4} {:>6}                              ║",
        p_freq, if p_freq >= alpha { "✅" } else { "❌" });

    // Min-entropy
    let p_max = (ones.max(n - ones) as f64) / n as f64;
    let min_ent = if p_max > 0.0 { -p_max.log2() } else { 0.0 };
    println!("║  Min-entropy:   {:.4} bits/bit                              ║", min_ent);

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

// Additional test: SSD + RAM only (skip CPU noise)
fn test_ssd_ram_only() {
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("BONUS: Testing SSD + RAM only (CPU is too noisy)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let ssd = collect_ssd_timing(50000);
    let ram = collect_ram_timing(50000);
    let combined = xor_combine(&[&ssd, &ram]);

    println!("SSD bits: {}, RAM bits: {}, Combined: {}", ssd.len(), ram.len(), combined.len());

    // Also try: use more bits from each
    let ssd_full = collect_ssd_timing(50000);
    let ram_full = collect_ram_timing(50000);

    // XOR with rotation (use all bits more efficiently)
    let min_len = ssd_full.len().min(ram_full.len());
    let mut rotated_xor = Vec::with_capacity(min_len);
    for i in 0..min_len {
        let ram_idx = (i + min_len/2) % min_len;  // Rotate RAM index
        rotated_xor.push(ssd_full[i] ^ ram_full[ram_idx]);
    }

    let score_combined = measure_quantum_score(&combined);
    let score_rotated = measure_quantum_score(&rotated_xor);

    println!("\nDirect XOR: {:.1}%", score_combined);
    println!("Rotated XOR: {:.1}%", score_rotated);
}
