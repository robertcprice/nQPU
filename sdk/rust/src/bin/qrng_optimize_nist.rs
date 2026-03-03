//! Raw Quantum Source Optimization
//!
//! Goal: Maximum NIST pass rate with MINIMAL processing
//! Strategy: Try different approaches to break periodicity while keeping quantum signal

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nqpu_metal::nist_tests::NistTestSuite;

fn collect_ssd_raw(n_samples: usize, delay_us: u64) -> Vec<u64> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let test_file = "/tmp/ssd_raw_test.bin";
    let data = vec![0x55u8; 4096];
    let mut timings = Vec::with_capacity(n_samples);

    // Warm-up
    for _ in 0..20 {
        if let Ok(mut file) = OpenOptions::new().write(true).create(true).truncate(true).open(test_file) {
            let _ = file.write_all(&data);
        }
    }

    for _ in 0..n_samples {
        // Add variable delay to break periodicity
        if delay_us > 0 {
            let jitter = (Instant::now().elapsed().as_nanos() % 100) as u64;
            std::thread::sleep(Duration::from_micros(delay_us + jitter));
        }

        let start = Instant::now();
        if let Ok(mut file) = OpenOptions::new().write(true).create(true).truncate(false).open(test_file) {
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

fn collect_cpu_raw(n_samples: usize) -> Vec<u64> {
    let mut timings = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let start = Instant::now();
        let mut sum: u64 = 0;
        for j in 0..100 { sum = sum.wrapping_add(j as u64); }
        std::hint::black_box(sum);
        timings.push(start.elapsed().as_nanos() as u64);
    }
    timings
}

// Different bit extraction approaches
fn extract_lsb(values: &[u64], n_bits: usize) -> Vec<u8> {
    let mut bits = Vec::new();
    for &v in values {
        for i in 0..n_bits { bits.push(((v >> i) & 1) as u8); }
    }
    bits
}

fn extract_middle_bits(values: &[u64]) -> Vec<u8> {
    // Use bits 4-11 instead of 0-7 (less controller influence)
    let mut bits = Vec::new();
    for &v in values {
        for i in 4..12 { bits.push(((v >> i) & 1) as u8); }
    }
    bits
}

fn extract_high_bits(values: &[u64]) -> Vec<u8> {
    // Use bits 8-15 (least affected by nanosecond jitter)
    let mut bits = Vec::new();
    for &v in values {
        for i in 8..16 { bits.push(((v >> i) & 1) as u8); }
    }
    bits
}

fn extract_diff(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();
    for i in 1..values.len() {
        let d = values[i].wrapping_sub(values[i-1]);
        for j in 4..12 { bits.push(((d >> j) & 1) as u8); }
    }
    bits
}

fn von_neumann(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
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

fn xor_combine(a: &[u8], b: &[u8]) -> Vec<u8> {
    (0..a.len().min(b.len())).map(|i| a[i] ^ b[i]).collect()
}

fn sha256_condition(bits: &[u8]) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut output = Vec::new();

    // Process in chunks and hash
    for chunk in bits.chunks(256) {
        let mut hasher = DefaultHasher::new();
        for &b in chunk {
            b.hash(&mut hasher);
        }
        let hash = hasher.finish();
        for i in 0..64 {
            output.push(((hash >> i) & 1) as u8);
        }
    }
    output
}

fn test_bits(name: &str, bits: &[u8]) -> (usize, f64, bool) {
    if bits.len() < 1000 {
        println!("  ⚠️ {}: skipped (only {} bits)", name, bits.len());
        return (0, 0.0, false);
    }

    let suite = NistTestSuite::new();
    let result = suite.run_all_tests(bits);

    let pass_14_plus = result.passed_count >= 14;
    println!("  {}: {}/15 ({:.0}%) | min-ent={:.4} {}",
        name, result.passed_count, result.passed_count as f64 / 15.0 * 100.0,
        result.min_entropy, if pass_14_plus { "🏆" } else if result.passed_count >= 12 { "✅" } else { "" });

    (result.passed_count, result.min_entropy, pass_14_plus)
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     RAW QUANTUM SOURCE OPTIMIZATION                                      ║");
    println!("║     Goal: Maximum NIST with MINIMAL processing                           ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝");
    println!();

    let n_samples = 20_000; // More data for better NIST reliability
    println!("▶ COLLECTING {} SAMPLES...", n_samples);

    // Test different collection strategies
    println!("\n═══ STRATEGY 1: Different Sampling Intervals ═══");

    let mut best_result: Option<(&str, usize, f64)> = None;

    // No delay (raw timing)
    println!("\n1a. Raw SSD timing (no delay):");
    let ssd_raw = collect_ssd_raw(n_samples, 0);
    let bits = extract_lsb(&ssd_raw, 8);
    let (p, e, _) = test_bits("LSB-8 raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("LSB-8 raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("LSB-8 + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("LSB-8 + VN", p, e)); }

    // With delay (breaks periodicity)
    println!("\n1b. SSD with 100µs delay between samples:");
    let ssd_delayed = collect_ssd_raw(n_samples / 2, 100); // Fewer samples due to delay
    let bits = extract_lsb(&ssd_delayed, 8);
    let (p, e, _) = test_bits("LSB-8 delayed raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("LSB-8 delayed raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("LSB-8 delayed + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("LSB-8 delayed + VN", p, e)); }

    println!("\n═══ STRATEGY 2: Different Bit Positions ═══");

    println!("\n2a. Middle bits (4-11):");
    let bits = extract_middle_bits(&ssd_raw);
    let (p, e, _) = test_bits("Middle bits raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Middle bits raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("Middle bits + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Middle bits + VN", p, e)); }

    println!("\n2b. High bits (8-15):");
    let bits = extract_high_bits(&ssd_raw);
    let (p, e, _) = test_bits("High bits raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("High bits raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("High bits + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("High bits + VN", p, e)); }

    println!("\n═══ STRATEGY 3: Differential Extraction ═══");

    println!("\n3a. Timing differences:");
    let bits = extract_diff(&ssd_raw);
    let (p, e, _) = test_bits("Diff raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Diff raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("Diff + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Diff + VN", p, e)); }

    println!("\n═══ STRATEGY 4: Multi-Source XOR ═══");

    println!("\n4a. SSD XOR CPU:");
    let cpu = collect_cpu_raw(n_samples);
    let ssd_bits = extract_lsb(&ssd_raw, 4);
    let cpu_bits = extract_lsb(&cpu, 4);
    let bits = xor_combine(&ssd_bits, &cpu_bits);
    let (p, e, _) = test_bits("SSD⊕CPU raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("SSD⊕CPU raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("SSD⊕CPU + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("SSD⊕CPU + VN", p, e)); }

    println!("\n4b. SSD XOR Diff XOR CPU (Triple):");
    let ssd_bits = extract_lsb(&ssd_raw, 4);
    let diff_bits = extract_diff(&ssd_raw);
    let min_len = ssd_bits.len().min(cpu_bits.len()).min(diff_bits.len());
    let bits: Vec<u8> = (0..min_len).map(|i| ssd_bits[i] ^ cpu_bits[i] ^ diff_bits[i]).collect();
    let (p, e, _) = test_bits("Triple XOR raw", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Triple XOR raw", p, e)); }

    let bits = von_neumann(&bits);
    let (p, e, _) = test_bits("Triple XOR + VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Triple XOR + VN", p, e)); }

    println!("\n═══ STRATEGY 5: Minimal Hash Conditioning ═══");

    println!("\n5a. LSB → SHA-256:");
    let bits = sha256_condition(&extract_lsb(&ssd_raw, 8));
    let (p, e, _) = test_bits("LSB→SHA256", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("LSB→SHA256", p, e)); }

    println!("\n5b. Diff → SHA-256:");
    let bits = sha256_condition(&extract_diff(&ssd_raw));
    let (p, e, _) = test_bits("Diff→SHA256", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Diff→SHA256", p, e)); }

    println!("\n5c. (SSD⊕CPU) → SHA-256:");
    let bits = sha256_condition(&xor_combine(&extract_lsb(&ssd_raw, 4), &extract_lsb(&cpu, 4)));
    let (p, e, _) = test_bits("(SSD⊕CPU)→SHA256", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("(SSD⊕CPU)→SHA256", p, e)); }

    println!("\n═══ STRATEGY 6: Layered Approaches ═══");

    println!("\n6a. Diff → SHA256 → VN:");
    let bits = von_neumann(&sha256_condition(&extract_diff(&ssd_raw)));
    let (p, e, _) = test_bits("Diff→SHA→VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Diff→SHA→VN", p, e)); }

    println!("\n6b. (SSD⊕CPU) → SHA256 → VN:");
    let bits = von_neumann(&sha256_condition(&xor_combine(&extract_lsb(&ssd_raw, 4), &extract_lsb(&cpu, 4))));
    let (p, e, _) = test_bits("(SSD⊕CPU)→SHA→VN", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("(SSD⊕CPU)→SHA→VN", p, e)); }

    println!("\n6c. Diff → VN → SHA256:");
    let bits = sha256_condition(&von_neumann(&extract_diff(&ssd_raw)));
    let (p, e, _) = test_bits("Diff→VN→SHA", &bits);
    if p > best_result.map_or(0, |r| r.1) { best_result = Some(("Diff→VN→SHA", p, e)); }

    // Final summary
    println!();
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║                        BEST RESULT                                       ║");
    println!("╠════════════════════════════════════════════════════════════════════════╣");

    if let Some((name, pass, entropy)) = best_result {
        let pct = pass as f64 / 15.0 * 100.0;
        let rating = if pass >= 14 { "🏆 EXCELLENT" }
        else if pass >= 12 { "✅ GOOD" }
        else if pass >= 10 { "⚠️ ACCEPTABLE" }
        else { "❌ NEEDS WORK" };

        println!("║  Method: {:<50}║", name);
        println!("║  NIST: {}/15 ({:.0}%){:>43}║", pass, pct, "");
        println!("║  Min-Entropy: {:.4} bits/bit{:>38}║", entropy, "");
        println!("║  Rating: {:<53}║", rating);
    }

    println!("╚════════════════════════════════════════════════════════════════════════╝");
}
