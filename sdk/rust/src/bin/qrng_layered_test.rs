//! QRNG Layered Extraction - Combine multiple techniques
//!
//! Hypothesis: Layering multiple extraction techniques might improve results
//! e.g., Differential → Hash → Von Neumann

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nqpu_metal::nist_tests::NistTestSuite;

fn collect_ssd_timings(n_samples: usize) -> Vec<u64> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let test_file = "/tmp/ssd_layered_test.bin";
    let data = vec![0x55u8; 4096];
    let mut timings = Vec::with_capacity(n_samples);

    // Warm-up
    for _ in 0..20 {
        if let Ok(mut file) = OpenOptions::new().write(true).create(true).truncate(true).open(test_file) {
            let _ = file.write_all(&data);
        }
    }

    println!("  Collecting {} SSD samples...", n_samples);

    for _ in 0..n_samples {
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

fn collect_cpu_jitter(n_samples: usize) -> Vec<u64> {
    let mut timings = Vec::with_capacity(n_samples);
    println!("  Collecting {} CPU jitter samples...", n_samples);

    for _ in 0..n_samples {
        let start = Instant::now();
        let mut sum: u64 = 0;
        for j in 0..100 { sum = sum.wrapping_add(j as u64); }
        std::hint::black_box(sum);
        timings.push(start.elapsed().as_nanos() as u64);
    }
    timings
}

// Extraction primitives
fn lsb_bits(values: &[u64], count: usize) -> Vec<u8> {
    let mut bits = Vec::new();
    for &v in values {
        for i in 0..count { bits.push(((v >> i) & 1) as u8); }
    }
    bits
}

fn diff_bits(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();
    for i in 1..values.len() {
        let d = values[i].wrapping_sub(values[i-1]);
        for j in 4..12 { bits.push(((d >> j) & 1) as u8); }
    }
    bits
}

fn xor_fold(values: &[u64]) -> Vec<u8> {
    values.iter().map(|&v| {
        let folded = (v ^ (v >> 32)) as u32;
        (0..8).map(|i| ((folded >> i) & 1) as u8).collect::<Vec<_>>()
    }).flatten().collect()
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

fn hash_condition(values: &[u64]) -> Vec<u8> {
    let mut bits = Vec::new();
    for &v in values {
        let mut hasher = DefaultHasher::new();
        v.hash(&mut hasher);
        let h = hasher.finish();
        for i in 0..8 { bits.push(((h >> i) & 1) as u8); }
    }
    bits
}

fn xor_combine(a: &[u8], b: &[u8]) -> Vec<u8> {
    (0..a.len().min(b.len())).map(|i| a[i] ^ b[i]).collect()
}

fn shannon_entropy(bits: &[u8]) -> f64 {
    let n = bits.len() as f64;
    let ones = bits.iter().filter(|&&b| b == 1).count() as f64;
    let p1 = ones / n;
    let p0 = 1.0 - p1;

    if p1 == 0.0 || p0 == 0.0 { return 0.0; }
    -(p0 * p0.log2() + p1 * p1.log2())
}

fn test_method(name: &str, bits: &[u8]) -> (usize, f64, f64) {
    if bits.len() < 1000 {
        println!("  ⚠️ {} skipped (not enough bits: {})", name, bits.len());
        return (0, 0.0, 0.0);
    }

    let suite = NistTestSuite::new();
    let result = suite.run_all_tests(bits);
    let se = shannon_entropy(bits);

    println!("  {}: {} bits, {}/15 NIST, min-ent={:.4}, Shannon={:.4}",
        name, bits.len(), result.passed_count, result.min_entropy, se);

    (result.passed_count, result.min_entropy, se)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     LAYERED EXTRACTION EXPERIMENTS                            ║");
    println!("║     Testing combinations of extraction techniques              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Collect more data this time
    let n_samples = 10_000;
    println!("▶ COLLECTING DATA ({} samples)...", n_samples);
    let ssd = collect_ssd_timings(n_samples);
    let cpu = collect_cpu_jitter(n_samples);
    println!();

    let mut results: Vec<(&str, usize, f64, f64)> = Vec::new();

    println!("▶ SINGLE LAYER METHODS:");
    // Raw LSB
    let mut bits = lsb_bits(&ssd, 8);
    let (p, m, s) = test_method("LSB-8 raw", &bits);
    results.push(("LSB-8 raw", p, m, s));
    bits = von_neumann(&bits);
    let (p, m, s) = test_method("LSB-8 + VN", &bits);
    results.push(("LSB-8 + VN", p, m, s));

    // Differential
    bits = diff_bits(&ssd);
    let (p, m, s) = test_method("Diff raw", &bits);
    results.push(("Diff raw", p, m, s));
    bits = von_neumann(&bits);
    let (p, m, s) = test_method("Diff + VN", &bits);
    results.push(("Diff + VN", p, m, s));

    // XOR Fold
    bits = xor_fold(&ssd);
    let (p, m, s) = test_method("XOR-fold raw", &bits);
    results.push(("XOR-fold raw", p, m, s));
    bits = von_neumann(&bits);
    let (p, m, s) = test_method("XOR-fold + VN", &bits);
    results.push(("XOR-fold + VN", p, m, s));

    // Hash
    bits = hash_condition(&ssd);
    let (p, m, s) = test_method("Hash raw", &bits);
    results.push(("Hash raw", p, m, s));

    println!("\n▶ TWO-LAYER COMBINATIONS:");
    // Diff → Hash
    bits = diff_bits(&ssd);
    let timings_for_hash: Vec<u64> = (0..bits.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((bits[i*8+j] as u64) << j))
    }).collect();
    bits = hash_condition(&timings_for_hash);
    let (p, m, s) = test_method("Diff→Hash", &bits);
    results.push(("Diff→Hash", p, m, s));

    // Diff → VN
    bits = von_neumann(&diff_bits(&ssd));
    let (p, m, s) = test_method("Diff→VN", &bits);
    results.push(("Diff→VN", p, m, s));

    // XOR-fold → Hash
    bits = xor_fold(&ssd);
    let timings_for_hash: Vec<u64> = (0..bits.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((bits[i*8+j] as u64) << j))
    }).collect();
    bits = hash_condition(&timings_for_hash);
    let (p, m, s) = test_method("XOR→Hash", &bits);
    results.push(("XOR→Hash", p, m, s));

    // Hash → VN
    bits = von_neumann(&hash_condition(&ssd));
    let (p, m, s) = test_method("Hash→VN", &bits);
    results.push(("Hash→VN", p, m, s));

    // LSB → Hash → VN
    bits = lsb_bits(&ssd, 8);
    let timings_for_hash: Vec<u64> = (0..bits.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((bits[i*8+j] as u64) << j))
    }).collect();
    bits = von_neumann(&hash_condition(&timings_for_hash));
    let (p, m, s) = test_method("LSB→Hash→VN", &bits);
    results.push(("LSB→Hash→VN", p, m, s));

    println!("\n▶ MULTI-SOURCE COMBINATIONS:");
    // SSD XOR CPU
    let ssd_bits = lsb_bits(&ssd, 4);
    let cpu_bits = lsb_bits(&cpu, 4);
    bits = xor_combine(&ssd_bits, &cpu_bits);
    let (p, m, s) = test_method("SSD⊕CPU raw", &bits);
    results.push(("SSD⊕CPU raw", p, m, s));
    bits = von_neumann(&bits);
    let (p, m, s) = test_method("SSD⊕CPU+VN", &bits);
    results.push(("SSD⊕CPU+VN", p, m, s));

    // SSD XOR CPU XOR Diff
    let ssd_bits = lsb_bits(&ssd, 4);
    let cpu_bits = lsb_bits(&cpu, 4);
    let diff = diff_bits(&ssd);
    let min_len = ssd_bits.len().min(cpu_bits.len()).min(diff.len());
    bits = (0..min_len).map(|i| ssd_bits[i] ^ cpu_bits[i] ^ diff[i]).collect();
    let (p, m, s) = test_method("Triple XOR raw", &bits);
    results.push(("Triple XOR raw", p, m, s));
    bits = von_neumann(&bits);
    let (p, m, s) = test_method("Triple XOR+VN", &bits);
    results.push(("Triple XOR+VN", p, m, s));

    // (SSD XOR CPU) → Hash
    let ssd_bits = lsb_bits(&ssd, 4);
    let cpu_bits = lsb_bits(&cpu, 4);
    bits = xor_combine(&ssd_bits, &cpu_bits);
    let timings_for_hash: Vec<u64> = (0..bits.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((bits[i*8+j] as u64) << j))
    }).collect();
    bits = hash_condition(&timings_for_hash);
    let (p, m, s) = test_method("SSD⊕CPU→Hash", &bits);
    results.push(("SSD⊕CPU→Hash", p, m, s));

    println!("\n▶ TRIPLE-LAYER COMBINATIONS:");
    // Diff → Hash → VN
    let d = diff_bits(&ssd);
    let t: Vec<u64> = (0..d.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((d[i*8+j] as u64) << j))
    }).collect();
    bits = von_neumann(&hash_condition(&t));
    let (p, m, s) = test_method("Diff→Hash→VN", &bits);
    results.push(("Diff→Hash→VN", p, m, s));

    // (SSD⊕CPU) → Hash → VN
    let xor_bits = xor_combine(&lsb_bits(&ssd, 4), &lsb_bits(&cpu, 4));
    let t: Vec<u64> = (0..xor_bits.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((xor_bits[i*8+j] as u64) << j))
    }).collect();
    bits = von_neumann(&hash_condition(&t));
    let (p, m, s) = test_method("SSD⊕CPU→Hash→VN", &bits);
    results.push(("SSD⊕CPU→Hash→VN", p, m, s));

    // XOR-fold → Hash → VN
    let xf = xor_fold(&ssd);
    let t: Vec<u64> = (0..xf.len()/8).map(|i| {
        (0..8).fold(0u64, |acc, j| acc | ((xf[i*8+j] as u64) << j))
    }).collect();
    bits = von_neumann(&hash_condition(&t));
    let (p, m, s) = test_method("XOR→Hash→VN", &bits);
    results.push(("XOR→Hash→VN", p, m, s));

    // Print summary
    println!();
    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║                      FINAL COMPARISON                               ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║ Method              │ NIST │ Min-Ent │ Shannon │ Rating           ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");

    results.sort_by(|a, b| b.1.cmp(&a.1));

    for (name, pass, min_ent, shannon) in &results {
        let rating = if *pass >= 14 { "🏆 BEST" }
        else if *pass >= 12 { "✅ GOOD" }
        else if *pass >= 10 { "⚠️ OK" }
        else if *pass >= 8 { "❌ POOR" }
        else { "💀 BAD" };

        println!("║ {:<19} │ {:>2}/15 │ {:>7.4} │ {:>7.4} │ {:<16} ║",
            name, pass, min_ent, shannon, rating);
    }

    println!("╚════════════════════════════════════════════════════════════════════╝");

    if let Some(best) = results.first() {
        println!("\n🏆 BEST: {} with {}/15 NIST passes", best.0, best.1);
    }
}
