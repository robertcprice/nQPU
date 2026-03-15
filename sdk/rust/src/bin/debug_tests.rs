use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("/dev/urandom").unwrap();
    let mut bytes = vec![0u8; 100_000];
    file.read_exact(&mut bytes).unwrap();

    let bits: Vec<u8> = bytes
        .iter()
        .flat_map(|&b| (0..8).map(move |i| ((b >> i) & 1) as u8))
        .collect();

    let n = bits.len();
    println!("Testing {} bits from /dev/urandom\n", n);

    // Debug Serial test
    let m = 5;
    let count_patterns = |pattern_len: usize| -> f64 {
        let num_patterns = 2_usize.pow(pattern_len as u32);
        let mut counts = vec![0i64; num_patterns];
        let extended: Vec<u8> = bits
            .iter()
            .chain(bits[..pattern_len - 1].iter())
            .copied()
            .collect();
        for i in 0..n {
            let pattern = extended[i..i + pattern_len]
                .iter()
                .fold(0usize, |acc, &b| (acc << 1) | b as usize);
            counts[pattern] += 1;
        }
        let sum_sq: f64 = counts.iter().map(|&c| (c as f64).powi(2)).sum();
        let n_f = n as f64;
        let two_pow_m = 2.0_f64.powi(pattern_len as i32);
        two_pow_m * sum_sq / n_f - n_f
    };

    let psi_m = count_patterns(m);
    let psi_m_1 = count_patterns(m - 1);
    let psi_m_2 = count_patterns(m - 2);
    let del1 = psi_m - psi_m_1;
    let del2 = psi_m - 2.0 * psi_m_1 + psi_m_2;

    println!("=== Serial Test ===");
    println!(
        "ψ²_m = {:.4}, ψ²_(m-1) = {:.4}, ψ²_(m-2) = {:.4}",
        psi_m, psi_m_1, psi_m_2
    );
    println!(
        "Δ1 = {:.4} (should be small, ~df={})",
        del1,
        2.0_f64.powi(m as i32 - 2)
    );
    println!(
        "Δ2 = {:.4} (should be small, ~df={})",
        del2,
        2.0_f64.powi(m as i32 - 3)
    );

    // Debug Approx Entropy
    let calc_phi = |pattern_len: usize| -> f64 {
        if pattern_len == 0 {
            return 0.0;
        }
        let num_patterns = 2_usize.pow(pattern_len as u32);
        let mut counts = vec![0usize; num_patterns];
        let extended: Vec<u8> = bits
            .iter()
            .chain(bits[..pattern_len - 1].iter())
            .copied()
            .collect();
        for i in 0..n {
            let pattern = extended[i..i + pattern_len]
                .iter()
                .fold(0usize, |acc, &b| (acc << 1) | b as usize);
            counts[pattern] += 1;
        }
        let mut sum = 0.0;
        let n_f = n as f64;
        for &c in &counts {
            if c > 0 {
                let p_i = c as f64 / n_f;
                sum += p_i * p_i.ln();
            }
        }
        sum
    };

    let phi_m = calc_phi(m);
    let phi_m_1 = calc_phi(m + 1);
    let ap_en = phi_m - phi_m_1;
    let chi_sq = 2.0 * n as f64 * (2.0_f64.ln() - ap_en);

    println!("\n=== Approximate Entropy ===");
    println!("φ(m) = {:.4}, φ(m+1) = {:.4}", phi_m, phi_m_1);
    println!(
        "ApEn = {:.6} (should be ≈ ln(2) = {:.6})",
        ap_en,
        2.0_f64.ln()
    );
    println!("χ² = {:.4} (should be small for random data)", chi_sq);

    // Debug Overlapping Template
    let m_ot = 9;
    let block_size = 1032;
    let num_blocks = n / block_size;
    let template: Vec<u8> = vec![1; m_ot];
    let mut counts = Vec::new();

    for i in 0..num_blocks {
        let block = &bits[i * block_size..(i + 1) * block_size];
        let mut count = 0;
        for j in 0..=block_size - template.len() {
            if block[j..j + template.len()] == template[..] {
                count += 1;
            }
        }
        counts.push(count.min(m_ot + 1));
    }

    let lambda = (block_size as f64 - m_ot as f64 + 1.0) / 2.0_f64.powi(m_ot as i32);
    let eta = lambda / 2.0;
    let k = m_ot + 2;

    println!("\n=== Overlapping Template ===");
    println!(
        "m = {}, block_size = {}, num_blocks = {}",
        m_ot, block_size, num_blocks
    );
    println!("λ = {:.4}, η = {:.4}", lambda, eta);

    // Count frequency distribution
    for i in 0..5 {
        let freq = counts.iter().filter(|&&c| c == i).count();
        println!("Count {} appears in {} blocks", i, freq);
    }
    let freq_ge5 = counts.iter().filter(|&&c| c >= 5).count();
    println!("Count ≥5 appears in {} blocks", freq_ge5);
}
