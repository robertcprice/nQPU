use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("/dev/urandom").unwrap();
    let mut bytes = vec![0u8; 100_000];
    file.read_exact(&mut bytes).unwrap();

    let bits: Vec<u8> = bytes.iter()
        .flat_map(|&b| (0..8).map(move |i| ((b >> i) & 1) as u8))
        .collect();

    let n = bits.len();
    let m = 5;

    // Calculate φ(m) = Σ C_i^(m) * ln(C_i^(m) / n)
    let calc_phi = |pattern_len: usize| -> f64 {
        if pattern_len == 0 {
            return 0.0;
        }

        let num_patterns = 2_usize.pow(pattern_len as u32);
        let mut counts = vec![0usize; num_patterns];

        // Extend sequence by wrapping
        let extended: Vec<u8> = bits.iter()
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

    println!("n = {}", n);
    println!("m = {}", m);
    println!("\nφ(m) = {}", phi_m);
    println!("φ(m+1) = {}", phi_m_1);

    // Approximate entropy: ApEn(m) = φ(m) - φ(m+1)
    let ap_en = phi_m - phi_m_1;
    println!("\nApEn(m) = φ(m) - φ(m+1) = {}", ap_en);

    // NIST formula: χ² = 2n[ln(2) - ApEn(m)]
    // But the code uses: χ² = 2n * [ln(2^m) - ApEn]
    let chi_sq_code = 2.0 * n as f64 * ((2.0_f64.powi(m as i32) as f64).ln() - ap_en);
    let chi_sq_nist = 2.0 * n as f64 * (2.0_f64.ln() - ap_en);

    println!("\nCode formula: χ² = 2n * [ln(2^m) - ApEn] = {}", chi_sq_code);
    println!("NIST formula: χ² = 2n * [ln(2) - ApEn] = {}", chi_sq_nist);

    // Expected: ApEn(m) should be close to ln(2) = 0.693 for random data
    // If ApEn << ln(2), the test fails
    println!("\nln(2) = {}", 2.0_f64.ln());
    println!("Expected ApEn ≈ ln(2) = 0.693 for random data");
}
