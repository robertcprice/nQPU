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
    let m = 5;

    // CORRECT NIST formula
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

        // NIST formula: ψ²_m = (2^m / n) * Σ count[i]² - n
        two_pow_m * sum_sq / n_f - n_f
    };

    let psi_sq_m = count_patterns(m);
    let psi_sq_m_1 = count_patterns(m - 1);
    let psi_sq_m_2 = count_patterns(m - 2);

    println!("n = {}", n);
    println!("\npsi^2_m (m=5) = {:.4}", psi_sq_m);
    println!("psi^2_(m-1) (m=4) = {:.4}", psi_sq_m_1);
    println!("psi^2_(m-2) (m=3) = {:.4}", psi_sq_m_2);

    let delta_sq_m = psi_sq_m - psi_sq_m_1;
    let delta_sq_2_m = psi_sq_m - 2.0 * psi_sq_m_1 + psi_sq_m_2;

    println!("\ndelta^2_m = {:.4}", delta_sq_m);
    println!("delta^2_2m = {:.4}", delta_sq_2_m);

    let v1 = 2.0_f64.powi(m as i32 - 1) * delta_sq_m;
    let v2 = 2.0_f64.powi(m as i32 - 2) * delta_sq_2_m;

    println!("\nv1 (chi-sq for first test) = {:.4}", v1);
    println!("v2 (chi-sq for second test) = {:.4}", v2);

    let df1 = 2.0_f64.powi(m as i32 - 2);
    let df2 = 2.0_f64.powi(m as i32 - 3);

    println!("\ndf1 = {}", df1);
    println!("df2 = {}", df2);

    // Expected: psi^2 should be small (close to 0) for random data
    // v1 and v2 should be small positive values
}
