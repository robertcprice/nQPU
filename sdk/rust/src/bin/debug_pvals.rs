use std::fs::File;
use std::io::Read;

fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 { return f64::INFINITY; }
    if (x - x.round()).abs() < 1e-10 && x <= 171.0 {
        let n = x.round() as i64;
        if n >= 1 && n <= 171 {
            let mut fact: f64 = 1.0;
            for i in 2..n { fact *= i as f64; }
            return fact.ln();
        }
    }
    let g = 7;
    let c: [f64; 9] = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut sum = c[0];
    for i in 1..g + 2 { sum += c[i] / (z + i as f64); }
    let t = z + g as f64 + 0.5;
    0.9189385332046727 + (t + 0.5).ln() * (z + 0.5) - t + sum.ln()
}

fn igamc(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 { return 0.0; }
    if x == 0.0 { return 1.0; }
    if x < a + 1.0 {
        let max_iterations = 500;
        let eps = 1e-14_f64;
        let mut term = 1.0 / a;
        let mut sum = term;
        for n in 1..max_iterations {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < eps * sum.abs() { break; }
        }
        let log_gamma_a = gamma_ln(a);
        let log_p = a * x.ln() - x - log_gamma_a + sum.ln();
        1.0 - log_p.exp().min(1.0).max(0.0)
    } else {
        let max_iterations = 500;
        let eps = 1e-14_f64;
        let fpmin = f64::MIN_POSITIVE;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / fpmin;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=max_iterations {
            let an = -i as f64 * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < fpmin { d = fpmin; }
            c = b + an / c;
            if c.abs() < fpmin { c = fpmin; }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < eps { break; }
        }
        let log_gamma_a = gamma_ln(a);
        let log_q = -x + a * x.ln() + h.ln() - log_gamma_a;
        log_q.exp().min(1.0).max(0.0)
    }
}

fn main() {
    let mut file = File::open("/dev/urandom").unwrap();
    let mut bytes = vec![0u8; 100_000];
    file.read_exact(&mut bytes).unwrap();

    let bits: Vec<u8> = bytes.iter()
        .flat_map(|&b| (0..8).map(move |i| ((b >> i) & 1) as u8))
        .collect();

    let n = bits.len();
    let m = 9;
    let block_size = 1032;
    let num_blocks = n / block_size;
    let template: Vec<u8> = vec![1; m];

    println!("Overlapping Template Test:");
    println!("  m = {}, block_size = {}, num_blocks = {}", m, block_size, num_blocks);

    let mut counts = Vec::new();
    for i in 0..num_blocks {
        let block = &bits[i * block_size..(i + 1) * block_size];
        let mut count = 0;
        for j in 0..=block_size - template.len() {
            if block[j..j + template.len()] == template[..] {
                count += 1;
            }
        }
        counts.push(count.min(m + 1));
    }

    // Distribution of counts
    for i in 0..=5 {
        let freq = counts.iter().filter(|&&c| c == i).count();
        println!("  Count {} appears in {} blocks", i, freq);
    }

    // Theoretical distribution using η = λ/2
    let lambda = (block_size as f64 - m as f64 + 1.0) / 2.0_f64.powi(m as i32);
    let eta = lambda / 2.0;
    println!("\n  λ = {:.4}, η = {:.4}", lambda, eta);

    let k = m + 2;
    let exp_neg_eta = (-eta).exp();
    let mut pi = vec![0.0; k];
    let mut factorial = 1.0;

    for u in 0..(k - 1) {
        if u > 0 { factorial *= u as f64; }
        pi[u] = eta.powi(u as i32) * exp_neg_eta / factorial;
    }
    let sum_pi: f64 = pi[..k-1].iter().sum();
    pi[k - 1] = 1.0 - sum_pi;

    println!("\n  Theoretical probabilities (Poisson):");
    for i in 0..k {
        println!("    π[{}] = {:.6} (expected {} blocks)", i, pi[i], (pi[i] * num_blocks as f64) as i64);
    }

    // Chi-squared
    let mut chi_sq = 0.0;
    for i in 0..k {
        let observed = counts.iter().filter(|&&c| c == i).count() as f64;
        let expected = pi[i] * num_blocks as f64;
        if expected > 0.0 {
            chi_sq += (observed - expected).powi(2) / expected;
        }
        println!("    i={}: observed={}, expected={:.1}, contribution={:.4}", 
                 i, observed, expected, (observed - expected).powi(2) / expected);
    }

    let df = (k - 1) as f64;
    let p_value = igamc(df / 2.0, chi_sq / 2.0);

    println!("\n  Chi-squared = {:.4}", chi_sq);
    println!("  df = {}", df);
    println!("  p-value = {:.4}", p_value);
}
