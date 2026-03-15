use num_complex::Complex;
use rustfft::FftPlanner;
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
    let n_fft = n.next_power_of_two();
    let half_n = n / 2;

    let mut input: Vec<Complex<f64>> = bits
        .iter()
        .map(|&b| Complex::new(if b == 1 { 1.0 } else { -1.0 }, 0.0))
        .collect();
    input.resize(n_fft, Complex::new(0.0, 0.0));

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    fft.process(&mut input);

    let m: Vec<f64> = input[1..=half_n]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    let threshold = ((-0.05_f64).ln() * n as f64).sqrt();
    let n_0 = 0.95 * half_n as f64;
    let n_1 = m.iter().filter(|&&v| v < threshold).count() as f64;

    let d = (n_1 - n_0) / (n as f64 * 0.95 * 0.05 / 4.0).sqrt();

    println!("n = {}", n);
    println!("n_fft = {}", n_fft);
    println!("half_n = {}", half_n);
    println!("threshold = {:.2}", threshold);
    println!("n_0 (expected below threshold) = {:.2}", n_0);
    println!("n_1 (actual below threshold) = {}", n_1);
    println!("n_1 / half_n = {:.4}", n_1 / half_n as f64);
    println!("d = {:.4}", d);

    // Sample magnitudes
    let max_m = m.iter().cloned().fold(0.0, f64::max);
    let min_m = m.iter().cloned().fold(f64::MAX, f64::min);
    println!("\nMagnitude range: {:.2} - {:.2}", min_m, max_m);

    // Distribution
    let mut bins = [0usize; 10];
    for &v in &m {
        let bin = ((v / max_m) * 9.0) as usize;
        bins[bin.min(9)] += 1;
    }
    println!("\nMagnitude distribution (0-10% of max, 10-20%, ...):");
    for (i, &count) in bins.iter().enumerate() {
        println!("  {}-{}%: {}", i * 10, (i + 1) * 10, count);
    }
}
