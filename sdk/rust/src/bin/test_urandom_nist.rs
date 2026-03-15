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

    println!("Testing {} bits from /dev/urandom...", bits.len());

    let suite = nqpu_metal::nist_tests::NistTestSuite::new();
    let result = suite.run_all_tests(&bits);

    println!("\nNIST Results for /dev/urandom:");
    for t in &result.tests {
        println!(
            "  {} - p={:.4} {}",
            t.test_name,
            t.p_value,
            if t.passed { "✅" } else { "❌" }
        );
    }

    println!("\nPassed: {}/{}", result.passed_count, result.tests.len());
    println!("Min-entropy: {:.4}", result.min_entropy);

    let failed: Vec<_> = result.tests.iter().filter(|t| !t.passed).collect();
    if !failed.is_empty() {
        println!("\n⚠️ These tests failed - possible implementation bugs:");
        for t in failed {
            println!("  - {} (p={:.4})", t.test_name, t.p_value);
        }
    }
}
