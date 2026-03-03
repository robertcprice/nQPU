//! Auto-Routing Stabilizer Simulator
//!
//! Automatically selects the optimal backend (CPU or GPU) based on circuit
//! characteristics for maximum performance.
//!
//! # How It Works
//!
//! 1. **On first use**: Runs a quick benchmark to find the crossover point
//!    where GPU becomes faster than CPU for YOUR specific hardware
//! 2. **Caches the result**: Stores the optimal threshold for future runs
//! 3. **Routes automatically**: Uses the cached threshold for fast decisions
//!
//! # Why Different Systems Have Different Crossover Points
//!
//! - **M4 Max** (12 GPU cores): GPU wins at ~200 qubits
//! - **M4 Pro** (5 GPU cores): GPU wins at ~300 qubits
//! - **M4** (4 GPU cores): GPU wins at ~400 qubits
//! - **CPU-heavy workload**: Crossover shifts to higher qubits
//!
//! # Usage
//!
//! ```rust,ignore
//! // Auto-detect optimal config on first use
//! let router = AutoStabilizer::auto_detect(1000)?;
//!
//! // Now router knows the optimal threshold for YOUR system
//! let backend = router.select_backend(&profile);
//! ```

use std::time::Instant;
use std::sync::OnceLock;

/// Global cached threshold (auto-detected on first use)
static CACHED_THRESHOLD: OnceLock<usize> = OnceLock::new();

/// Backend selection for stabilizer simulation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StabilizerBackend {
    /// CPU with NEON SIMD (better for small circuits)
    Cpu,
    /// Metal GPU (better for large circuits)
    Gpu,
    /// Auto-select based on circuit characteristics
    Auto,
}

/// Configuration for the auto-routing simulator
#[derive(Clone, Debug)]
pub struct RouterConfig {
    /// Minimum qubits to use GPU (crossover point)
    /// If None, will auto-detect on first use
    pub gpu_threshold: Option<usize>,
    /// Minimum gates to bother with GPU dispatch
    pub min_gates_for_gpu: usize,
    /// Number of warmup iterations before benchmarking
    pub warmup_iters: usize,
    /// Number of benchmark iterations for averaging
    pub bench_iters: usize,
    /// Test sizes to benchmark (in qubits)
    pub test_sizes: Vec<usize>,
    /// Gates to use for benchmark
    pub bench_gates: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: None,  // Auto-detect!
            min_gates_for_gpu: 1000,
            warmup_iters: 2,
            bench_iters: 3,
            test_sizes: vec![50, 100, 150, 200, 250, 300, 400, 500],
            bench_gates: 5000,
        }
    }
}

impl RouterConfig {
    /// Create config with a pre-known threshold (skip auto-detection)
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            gpu_threshold: Some(threshold),
            ..Default::default()
        }
    }

    /// Get the effective threshold (cached or fallback)
    pub fn effective_threshold(&self) -> usize {
        self.gpu_threshold
            .or_else(|| CACHED_THRESHOLD.get().copied())
            .unwrap_or(300)  // Conservative fallback
    }
}

/// Circuit characteristics for routing decisions
#[derive(Clone, Debug)]
pub struct CircuitProfile {
    pub num_qubits: usize,
    pub num_gates: usize,
    pub two_qubit_ratio: f32,  // Fraction of 2-qubit gates
    pub depth: usize,
}

impl CircuitProfile {
    /// Estimate if GPU would be faster
    pub fn should_use_gpu(&self, config: &RouterConfig) -> bool {
        let threshold = config.effective_threshold();
        self.num_qubits >= threshold && self.num_gates >= config.min_gates_for_gpu
    }

    /// Estimate relative performance (GPU/CPU ratio)
    pub fn estimated_speedup(&self) -> f64 {
        let threshold = CACHED_THRESHOLD.get().copied().unwrap_or(300);

        if self.num_qubits < threshold {
            0.5 * (self.num_qubits as f64 / threshold as f64)
        } else {
            1.0 + 0.5 * ((self.num_qubits - threshold) as f64 / 500.0).min(2.0)
        }
    }
}

/// Benchmark result for a single qubit size
#[derive(Clone, Debug)]
pub struct BenchmarkPoint {
    pub num_qubits: usize,
    pub cpu_time_ms: f64,
    pub gpu_time_ms: f64,
    pub ratio: f64,  // CPU/GPU - >1 means GPU faster
}

/// Full benchmark results
#[derive(Clone, Debug)]
pub struct BenchmarkResults {
    pub points: Vec<BenchmarkPoint>,
    pub optimal_threshold: usize,
    pub device_name: String,
    pub cpu_info: String,
}

impl BenchmarkResults {
    /// Print a nice summary table
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(70));
        println!("Auto-Stabilizer Benchmark Results");
        println!("{}", "=".repeat(70));
        println!("Device: {}", self.device_name);
        println!("CPU: {}", self.cpu_info);
        println!();
        println!("{:<10} {:>12} {:>12} {:>10} {:>10}",
            "Qubits", "CPU (ms)", "GPU (ms)", "Ratio", "Winner");
        println!("{}", "-".repeat(70));

        for point in &self.points {
            let winner = if point.ratio > 1.1 { "GPU ✅" }
                        else if point.ratio < 0.9 { "CPU ✅" }
                        else { "Tie" };
            println!("{:<10} {:>12.2} {:>12.2} {:>10.2}x {:>10}",
                point.num_qubits,
                point.cpu_time_ms,
                point.gpu_time_ms,
                point.ratio,
                winner);
        }

        println!("{}", "-".repeat(70));
        println!("Optimal crossover point: {} qubits", self.optimal_threshold);
        println!("→ Use CPU for n < {}, GPU for n >= {}",
            self.optimal_threshold, self.optimal_threshold);
        println!("{}", "=".repeat(70));
    }
}

/// Auto-routing stabilizer simulator
///
/// Automatically selects CPU or GPU based on circuit size.
pub struct AutoStabilizer {
    config: RouterConfig,
    #[cfg(target_os = "macos")]
    gpu_sim: Option<crate::metal_stabilizer::MetalStabilizerSimulator>,
    benchmark_results: Option<BenchmarkResults>,
}

impl AutoStabilizer {
    /// Create new auto-routing simulator with AUTO-DETECTION
    ///
    /// This will benchmark on first use to find the optimal crossover
    /// point for YOUR specific hardware.
    pub fn auto_detect(num_qubits: usize) -> Result<Self, String> {
        Self::with_config(num_qubits, RouterConfig::default())
    }

    /// Create with a pre-known threshold (skip auto-detection)
    pub fn with_threshold(num_qubits: usize, threshold: usize) -> Result<Self, String> {
        Self::with_config(num_qubits, RouterConfig::with_threshold(threshold))
    }

    /// Create with custom configuration
    pub fn with_config(num_qubits: usize, config: RouterConfig) -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let gpu_sim = crate::metal_stabilizer::MetalStabilizerSimulator::new(num_qubits).ok();
            Ok(Self {
                config,
                gpu_sim,
                benchmark_results: None,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            Ok(Self {
                config,
                benchmark_results: None,
            })
        }
    }

    /// Check if we have a cached threshold (fast)
    pub fn has_cached_threshold() -> bool {
        CACHED_THRESHOLD.get().is_some()
    }

    /// Get the cached threshold without benchmarking
    pub fn get_cached_threshold() -> Option<usize> {
        CACHED_THRESHOLD.get().copied()
    }

    /// Select backend for given circuit
    ///
    /// If no threshold is known, triggers auto-benchmark.
    pub fn select_backend(&mut self, profile: &CircuitProfile) -> StabilizerBackend {
        // Trigger auto-detection if needed
        if self.config.gpu_threshold.is_none() && CACHED_THRESHOLD.get().is_none() {
            self.benchmark_and_cache();
        }

        if profile.should_use_gpu(&self.config) {
            StabilizerBackend::Gpu
        } else {
            StabilizerBackend::Cpu
        }
    }

    /// Run benchmark to find optimal crossover point and CACHE it globally
    ///
    /// This only runs once per process - subsequent calls use the cached result.
    pub fn benchmark_and_cache(&mut self) -> usize {
        // Check if already cached globally
        if let Some(&threshold) = CACHED_THRESHOLD.get() {
            return threshold;
        }

        let results = self.run_full_benchmark();

        // Cache globally
        let _ = CACHED_THRESHOLD.set(results.optimal_threshold);

        // Also store in config
        self.config.gpu_threshold = Some(results.optimal_threshold);
        self.benchmark_results = Some(results.clone());

        results.optimal_threshold
    }

    /// Run full benchmark and return detailed results
    #[cfg(target_os = "macos")]
    pub fn run_full_benchmark(&mut self) -> BenchmarkResults {
        let mut points = Vec::new();
        let mut optimal_threshold = 500;  // Default high

        let device_name = self.gpu_sim
            .as_ref()
            .map(|g| g.device_name())
            .unwrap_or_else(|| "Unknown".to_string());

        let cpu_info = std::env::var("HOSTNAME")
            .unwrap_or_else(|_| "Apple Silicon".to_string());

        let test_sizes = self.config.test_sizes.clone();
        let bench_gates = self.config.bench_gates;
        let warmup_iters = self.config.warmup_iters;
        let bench_iters = self.config.bench_iters;

        for &n in &test_sizes {
            let num_gates = bench_gates;

            // CPU benchmark
            let cpu_time = self.benchmark_cpu(n, num_gates, warmup_iters, bench_iters);

            // GPU benchmark
            let gpu_time = self.benchmark_gpu(n, num_gates, warmup_iters, bench_iters);

            let ratio = if gpu_time > 0.0 { cpu_time / gpu_time } else { 0.0 };

            points.push(BenchmarkPoint {
                num_qubits: n,
                cpu_time_ms: cpu_time * 1000.0,
                gpu_time_ms: gpu_time * 1000.0,
                ratio,
            });

            // Find first point where GPU is 10% faster
            if ratio > 1.1 && optimal_threshold == 500 {
                optimal_threshold = n;
            }
        }

        BenchmarkResults {
            points,
            optimal_threshold,
            device_name,
            cpu_info,
        }
    }

    #[cfg(target_os = "macos")]
    fn benchmark_cpu(&self, n: usize, num_gates: usize, warmup_iters: usize, bench_iters: usize) -> f64 {
        let mut cpu = crate::fast_stabilizer::FastTableau::new(n);
        let cpu_gates = crate::fast_stabilizer::GateOp::random_circuit(n, num_gates, 42);

        // Warmup
        for _ in 0..warmup_iters {
            cpu.reset();
            cpu.apply_batch(&cpu_gates);
        }

        // Benchmark
        let mut total_time = 0.0;
        for _ in 0..bench_iters {
            cpu.reset();
            let start = Instant::now();
            cpu.apply_batch(&cpu_gates);
            total_time += start.elapsed().as_secs_f64();
        }

        total_time / bench_iters as f64
    }

    #[cfg(target_os = "macos")]
    fn benchmark_gpu(&mut self, n: usize, num_gates: usize, warmup_iters: usize, bench_iters: usize) -> f64 {
        // Need to resize GPU simulator
        let Ok(mut gpu) = crate::metal_stabilizer::MetalStabilizerSimulator::new(n) else {
            return f64::INFINITY;
        };

        let gates: Vec<_> = (0..num_gates).map(|i| {
            let q = (i % n) as u32;
            let q2 = ((i + 1) % n) as u32;
            match i % 4 {
                0 => crate::metal_stabilizer::StabilizerGate::H { qubit: q },
                1 => crate::metal_stabilizer::StabilizerGate::S { qubit: q },
                2 => crate::metal_stabilizer::StabilizerGate::CX { control: q, target: q2 },
                _ => crate::metal_stabilizer::StabilizerGate::CZ { qubit1: q, qubit2: q2 },
            }
        }).collect();

        // Warmup
        for _ in 0..warmup_iters {
            gpu.reset();
            let _ = gpu.run_circuit_batch(&gates);
        }

        // Benchmark
        let mut total_time = 0.0;
        for _ in 0..bench_iters {
            gpu.reset();
            let start = Instant::now();
            let _ = gpu.run_circuit_batch(&gates);
            total_time += start.elapsed().as_secs_f64();
        }

        total_time / bench_iters as f64
    }

    #[cfg(not(target_os = "macos"))]
    pub fn run_full_benchmark(&mut self) -> BenchmarkResults {
        BenchmarkResults {
            points: vec![],
            optimal_threshold: usize::MAX,
            device_name: "N/A".to_string(),
            cpu_info: "Non-macOS".to_string(),
        }
    }

    /// Get benchmark results (if benchmark was run)
    pub fn benchmark_results(&self) -> Option<&BenchmarkResults> {
        self.benchmark_results.as_ref()
    }

    /// Get current configuration
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: RouterConfig) {
        self.config = config;
    }
}

/// One-time auto-detection function
///
/// Call this once at program startup to find and cache the optimal threshold.
/// Takes ~2-3 seconds but only runs once.
#[cfg(target_os = "macos")]
pub fn auto_detect_optimal_threshold() -> usize {
    if let Some(&threshold) = CACHED_THRESHOLD.get() {
        return threshold;
    }

    println!("Auto-detecting optimal CPU/GPU crossover point...");
    let mut router = AutoStabilizer::auto_detect(500).unwrap();
    let results = router.run_full_benchmark();
    results.print_summary();
    router.benchmark_and_cache()
}

#[cfg(not(target_os = "macos"))]
pub fn auto_detect_optimal_threshold() -> usize {
    usize::MAX  // Never use GPU on non-macOS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_profile_gpu_decision() {
        let config = RouterConfig::with_threshold(300);

        // Small circuit should use CPU
        let small = CircuitProfile {
            num_qubits: 100,
            num_gates: 1000,
            two_qubit_ratio: 0.5,
            depth: 50,
        };
        assert_eq!(small.should_use_gpu(&config), false);

        // Large circuit should use GPU
        let large = CircuitProfile {
            num_qubits: 500,
            num_gates: 10000,
            two_qubit_ratio: 0.5,
            depth: 500,
        };
        assert_eq!(large.should_use_gpu(&config), true);
    }

    #[test]
    fn test_config_auto_detect() {
        // Default config should have None threshold (auto-detect)
        let config = RouterConfig::default();
        assert!(config.gpu_threshold.is_none());
    }

    #[test]
    fn test_cached_threshold() {
        // Set a cached threshold
        let _ = CACHED_THRESHOLD.set(250);

        let config = RouterConfig::default();
        assert_eq!(config.effective_threshold(), 250);
    }
}
