//! Thermal-Aware GPU Scheduling for Metal
//!
//! Reduces performance variance by adapting GPU parameters based on thermal state.
//!
//! # Problem
//!
//! Metal on Apple Silicon shows significant performance variance (3-5x) due to:
//! - Thermal throttling under sustained load
//! - Dynamic power management
//! - Fixed threadgroup sizes that don't adapt to conditions
//!
//! # Solution
//!
//! Monitor execution times and adapt threadgroup size to maintain stable performance:
//! - Detect performance degradation (likely thermal throttling)
//! - Reduce threadgroup size to lower power consumption
//! - Gradually increase when thermals allow
//!
//! # Usage
//!
//! ```ignore
//! let mut scheduler = ThermalScheduler::new(initial_threadgroup_size);
//!
//! loop {
//!     let start = Instant::now();
//!     execute_on_gpu(&mut sim, gates);
//!     let elapsed = start.elapsed();
//!
//!     if let Some(adjusted_size) = scheduler.adjust(elapsed) {
//!         sim.set_threadgroup_size(adjusted_size);
//!     }
//! }
//! ```

use std::collections::VecDeque;

/// Thermal-aware GPU scheduler that adapts to performance changes.
#[derive(Debug, Clone)]
pub struct ThermalScheduler {
    /// Current threadgroup size
    threadgroup_size: u64,
    /// Maximum allowed threadgroup size (from pipeline query)
    max_threadgroup_size: u64,
    /// Minimum allowed threadgroup size (avoid too-small dispatches)
    min_threadgroup_size: u64,
    /// Recent execution times (most recent last)
    performance_history: VecDeque<f64>,
    /// Number of samples to track
    history_capacity: usize,
    /// Threshold for detecting performance degradation (0.7 = 30% slowdown)
    degradation_threshold: f64,
    /// Threshold for detecting improvement (can increase threadgroup size)
    improvement_threshold: f64,
    /// Consecutive samples required before changing configuration
    required_samples: usize,
    /// Consecutive degradation samples observed
    degradation_count: usize,
    /// Consecutive improvement samples observed
    improvement_count: usize,
}

impl ThermalScheduler {
    /// Create a new thermal scheduler.
    ///
    /// # Arguments
    /// * `initial_size` - Starting threadgroup size
    /// * `max_size` - Maximum allowed (from `max_total_threads_per_threadgroup`)
    /// * `min_size` - Minimum allowed (avoid pathological cases)
    pub fn new(initial_size: u64, max_size: u64, min_size: u64) -> Self {
        ThermalScheduler {
            threadgroup_size: initial_size,
            max_threadgroup_size: max_size,
            min_threadgroup_size: min_size.max(32), // At least 32
            performance_history: VecDeque::with_capacity(10),
            history_capacity: 10,
            degradation_threshold: 0.75, // 25% slowdown = throttling likely
            improvement_threshold: 0.95, // 5% improvement = thermals recovered
            required_samples: 2,         // Require 2 consecutive samples
            degradation_count: 0,
            improvement_count: 0,
        }
    }

    /// Update with new execution time and optionally return adjusted threadgroup size.
    ///
    /// # Returns
    /// * `Some(new_size)` - if threadgroup size should be adjusted
    /// * `None` - if no adjustment needed
    ///
    /// # Arguments
    /// * `elapsed_ms` - Execution time in milliseconds
    pub fn adjust(&mut self, elapsed_ms: f64) -> Option<u64> {
        // Add to history
        self.performance_history.push_back(elapsed_ms);
        if self.performance_history.len() > self.history_capacity {
            self.performance_history.pop_front();
        }

        // Need at least 3 samples to make decisions
        if self.performance_history.len() < 3 {
            return None;
        }

        // Calculate baseline (median of first half)
        let baseline = self.estimate_baseline();

        // Compare recent performance to baseline
        let recent = *self.performance_history.back().unwrap();
        let ratio = recent / baseline;

        // Detect degradation (likely thermal throttling)
        if ratio > 1.0 / self.degradation_threshold {
            self.degradation_count += 1;
            self.improvement_count = 0;

            if self.degradation_count >= self.required_samples {
                // Reduce threadgroup size to lower power consumption
                let new_size = (self.threadgroup_size * 3) / 4; // Reduce by 25%
                let new_size = new_size.max(self.min_threadgroup_size);

                if new_size != self.threadgroup_size {
                    self.threadgroup_size = new_size;
                    self.degradation_count = 0;
                    return Some(new_size);
                }
            }
        }
        // Detect improvement (thermals recovered, can increase size)
        else if ratio < self.improvement_threshold {
            self.improvement_count += 1;
            self.degradation_count = 0;

            if self.improvement_count >= self.required_samples {
                // Increase threadgroup size for better performance
                let new_size = (self.threadgroup_size * 5) / 4; // Increase by 25%
                let new_size = new_size.min(self.max_threadgroup_size);

                if new_size != self.threadgroup_size {
                    self.threadgroup_size = new_size;
                    self.improvement_count = 0;
                    return Some(new_size);
                }
            }
        } else {
            // Stable performance
            self.degradation_count = 0;
            self.improvement_count = 0;
        }

        None
    }

    /// Estimate baseline performance from recent history.
    ///
    /// Uses the median of the earlier samples (excluding most recent 2).
    fn estimate_baseline(&self) -> f64 {
        let len = self.performance_history.len();
        if len <= 2 {
            return self.performance_history.iter().copied().sum::<f64>() / len as f64;
        }

        // Use first (len - 2) samples for baseline
        let baseline_samples: Vec<_> = self
            .performance_history
            .iter()
            .take(len.saturating_sub(2))
            .copied()
            .collect();

        let n = baseline_samples.len();
        let mut sorted = baseline_samples;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Median
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    /// Get current threadgroup size.
    pub fn threadgroup_size(&self) -> u64 {
        self.threadgroup_size
    }

    /// Force reset to initial state (e.g., after long idle period).
    pub fn reset(&mut self, initial_size: u64) {
        self.threadgroup_size = initial_size;
        self.performance_history.clear();
        self.degradation_count = 0;
        self.improvement_count = 0;
    }

    /// Get current thermal state for debugging.
    pub fn thermal_state(&self) -> ThermalState {
        ThermalState {
            threadgroup_size: self.threadgroup_size,
            performance_baseline_ms: self.estimate_baseline(),
            recent_performance_ms: self.performance_history.back().copied(),
            degradation_count: self.degradation_count,
            improvement_count: self.improvement_count,
            max_threadgroup_size: self.max_threadgroup_size,
        }
    }

    /// Get maximum allowed threadgroup size.
    pub fn max_threadgroup_size(&self) -> u64 {
        self.max_threadgroup_size
    }
}

/// Current thermal state for debugging/monitoring.
#[derive(Debug, Clone, Copy)]
pub struct ThermalState {
    pub threadgroup_size: u64,
    pub performance_baseline_ms: f64,
    pub recent_performance_ms: Option<f64>,
    pub degradation_count: usize,
    pub improvement_count: usize,
    pub max_threadgroup_size: u64,
}

impl std::fmt::Display for ThermalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ThermalState {{ tg_size: {}/{} ({:.1}%), baseline: {:.2}ms, recent: {:?}ms }}",
            self.threadgroup_size,
            self.max_threadgroup_size,
            (self.threadgroup_size as f64 / self.max_threadgroup_size as f64) * 100.0,
            self.performance_baseline_ms,
            self.recent_performance_ms
        )
    }
}

/// Conservative scheduler that minimizes thermal throttling risk.
///
/// Starts with smaller threadgroup size and only increases cautiously.
#[derive(Debug, Clone)]
pub struct ConservativeScheduler {
    base: ThermalScheduler,
}

impl ConservativeScheduler {
    pub fn new(max_size: u64) -> Self {
        // Start at 50% of max to avoid immediate throttling
        let initial = (max_size / 2).max(64);

        ConservativeScheduler {
            base: ThermalScheduler::new(initial, max_size, 64),
        }
    }

    pub fn adjust(&mut self, elapsed_ms: f64) -> Option<u64> {
        // More conservative thresholds
        self.base.degradation_threshold = 0.85; // More sensitive to slowdown
        self.base.improvement_threshold = 0.98; // More conservative recovery
        self.base.required_samples = 3; // Require more samples

        self.base.adjust(elapsed_ms)
    }

    pub fn threadgroup_size(&self) -> u64 {
        self.base.threadgroup_size()
    }

    pub fn thermal_state(&self) -> ThermalState {
        self.base.thermal_state()
    }
}

/// Aggressive scheduler that maximizes performance.
///
/// Starts with large threadgroup size and reduces only when necessary.
#[derive(Debug, Clone)]
pub struct AggressiveScheduler {
    base: ThermalScheduler,
}

impl AggressiveScheduler {
    pub fn new(max_size: u64) -> Self {
        // Start at max for best initial performance
        AggressiveScheduler {
            base: ThermalScheduler::new(max_size, max_size, 128),
        }
    }

    pub fn adjust(&mut self, elapsed_ms: f64) -> Option<u64> {
        // More aggressive thresholds
        self.base.degradation_threshold = 0.6; // Only respond to severe slowdown
        self.base.improvement_threshold = 0.90; // Quick to recover
        self.base.required_samples = 1; // Respond immediately

        self.base.adjust(elapsed_ms)
    }

    pub fn threadgroup_size(&self) -> u64 {
        self.base.threadgroup_size()
    }

    pub fn thermal_state(&self) -> ThermalState {
        self.base.thermal_state()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = ThermalScheduler::new(256, 512, 64);
        assert_eq!(scheduler.threadgroup_size(), 256);
    }

    #[test]
    fn test_scheduler_degradation_detection() {
        let mut scheduler = ThermalScheduler::new(256, 512, 64);

        // Stable baseline
        for _ in 0..3 {
            scheduler.adjust(10.0);
        }

        // Degraded performance (slower)
        for _ in 0..2 {
            let result = scheduler.adjust(15.0); // 50% slower
                                                 // Should trigger reduction after 2 samples
            if let Some(new_size) = result {
                assert!(new_size < 256);
                return;
            }
        }

        panic!("Expected threadgroup size reduction");
    }

    #[test]
    fn test_scheduler_min_size_respected() {
        let mut scheduler = ThermalScheduler::new(64, 256, 64);

        // Already at minimum
        for _ in 0..5 {
            scheduler.adjust(100.0);
        }

        // Try to reduce further
        let result = scheduler.adjust(200.0);
        // Should not go below minimum
        assert!(result.map_or(true, |s| s >= 64));
    }

    #[test]
    fn test_conservative_scheduler_starts_lower() {
        let scheduler = ConservativeScheduler::new(512);
        assert!(scheduler.threadgroup_size() < 512);
        assert!(scheduler.threadgroup_size() >= 64);
    }

    #[test]
    fn test_aggressive_scheduler_starts_max() {
        let scheduler = AggressiveScheduler::new(512);
        assert_eq!(scheduler.threadgroup_size(), 512);
    }

    #[test]
    fn test_thermal_state_display() {
        let scheduler = ThermalScheduler::new(256, 512, 64);
        let state = scheduler.thermal_state();
        assert!(format!("{}", state).contains("ThermalState"));
    }
}
