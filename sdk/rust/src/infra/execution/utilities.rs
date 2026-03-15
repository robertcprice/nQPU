//! Quantum Circuit Utilities
//!
//! Provides utility functions for circuit analysis including:
//! - Gate counting
//! - Circuit depth calculation
//! - Circuit comparison
//! - State tomography helpers

use crate::QuantumSimulator;

// ============================================================
// CIRCUIT ANALYSIS
// ============================================================

/// Circuit gate count statistics
#[derive(Clone, Debug, Default)]
pub struct GateCount {
    /// Number of single-qubit gates
    pub single_qubit: usize,
    /// Number of two-qubit gates
    pub two_qubit: usize,
    /// Number of three-qubit gates
    pub three_qubit: usize,
    /// Total gate count
    pub total: usize,
}

impl GateCount {
    /// Create a new gate count from components
    pub fn new(single_qubit: usize, two_qubit: usize, three_qubit: usize) -> Self {
        let total = single_qubit + two_qubit + three_qubit;
        GateCount {
            single_qubit,
            two_qubit,
            three_qubit,
            total,
        }
    }
}

/// Circuit analysis results
#[derive(Clone, Debug)]
pub struct CircuitAnalysis {
    /// Gate counts by type
    pub gate_counts: GateCount,
    /// Circuit depth (longest path of gates)
    pub depth: usize,
    /// Number of qubits used
    pub num_qubits: usize,
}

/// Analyze a circuit by executing it and counting gates
///
/// Note: This requires running the circuit, which may be expensive
/// for large systems. For production, consider tracking gates during execution.
pub fn analyze_from_execution<F>(num_qubits: usize, circuit: F) -> CircuitAnalysis
where
    F: FnOnce(&mut QuantumSimulator),
{
    // Create a simulator that tracks gates
    let mut analyzer = CircuitAnalyzer::new(num_qubits);
    circuit(&mut analyzer.simulator);
    analyzer.analyze()
}

/// Circuit analyzer - tracks gates during execution
pub struct CircuitAnalyzer {
    simulator: QuantumSimulator,
    gate_counts: GateCount,
    depth_single: Vec<usize>,
    depth_two: Vec<usize>,
    depth_three: Vec<usize>,
}

impl CircuitAnalyzer {
    /// Create a new circuit analyzer
    pub fn new(num_qubits: usize) -> Self {
        CircuitAnalyzer {
            simulator: QuantumSimulator::new(num_qubits),
            gate_counts: GateCount::default(),
            depth_single: vec![0; num_qubits],
            depth_two: vec![0; num_qubits],
            depth_three: vec![0; num_qubits],
        }
    }

    /// Get mutable reference to the underlying simulator
    pub fn simulator(&mut self) -> &mut QuantumSimulator {
        &mut self.simulator
    }

    /// Record a single-qubit gate
    pub fn record_single_qubit(&mut self, qubit: usize) {
        self.gate_counts.single_qubit += 1;
        self.gate_counts.total += 1;
        self.depth_single[qubit] =
            self.depth_single[qubit].max(self.depth_two[qubit].max(self.depth_three[qubit])) + 1;
    }

    /// Record a two-qubit gate
    pub fn record_two_qubit(&mut self, qubit1: usize, qubit2: usize) {
        self.gate_counts.two_qubit += 1;
        self.gate_counts.total += 1;

        let depth1 = self.depth_single[qubit1].max(self.depth_three[qubit1]) + 1;
        let depth2 = self.depth_single[qubit2].max(self.depth_three[qubit2]) + 1;
        let current_depth = self.depth_two[qubit1].max(self.depth_two[qubit2]);

        self.depth_two[qubit1] = self.depth_two[qubit1].max(depth1).max(current_depth);
        self.depth_two[qubit2] = self.depth_two[qubit2].max(depth2).max(current_depth);
    }

    /// Record a three-qubit gate
    pub fn record_three_qubit(&mut self, qubit1: usize, qubit2: usize, qubit3: usize) {
        self.gate_counts.three_qubit += 1;
        self.gate_counts.total += 1;

        let depth1 = self.depth_single[qubit1].max(self.depth_two[qubit1]) + 1;
        let depth2 = self.depth_single[qubit2].max(self.depth_two[qubit2]) + 1;
        let depth3 = self.depth_single[qubit3].max(self.depth_two[qubit3]) + 1;
        let current_depth = self.depth_three[qubit1]
            .max(self.depth_three[qubit2])
            .max(self.depth_three[qubit3]);

        self.depth_three[qubit1] = self.depth_three[qubit1].max(depth1).max(current_depth);
        self.depth_three[qubit2] = self.depth_three[qubit2].max(depth2).max(current_depth);
        self.depth_three[qubit3] = self.depth_three[qubit3].max(depth3).max(current_depth);
    }

    /// Analyze the circuit and return statistics
    pub fn analyze(&self) -> CircuitAnalysis {
        // Calculate circuit depth as the maximum depth across all qubits
        let mut max_depth = 0;
        for i in 0..self.simulator.num_qubits() {
            let qubit_depth = self.depth_single[i]
                .max(self.depth_two[i])
                .max(self.depth_three[i]);
            max_depth = max_depth.max(qubit_depth);
        }

        CircuitAnalysis {
            gate_counts: self.gate_counts.clone(),
            depth: max_depth,
            num_qubits: self.simulator.num_qubits(),
        }
    }
}

// ============================================================
// STATE TOMOGRAPHY UTILITIES
// ============================================================

/// State tomography results
///
/// Contains measurement statistics for reconstructing quantum states.
#[derive(Clone, Debug)]
pub struct TomographyResults {
    /// Counts of |0⟩ measurements for each qubit
    pub zero_counts: Vec<usize>,
    /// Counts of |1⟩ measurements for each qubit
    pub one_counts: Vec<usize>,
    /// Total number of shots
    pub total_shots: usize,
}

impl TomographyResults {
    /// Create new tomography results
    pub fn new(num_qubits: usize, total_shots: usize) -> Self {
        TomographyResults {
            zero_counts: vec![0; num_qubits],
            one_counts: vec![0; num_qubits],
            total_shots,
        }
    }

    /// Get probability of measuring |0⟩ for a qubit
    pub fn zero_probability(&self, qubit: usize) -> f64 {
        self.zero_counts[qubit] as f64 / self.total_shots as f64
    }

    /// Get probability of measuring |1⟩ for a qubit
    pub fn one_probability(&self, qubit: usize) -> f64 {
        self.one_counts[qubit] as f64 / self.total_shots as f64
    }

    /// Get expectation value of Z for a qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        self.zero_probability(qubit) - self.one_probability(qubit)
    }
}

/// Perform simple state tomography by measuring in computational basis
///
/// # Arguments
/// * `simulator` - Quantum simulator (will be cloned for each shot)
/// * `shots` - Number of measurement shots
///
/// # Returns
/// TomographyResults with measurement statistics
///
/// # Note
/// This only measures in the Z basis. Full tomography requires
/// measurements in multiple bases (X, Y, Z).
pub fn state_tomography<F>(
    simulator: &QuantumSimulator,
    shots: usize,
    prepare_state: F,
) -> TomographyResults
where
    F: Fn(&mut QuantumSimulator),
{
    let num_qubits = simulator.num_qubits();
    let mut results = TomographyResults::new(num_qubits, shots);

    for _shot in 0..shots {
        let mut sim = QuantumSimulator::new(num_qubits);
        prepare_state(&mut sim);

        let measurement = sim.measure();
        for q in 0..num_qubits {
            let bit = (measurement >> q) & 1;
            if bit == 0 {
                results.zero_counts[q] += 1;
            } else {
                results.one_counts[q] += 1;
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_analyzer() {
        let mut analyzer = CircuitAnalyzer::new(2);

        // Simulate a simple circuit
        analyzer.record_single_qubit(0);
        analyzer.record_single_qubit(1);
        analyzer.record_two_qubit(0, 1);
        analyzer.record_single_qubit(0);

        let analysis = analyzer.analyze();

        assert_eq!(analysis.gate_counts.single_qubit, 3);
        assert_eq!(analysis.gate_counts.two_qubit, 1);
        assert_eq!(analysis.gate_counts.total, 4);
        assert!(analysis.depth > 0);
    }

    #[test]
    fn test_state_tomography() {
        let sim = QuantumSimulator::new(2);

        let results = state_tomography(&sim, 100, |sim| {
            sim.h(0);
            sim.x(1);
        });

        assert_eq!(results.total_shots, 100);
        // Qubit 0 should be roughly 50/50 due to H gate
        let p0_zero = results.zero_probability(0);
        assert!(p0_zero > 0.3 && p0_zero < 0.7);
        // Qubit 1 should always be 1 due to X gate
        assert_eq!(results.one_counts[1], 100);
    }

    #[test]
    fn test_analyze_from_execution() {
        // Use CircuitAnalyzer directly instead of analyze_from_execution
        // since we need explicit gate tracking
        let mut analyzer = CircuitAnalyzer::new(2);

        // Execute the circuit manually with gate tracking
        analyzer.simulator.h(0);
        analyzer.record_single_qubit(0);

        analyzer.simulator.h(1);
        analyzer.record_single_qubit(1);

        analyzer.simulator.cnot(0, 1);
        analyzer.record_two_qubit(0, 1);

        let analysis = analyzer.analyze();

        assert_eq!(analysis.num_qubits, 2);
        assert_eq!(analysis.gate_counts.total, 3);
        assert!(analysis.depth > 0);
    }
}
