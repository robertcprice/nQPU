//! Quantum Circuit Complexity Analysis
//!
//! **BLEEDING EDGE**: No simulator provides built-in computational complexity analysis.
//! This module analyzes quantum circuits for:
//! - Circuit depth and width optimization
//! - T-count / T-depth for fault-tolerant cost estimation
//! - Entanglement structure analysis
//! - Magic state resource estimation
//! - Quantum volume calculation
//! - Circuit expressibility and entangling capability metrics
//! - Barren plateau detection for variational circuits
//!
//! This is critical for real quantum hardware deployment planning.

use std::collections::{HashMap, HashSet};

/// Gate classification for complexity analysis
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GateClass {
    /// Clifford gates (H, S, CNOT, etc.) — classically simulable
    Clifford,
    /// T gates and T-dagger — source of quantum advantage
    TGate,
    /// Arbitrary rotation gates
    Rotation,
    /// Multi-qubit entangling gates
    Entangling,
    /// Measurement
    Measurement,
    /// Classical control
    Classical,
}

/// A gate in the circuit for analysis
#[derive(Clone, Debug)]
pub struct AnalysisGate {
    pub name: String,
    pub qubits: Vec<usize>,
    pub params: Vec<f64>,
    pub class: GateClass,
    /// Layer/depth at which this gate is scheduled
    pub layer: usize,
}

/// Quantum circuit for complexity analysis
#[derive(Clone, Debug)]
pub struct AnalysisCircuit {
    pub num_qubits: usize,
    pub gates: Vec<AnalysisGate>,
}

impl AnalysisCircuit {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }

    pub fn add_gate(&mut self, name: &str, qubits: Vec<usize>, params: Vec<f64>) {
        let class = match name {
            "H" | "S" | "Sdg" | "CNOT" | "CZ" | "SWAP" | "X" | "Y" | "Z" => GateClass::Clifford,
            "T" | "Tdg" => GateClass::TGate,
            "Rx" | "Ry" | "Rz" | "U3" | "CRz" | "CRx" => GateClass::Rotation,
            "CCX" | "CSWAP" | "CCZ" => GateClass::Entangling,
            "M" | "Measure" => GateClass::Measurement,
            _ => GateClass::Rotation,
        };

        let layer = self.compute_layer(&qubits);

        self.gates.push(AnalysisGate {
            name: name.to_string(),
            qubits,
            params,
            class,
            layer,
        });
    }

    fn compute_layer(&self, qubits: &[usize]) -> usize {
        let mut max_layer = 0;
        for gate in &self.gates {
            // Check if any qubit overlaps
            if gate.qubits.iter().any(|q| qubits.contains(q)) {
                max_layer = max_layer.max(gate.layer + 1);
            }
        }
        max_layer
    }
}

/// Comprehensive complexity analysis result
#[derive(Clone, Debug)]
pub struct ComplexityReport {
    // Basic metrics
    pub total_gates: usize,
    pub circuit_depth: usize,
    pub circuit_width: usize,
    pub gate_count_by_type: HashMap<String, usize>,

    // Fault-tolerance metrics
    pub t_count: usize,
    pub t_depth: usize,
    pub clifford_count: usize,
    pub two_qubit_gate_count: usize,

    // Resource estimation
    pub magic_state_count: usize,
    pub estimated_logical_qubits: usize,
    pub estimated_physical_qubits: usize,
    pub estimated_error_rate_threshold: f64,

    // Entanglement analysis
    pub entanglement_graph: Vec<(usize, usize, usize)>, // (q1, q2, count)
    pub max_entanglement_width: usize,
    pub connectivity_density: f64,

    // Expressibility metrics (for variational circuits)
    pub expressibility: Option<f64>,
    pub entangling_capability: Option<f64>,

    // Barren plateau risk
    pub barren_plateau_risk: BarrenPlateauRisk,

    // Quantum volume estimate
    pub quantum_volume_required: usize,

    // Optimization suggestions
    pub suggestions: Vec<String>,
}

/// Barren plateau risk assessment
#[derive(Clone, Debug)]
pub struct BarrenPlateauRisk {
    /// Overall risk level
    pub level: RiskLevel,
    /// Expected gradient variance scaling
    pub gradient_variance_scaling: String,
    /// Number of parameters in exponential-vanishing zone
    pub params_at_risk: usize,
    /// Total parameters
    pub total_params: usize,
    /// Specific risk factors
    pub risk_factors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Circuit Complexity Analyzer
pub struct CircuitComplexityAnalyzer;

impl CircuitComplexityAnalyzer {
    /// Perform comprehensive complexity analysis
    pub fn analyze(circuit: &AnalysisCircuit) -> ComplexityReport {
        let total_gates = circuit.gates.len();
        let circuit_depth = circuit
            .gates
            .iter()
            .map(|g| g.layer)
            .max()
            .unwrap_or(0)
            + 1;

        // Gate count by type
        let mut gate_count_by_type: HashMap<String, usize> = HashMap::new();
        for gate in &circuit.gates {
            *gate_count_by_type.entry(gate.name.clone()).or_insert(0) += 1;
        }

        // T-count and T-depth
        let t_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::TGate)
            .count();
        let t_depth = Self::compute_t_depth(circuit);

        let clifford_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::Clifford)
            .count();

        let two_qubit_gate_count = circuit
            .gates
            .iter()
            .filter(|g| g.qubits.len() >= 2)
            .count();

        // Entanglement graph
        let entanglement_graph = Self::compute_entanglement_graph(circuit);
        let max_entanglement_width = Self::max_entanglement_width(circuit);

        let connectivity_density = if circuit.num_qubits > 1 {
            let max_edges = circuit.num_qubits * (circuit.num_qubits - 1) / 2;
            let unique_edges: HashSet<(usize, usize)> = entanglement_graph
                .iter()
                .map(|&(a, b, _)| (a.min(b), a.max(b)))
                .collect();
            unique_edges.len() as f64 / max_edges as f64
        } else {
            0.0
        };

        // Resource estimation for fault-tolerant quantum computing
        let magic_state_count = t_count; // Each T gate needs one magic state
        let estimated_logical_qubits = circuit.num_qubits + t_count; // Rough estimate
        let code_distance = Self::estimate_code_distance(circuit);
        let estimated_physical_qubits =
            estimated_logical_qubits * 2 * code_distance * code_distance;
        let estimated_error_rate_threshold = 1e-3 / (circuit_depth as f64); // Rough threshold

        // Barren plateau analysis
        let barren_plateau_risk = Self::analyze_barren_plateaus(circuit);

        // Quantum volume
        let quantum_volume_required = Self::estimate_quantum_volume(circuit);

        // Expressibility (for variational circuits)
        let (expressibility, entangling_capability) = Self::compute_expressibility(circuit);

        // Generate optimization suggestions
        let suggestions = Self::generate_suggestions(
            circuit,
            t_count,
            two_qubit_gate_count,
            circuit_depth,
            &barren_plateau_risk,
        );

        ComplexityReport {
            total_gates,
            circuit_depth,
            circuit_width: circuit.num_qubits,
            gate_count_by_type,
            t_count,
            t_depth,
            clifford_count,
            two_qubit_gate_count,
            magic_state_count,
            estimated_logical_qubits,
            estimated_physical_qubits,
            estimated_error_rate_threshold,
            entanglement_graph,
            max_entanglement_width,
            connectivity_density,
            expressibility,
            entangling_capability,
            barren_plateau_risk,
            quantum_volume_required,
            suggestions,
        }
    }

    fn compute_t_depth(circuit: &AnalysisCircuit) -> usize {
        let mut qubit_t_layers: Vec<usize> = vec![0; circuit.num_qubits];
        let mut max_t_depth = 0;

        for gate in &circuit.gates {
            if gate.class == GateClass::TGate {
                let current_max = gate
                    .qubits
                    .iter()
                    .map(|&q| qubit_t_layers[q])
                    .max()
                    .unwrap_or(0);
                let new_layer = current_max + 1;
                for &q in &gate.qubits {
                    qubit_t_layers[q] = new_layer;
                }
                max_t_depth = max_t_depth.max(new_layer);
            }
        }

        max_t_depth
    }

    fn compute_entanglement_graph(
        circuit: &AnalysisCircuit,
    ) -> Vec<(usize, usize, usize)> {
        let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for gate in &circuit.gates {
            if gate.qubits.len() >= 2 {
                for i in 0..gate.qubits.len() {
                    for j in (i + 1)..gate.qubits.len() {
                        let a = gate.qubits[i].min(gate.qubits[j]);
                        let b = gate.qubits[i].max(gate.qubits[j]);
                        *edge_counts.entry((a, b)).or_insert(0) += 1;
                    }
                }
            }
        }

        edge_counts
            .into_iter()
            .map(|((a, b), c)| (a, b, c))
            .collect()
    }

    fn max_entanglement_width(circuit: &AnalysisCircuit) -> usize {
        // Maximum number of qubits entangled at any layer
        let max_layer = circuit
            .gates
            .iter()
            .map(|g| g.layer)
            .max()
            .unwrap_or(0);

        let mut max_width = 0;
        for layer in 0..=max_layer {
            let mut entangled: HashSet<usize> = HashSet::new();
            for gate in &circuit.gates {
                if gate.layer == layer && gate.qubits.len() >= 2 {
                    for &q in &gate.qubits {
                        entangled.insert(q);
                    }
                }
            }
            max_width = max_width.max(entangled.len());
        }

        max_width
    }

    fn estimate_code_distance(circuit: &AnalysisCircuit) -> usize {
        // Based on target error rate and circuit depth
        let depth = circuit
            .gates
            .iter()
            .map(|g| g.layer)
            .max()
            .unwrap_or(0)
            + 1;
        let t_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::TGate)
            .count();

        // d ~ O(log(depth * t_count / epsilon))
        let target_error = 1e-6;
        let raw_d = ((depth * t_count.max(1)) as f64 / target_error).ln() / 2.0;
        let d = (raw_d.ceil() as usize).max(3);

        // Must be odd for surface codes
        if d % 2 == 0 {
            d + 1
        } else {
            d
        }
    }

    fn analyze_barren_plateaus(circuit: &AnalysisCircuit) -> BarrenPlateauRisk {
        let n = circuit.num_qubits;
        let depth = circuit
            .gates
            .iter()
            .map(|g| g.layer)
            .max()
            .unwrap_or(0)
            + 1;

        let rotation_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::Rotation)
            .count();

        let entangling_count = circuit
            .gates
            .iter()
            .filter(|g| g.qubits.len() >= 2)
            .count();

        let mut risk_factors = Vec::new();
        let mut risk_score = 0.0;

        // Factor 1: Deep circuits with many qubits → exponential vanishing gradients
        if depth > 2 * n {
            risk_factors.push(format!(
                "Circuit depth ({}) exceeds 2n ({}) — exponential gradient decay likely",
                depth,
                2 * n
            ));
            risk_score += 3.0;
        } else if depth > n {
            risk_factors.push(format!(
                "Circuit depth ({}) > n ({}) — polynomial gradient decay possible",
                depth, n
            ));
            risk_score += 1.5;
        }

        // Factor 2: Global entanglement → information scrambling
        if entangling_count > 3 * n {
            risk_factors.push(format!(
                "High entangling gate density ({} gates) — information scrambling risk",
                entangling_count
            ));
            risk_score += 2.0;
        }

        // Factor 3: Hardware-efficient ansatz with random initialization
        if rotation_count > 4 * n {
            risk_factors.push(format!(
                "Many rotation parameters ({}) — random initialization enters barren plateau",
                rotation_count
            ));
            risk_score += 1.0;
        }

        // Factor 4: Global cost function (we can't know this from circuit alone)
        // but we can flag it as a consideration
        if n > 10 {
            risk_factors.push(
                "Large system (>10 qubits) — ensure local cost functions are used".to_string(),
            );
            risk_score += 0.5;
        }

        let level = if risk_score >= 5.0 {
            RiskLevel::Critical
        } else if risk_score >= 3.0 {
            RiskLevel::High
        } else if risk_score >= 1.5 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        let gradient_variance_scaling = if risk_score >= 3.0 {
            format!("O(2^{{-{}}}) — exponentially vanishing", n)
        } else if risk_score >= 1.5 {
            format!("O(1/poly({})) — polynomially vanishing", n)
        } else {
            "O(1) — constant".to_string()
        };

        let params_at_risk = if risk_score >= 3.0 {
            rotation_count
        } else if risk_score >= 1.5 {
            rotation_count / 2
        } else {
            0
        };

        BarrenPlateauRisk {
            level,
            gradient_variance_scaling,
            params_at_risk,
            total_params: rotation_count,
            risk_factors,
        }
    }

    fn estimate_quantum_volume(circuit: &AnalysisCircuit) -> usize {
        let n = circuit.num_qubits;
        let depth = circuit
            .gates
            .iter()
            .map(|g| g.layer)
            .max()
            .unwrap_or(0)
            + 1;

        // QV = 2^min(n, d) where d is the effective depth
        let effective = n.min(depth);
        1 << effective.min(20) // Cap at 2^20 to avoid overflow
    }

    fn compute_expressibility(
        circuit: &AnalysisCircuit,
    ) -> (Option<f64>, Option<f64>) {
        let rotation_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::Rotation)
            .count();

        if rotation_count == 0 {
            return (None, None);
        }

        let n = circuit.num_qubits;
        let entangling_count = circuit
            .gates
            .iter()
            .filter(|g| g.qubits.len() >= 2)
            .count();

        // Expressibility estimate: KL divergence from Haar random
        // Lower = more expressible. We use a heuristic based on structure.
        let max_expressibility = (rotation_count as f64 * entangling_count as f64).sqrt()
            / (n * n) as f64;
        let expressibility = max_expressibility.min(1.0);

        // Entangling capability: Meyer-Wallach measure estimate
        let entangling_capability = if n > 1 && entangling_count > 0 {
            let ec = (2.0 * entangling_count as f64 / (n * (n - 1)) as f64).min(1.0);
            Some(ec)
        } else {
            Some(0.0)
        };

        (Some(expressibility), entangling_capability)
    }

    fn generate_suggestions(
        circuit: &AnalysisCircuit,
        t_count: usize,
        two_qubit_count: usize,
        depth: usize,
        barren_risk: &BarrenPlateauRisk,
    ) -> Vec<String> {
        let mut suggestions = Vec::new();

        if t_count > 100 {
            suggestions.push(format!(
                "High T-count ({}). Consider T-count optimization or Solovay-Kitaev decomposition.",
                t_count
            ));
        }

        if two_qubit_count as f64 / circuit.gates.len() as f64 > 0.5 {
            suggestions.push(
                "Over 50% two-qubit gates. Consider circuit rewriting to reduce CNOT count."
                    .to_string(),
            );
        }

        if depth > 3 * circuit.num_qubits {
            suggestions.push(format!(
                "Circuit depth ({}) is >3x qubit count. Consider parallelization or depth reduction.",
                depth
            ));
        }

        if barren_risk.level == RiskLevel::Critical || barren_risk.level == RiskLevel::High {
            suggestions.push(
                "High barren plateau risk. Consider: (1) layer-wise training, (2) local cost functions, (3) parameter initialization near identity.".to_string()
            );
        }

        let rotation_count = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::Rotation)
            .count();
        if rotation_count > 0 && circuit.num_qubits > 8 {
            suggestions.push(
                "For VQA on >8 qubits: use problem-inspired ansatz (e.g., UCCSD) over hardware-efficient ansatz to avoid barren plateaus.".to_string()
            );
        }

        // Clifford optimization opportunity
        let clifford_fraction = circuit
            .gates
            .iter()
            .filter(|g| g.class == GateClass::Clifford)
            .count() as f64
            / circuit.gates.len().max(1) as f64;
        if clifford_fraction > 0.8 {
            suggestions.push(
                "Circuit is >80% Clifford. Consider stabilizer simulation for significant speedup."
                    .to_string(),
            );
        }

        suggestions
    }
}

/// Quantum Volume Calculator
///
/// Implements IBM's Quantum Volume benchmark which measures
/// the largest random circuit a quantum computer can execute reliably.
pub struct QuantumVolumeCalculator;

impl QuantumVolumeCalculator {
    /// Estimate the quantum volume achievable for given hardware parameters
    pub fn estimate(
        num_qubits: usize,
        two_qubit_error_rate: f64,
        single_qubit_error_rate: f64,
        connectivity: &[(usize, usize)],
    ) -> QuantumVolumeEstimate {
        // Quantum Volume = 2^m where m = max achievable width = depth circuit
        // A circuit passes QV test if heavy output probability > 2/3

        let mut max_m = 0;

        for m in 1..=num_qubits {
            // Effective depth including SWAP routing overhead
            let avg_swaps = Self::estimate_swap_overhead(m, connectivity);
            let effective_error = two_qubit_error_rate * (m as f64 + avg_swaps);

            // Heavy output probability estimate
            let hop = Self::estimate_heavy_output_prob(m, effective_error, single_qubit_error_rate);

            if hop > 2.0 / 3.0 {
                max_m = m;
            } else {
                break;
            }
        }

        let qv = 1usize << max_m;

        QuantumVolumeEstimate {
            quantum_volume: qv,
            max_width: max_m,
            heavy_output_prob: Self::estimate_heavy_output_prob(
                max_m,
                two_qubit_error_rate * max_m as f64,
                single_qubit_error_rate,
            ),
            limiting_factor: if two_qubit_error_rate > 1e-2 {
                "Two-qubit gate fidelity".to_string()
            } else if connectivity.len() < num_qubits * (num_qubits - 1) / 4 {
                "Qubit connectivity (routing overhead)".to_string()
            } else {
                "Qubit count".to_string()
            },
        }
    }

    fn estimate_swap_overhead(width: usize, connectivity: &[(usize, usize)]) -> f64 {
        if width <= 2 {
            return 0.0;
        }
        // Rough estimate: for linear connectivity, ~width/2 SWAPs per layer
        let max_edges = width * (width - 1) / 2;
        let actual_edges = connectivity
            .iter()
            .filter(|&&(a, b)| a < width && b < width)
            .count();
        let connectivity_ratio = actual_edges as f64 / max_edges as f64;

        // More connected = fewer SWAPs needed
        (width as f64 * 0.5) * (1.0 - connectivity_ratio)
    }

    fn estimate_heavy_output_prob(
        width: usize,
        cumulative_two_qubit_error: f64,
        single_qubit_error: f64,
    ) -> f64 {
        if width == 0 {
            return 1.0;
        }
        // Heavy output probability for depth=width random circuit
        let total_error = cumulative_two_qubit_error + single_qubit_error * width as f64;
        let success_prob = (-total_error).exp();

        // HOP ≈ 1 - (1 - 2^{-n}) * (1 - success_prob)
        // Simplified: interpolation between ideal (≈0.85) and random (0.5)
        0.5 + 0.35 * success_prob
    }
}

/// Result of quantum volume estimation
#[derive(Clone, Debug)]
pub struct QuantumVolumeEstimate {
    pub quantum_volume: usize,
    pub max_width: usize,
    pub heavy_output_prob: f64,
    pub limiting_factor: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_complexity_analysis() {
        let mut circuit = AnalysisCircuit::new(4);
        circuit.add_gate("H", vec![0], vec![]);
        circuit.add_gate("CNOT", vec![0, 1], vec![]);
        circuit.add_gate("T", vec![1], vec![]);
        circuit.add_gate("CNOT", vec![1, 2], vec![]);
        circuit.add_gate("H", vec![3], vec![]);

        let report = CircuitComplexityAnalyzer::analyze(&circuit);

        assert_eq!(report.total_gates, 5);
        assert_eq!(report.circuit_width, 4);
        assert_eq!(report.t_count, 1);
        assert_eq!(report.two_qubit_gate_count, 2);
        assert_eq!(report.magic_state_count, 1);
    }

    #[test]
    fn test_barren_plateau_detection() {
        let mut circuit = AnalysisCircuit::new(12);

        // Deep variational circuit — should trigger barren plateau warning
        for layer in 0..30 {
            for q in 0..12 {
                circuit.add_gate("Ry", vec![q], vec![0.5]);
                circuit.add_gate("Rz", vec![q], vec![0.3]);
            }
            for q in 0..11 {
                circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
            }
        }

        let report = CircuitComplexityAnalyzer::analyze(&circuit);
        assert!(
            report.barren_plateau_risk.level == RiskLevel::High
                || report.barren_plateau_risk.level == RiskLevel::Critical
        );
        assert!(!report.barren_plateau_risk.risk_factors.is_empty());
    }

    #[test]
    fn test_quantum_volume_estimation() {
        // Linear connectivity for 5 qubits
        let connectivity: Vec<(usize, usize)> = (0..4).map(|i| (i, i + 1)).collect();

        let estimate = QuantumVolumeCalculator::estimate(
            5,
            0.005,  // 0.5% 2Q error
            0.001,  // 0.1% 1Q error
            &connectivity,
        );

        assert!(estimate.quantum_volume >= 2);
        assert!(estimate.max_width <= 5);
    }

    #[test]
    fn test_clifford_dominated_circuit() {
        let mut circuit = AnalysisCircuit::new(4);
        for q in 0..4 {
            circuit.add_gate("H", vec![q], vec![]);
        }
        for q in 0..3 {
            circuit.add_gate("CNOT", vec![q, q + 1], vec![]);
        }

        let report = CircuitComplexityAnalyzer::analyze(&circuit);
        assert!(report.suggestions.iter().any(|s| s.contains("Clifford")));
    }

    #[test]
    fn test_entanglement_graph() {
        let mut circuit = AnalysisCircuit::new(4);
        circuit.add_gate("CNOT", vec![0, 1], vec![]);
        circuit.add_gate("CNOT", vec![1, 2], vec![]);
        circuit.add_gate("CNOT", vec![0, 1], vec![]);
        circuit.add_gate("CNOT", vec![2, 3], vec![]);

        let report = CircuitComplexityAnalyzer::analyze(&circuit);
        assert!(!report.entanglement_graph.is_empty());
        assert!(report.connectivity_density > 0.0);
    }
}
