//! Stim Circuit Format Import/Export
//!
//! This module provides interoperability with Google's Stim library,
//! the de-facto standard for stabilizer circuit simulation in QEC research.
//!
//! # Supported Operations
//!
//! - Parsing Stim circuit format text
//! - Exporting to Stim circuit format
//! - Conversion to/from internal gate representations
//!
//! # Stim Format Example
//!
//! ```stim
//! # Bell state preparation
//! H 0
//! CX 0 1
//! M 0 1
//!
//! # Surface code cycle
//! R 0 1 2 3 4
//! H 0 2 4
//! CX 0 1 2 3 4 5
//! M 0 1 2 3 4
//! ```
//!
//! # Reference
//!
//! Gidney, C. (2021). "Stim: a fast stabilizer circuit simulator"
//! https://github.com/quantumlib/Stim

use std::collections::HashMap;
use std::fmt;

/// Stim gate operation
#[derive(Clone, Debug, PartialEq)]
pub enum StimGate {
    /// Hadamard
    H { targets: Vec<usize> },
    /// Phase gate (S)
    S { targets: Vec<usize> },
    /// S† gate
    Sdg { targets: Vec<usize> },
    /// T gate
    T { targets: Vec<usize> },
    /// T† gate
    Tdg { targets: Vec<usize> },
    /// Pauli X
    X { targets: Vec<usize> },
    /// Pauli Y
    Y { targets: Vec<usize> },
    /// Pauli Z
    Z { targets: Vec<usize> },
    /// CNOT (CX)
    CX { pairs: Vec<(usize, usize)> },
    /// CY
    CY { pairs: Vec<(usize, usize)> },
    /// CZ
    CZ { pairs: Vec<(usize, usize)> },
    /// SWAP
    SWAP { pairs: Vec<(usize, usize)> },
    /// ISWAP
    ISWAP { pairs: Vec<(usize, usize)> },
    /// Measure in Z basis
    M { targets: Vec<usize> },
    /// Measure in X basis
    MX { targets: Vec<usize> },
    /// Measure in Y basis
    MY { targets: Vec<usize> },
    /// Reset to |0⟩
    R { targets: Vec<usize> },
    /// Reset to |+⟩
    RX { targets: Vec<usize> },
    /// Reset to |+i⟩
    RY { targets: Vec<usize> },
    /// Pauli channel (noise)
    PauliChannel1 { target: usize, px: f64, py: f64, pz: f64 },
    /// Depolarizing noise
    Depolarize1 { targets: Vec<usize>, p: f64 },
    Depolarize2 { pairs: Vec<(usize, usize)>, p: f64 },
    /// Measurement error
    XError { targets: Vec<usize>, p: f64 },
    ZError { targets: Vec<usize>, p: f64 },
    /// Tick (circuit boundary)
    Tick,
    /// Detector declaration
    Detector { targets: Vec<usize> },
    /// Observable declaration
    ObservableInclude { observable: usize, targets: Vec<usize> },
}

impl StimGate {
    /// Get the qubits involved in this gate
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            StimGate::H { targets } |
            StimGate::S { targets } |
            StimGate::Sdg { targets } |
            StimGate::T { targets } |
            StimGate::Tdg { targets } |
            StimGate::X { targets } |
            StimGate::Y { targets } |
            StimGate::Z { targets } |
            StimGate::M { targets } |
            StimGate::MX { targets } |
            StimGate::MY { targets } |
            StimGate::R { targets } |
            StimGate::RX { targets } |
            StimGate::RY { targets } |
            StimGate::Depolarize1 { targets, .. } |
            StimGate::XError { targets, .. } |
            StimGate::ZError { targets, .. } => targets.clone(),

            StimGate::CX { pairs } |
            StimGate::CY { pairs } |
            StimGate::CZ { pairs } |
            StimGate::SWAP { pairs } |
            StimGate::ISWAP { pairs } |
            StimGate::Depolarize2 { pairs, .. } => {
                let mut qs = Vec::new();
                for (a, b) in pairs {
                    qs.push(*a);
                    qs.push(*b);
                }
                qs
            }

            StimGate::PauliChannel1 { target, .. } => vec![*target],

            StimGate::Tick => vec![],
            StimGate::Detector { targets } => targets.clone(),
            StimGate::ObservableInclude { targets, .. } => targets.clone(),
        }
    }

    /// Check if this is a measurement gate
    pub fn is_measurement(&self) -> bool {
        matches!(self, StimGate::M { .. } | StimGate::MX { .. } | StimGate::MY { .. })
    }

    /// Check if this is a Clifford gate
    pub fn is_clifford(&self) -> bool {
        matches!(self,
            StimGate::H { .. } | StimGate::S { .. } | StimGate::Sdg { .. } |
            StimGate::X { .. } | StimGate::Y { .. } | StimGate::Z { .. } |
            StimGate::CX { .. } | StimGate::CY { .. } | StimGate::CZ { .. } |
            StimGate::SWAP { .. }
        )
    }
}

impl fmt::Display for StimGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StimGate::H { targets } => write!(f, "H {}", format_targets(targets)),
            StimGate::S { targets } => write!(f, "S {}", format_targets(targets)),
            StimGate::Sdg { targets } => write!(f, "S_DAG {}", format_targets(targets)),
            StimGate::T { targets } => write!(f, "T {}", format_targets(targets)),
            StimGate::Tdg { targets } => write!(f, "T_DAG {}", format_targets(targets)),
            StimGate::X { targets } => write!(f, "X {}", format_targets(targets)),
            StimGate::Y { targets } => write!(f, "Y {}", format_targets(targets)),
            StimGate::Z { targets } => write!(f, "Z {}", format_targets(targets)),
            StimGate::CX { pairs } => write!(f, "CX {}", format_pairs(pairs)),
            StimGate::CY { pairs } => write!(f, "CY {}", format_pairs(pairs)),
            StimGate::CZ { pairs } => write!(f, "CZ {}", format_pairs(pairs)),
            StimGate::SWAP { pairs } => write!(f, "SWAP {}", format_pairs(pairs)),
            StimGate::ISWAP { pairs } => write!(f, "ISWAP {}", format_pairs(pairs)),
            StimGate::M { targets } => write!(f, "M {}", format_targets(targets)),
            StimGate::MX { targets } => write!(f, "MX {}", format_targets(targets)),
            StimGate::MY { targets } => write!(f, "MY {}", format_targets(targets)),
            StimGate::R { targets } => write!(f, "R {}", format_targets(targets)),
            StimGate::RX { targets } => write!(f, "RX {}", format_targets(targets)),
            StimGate::RY { targets } => write!(f, "RY {}", format_targets(targets)),
            StimGate::PauliChannel1 { target, px, py, pz } => {
                write!(f, "PAULI_CHANNEL_1({},{},{}) {}", px, py, pz, target)
            }
            StimGate::Depolarize1 { targets, p } => {
                write!(f, "DEPOLARIZE1({}) {}", p, format_targets(targets))
            }
            StimGate::Depolarize2 { pairs, p } => {
                write!(f, "DEPOLARIZE2({}) {}", p, format_pairs(pairs))
            }
            StimGate::XError { targets, p } => {
                write!(f, "X_ERROR({}) {}", p, format_targets(targets))
            }
            StimGate::ZError { targets, p } => {
                write!(f, "Z_ERROR({}) {}", p, format_targets(targets))
            }
            StimGate::Tick => write!(f, "TICK"),
            StimGate::Detector { targets } => write!(f, "DETECTOR {}", format_targets(targets)),
            StimGate::ObservableInclude { observable, targets } => {
                write!(f, "OBSERVABLE_INCLUDE({}) {}", observable, format_targets(targets))
            }
        }
    }
}

fn format_targets(targets: &[usize]) -> String {
    targets.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ")
}

fn format_pairs(pairs: &[(usize, usize)]) -> String {
    pairs.iter().map(|(a, b)| format!("{} {}", a, b)).collect::<Vec<_>>().join(" ")
}

/// A Stim circuit
#[derive(Clone, Debug, Default)]
pub struct StimCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of measurements
    pub num_measurements: usize,
    /// Gate sequence
    pub gates: Vec<StimGate>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl StimCircuit {
    /// Create empty circuit
    pub fn new() -> Self {
        StimCircuit {
            num_qubits: 0,
            num_measurements: 0,
            gates: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create circuit with specified number of qubits
    pub fn with_qubits(n: usize) -> Self {
        StimCircuit {
            num_qubits: n,
            num_measurements: 0,
            gates: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a gate
    pub fn add_gate(&mut self, gate: StimGate) {
        // Update qubit count
        for q in gate.qubits() {
            self.num_qubits = self.num_qubits.max(q + 1);
        }

        // Update measurement count
        if gate.is_measurement() {
            self.num_measurements += gate.qubits().len();
        }

        self.gates.push(gate);
    }

    /// Parse from Stim format text
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut circuit = StimCircuit::new();
        let mut line_num = 0;

        for line in text.lines() {
            line_num += 1;
            let line = line.trim();

            // Skip empty lines and full-line comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Strip inline comments
            let line = if let Some(pos) = line.find('#') {
                &line[..pos].trim()
            } else {
                line
            };

            // Skip if only comment was on the line
            if line.is_empty() {
                continue;
            }

            // Parse the line
            match parse_line(line) {
                Ok(gates) => {
                    for gate in gates {
                        circuit.add_gate(gate);
                    }
                }
                Err(e) => {
                    return Err(format!("Line {}: {}", line_num, e));
                }
            }
        }

        Ok(circuit)
    }

    /// Export to Stim format text
    pub fn to_stim(&self) -> String {
        let mut output = String::new();

        // Add header comment
        output.push_str(&format!("# nQPU-Metal Stim Export\n"));
        output.push_str(&format!("# Qubits: {}\n", self.num_qubits));
        output.push_str(&format!("# Measurements: {}\n\n", self.num_measurements));

        for gate in &self.gates {
            output.push_str(&format!("{}\n", gate));
        }

        output
    }

    /// Get all measurements
    pub fn measurements(&self) -> Vec<&StimGate> {
        self.gates.iter().filter(|g| g.is_measurement()).collect()
    }

    /// Get all Clifford gates
    pub fn clifford_gates(&self) -> Vec<&StimGate> {
        self.gates.iter().filter(|g| g.is_clifford()).collect()
    }

    /// Count gates by type
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();

        for gate in &self.gates {
            let name = match gate {
                StimGate::H { .. } => "H",
                StimGate::S { .. } => "S",
                StimGate::CX { .. } => "CX",
                StimGate::CZ { .. } => "CZ",
                StimGate::M { .. } => "M",
                StimGate::R { .. } => "R",
                StimGate::Depolarize1 { .. } => "DEPOLARIZE1",
                StimGate::Depolarize2 { .. } => "DEPOLARIZE2",
                _ => "OTHER",
            };
            *counts.entry(name.to_string()).or_insert(0) += 1;
        }

        counts
    }
}

impl fmt::Display for StimCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_stim())
    }
}

/// Parse a single line into gate(s)
fn parse_line(line: &str) -> Result<Vec<StimGate>, String> {
    // Split into parts
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.is_empty() {
        return Ok(Vec::new());
    }

    let gate_name = parts[0].to_uppercase();

    // Handle gates with parameters
    let (gate_type, params, targets_start) = if gate_name.contains('(') {
        // Parse gate with parameters: GATE(param) targets
        let paren_pos = gate_name.find('(').unwrap();
        let close_paren = gate_name.find(')').ok_or("Missing closing parenthesis")?;

        let name = &gate_name[..paren_pos];
        let param_str = &gate_name[paren_pos + 1..close_paren];
        let params: Vec<f64> = param_str.split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Invalid parameter: {}", e))?;

        (name.to_string(), params, 1)
    } else {
        (gate_name, Vec::new(), 1)
    };

    // Parse targets
    let targets: Vec<usize> = parts[targets_start..]
        .iter()
        .map(|s| s.parse::<usize>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Invalid target: {}", e))?;

    // Create gate(s)
    match gate_type.as_str() {
        "H" => Ok(vec![StimGate::H { targets }]),
        "S" => Ok(vec![StimGate::S { targets }]),
        "S_DAG" | "SDG" => Ok(vec![StimGate::Sdg { targets }]),
        "T" => Ok(vec![StimGate::T { targets }]),
        "T_DAG" | "TDG" => Ok(vec![StimGate::Tdg { targets }]),
        "X" => Ok(vec![StimGate::X { targets }]),
        "Y" => Ok(vec![StimGate::Y { targets }]),
        "Z" => Ok(vec![StimGate::Z { targets }]),
        "CX" | "CNOT" => {
            if targets.len() % 2 != 0 {
                return Err("CX requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::CX { pairs }])
        }
        "CY" => {
            if targets.len() % 2 != 0 {
                return Err("CY requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::CY { pairs }])
        }
        "CZ" => {
            if targets.len() % 2 != 0 {
                return Err("CZ requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::CZ { pairs }])
        }
        "SWAP" => {
            if targets.len() % 2 != 0 {
                return Err("SWAP requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::SWAP { pairs }])
        }
        "ISWAP" => {
            if targets.len() % 2 != 0 {
                return Err("ISWAP requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::ISWAP { pairs }])
        }
        "M" => Ok(vec![StimGate::M { targets }]),
        "MX" => Ok(vec![StimGate::MX { targets }]),
        "MY" => Ok(vec![StimGate::MY { targets }]),
        "R" | "RESET" => Ok(vec![StimGate::R { targets }]),
        "RX" => Ok(vec![StimGate::RX { targets }]),
        "RY" => Ok(vec![StimGate::RY { targets }]),
        "TICK" => Ok(vec![StimGate::Tick]),
        "DETECTOR" => Ok(vec![StimGate::Detector { targets }]),
        "PAULI_CHANNEL_1" => {
            if params.len() != 3 || targets.len() != 1 {
                return Err("PAULI_CHANNEL_1 requires px,py,pz and single target".to_string());
            }
            Ok(vec![StimGate::PauliChannel1 {
                target: targets[0],
                px: params[0],
                py: params[1],
                pz: params[2],
            }])
        }
        "DEPOLARIZE1" => {
            if params.len() != 1 {
                return Err("DEPOLARIZE1 requires probability parameter".to_string());
            }
            Ok(vec![StimGate::Depolarize1 { targets, p: params[0] }])
        }
        "DEPOLARIZE2" => {
            if params.len() != 1 {
                return Err("DEPOLARIZE2 requires probability parameter".to_string());
            }
            if targets.len() % 2 != 0 {
                return Err("DEPOLARIZE2 requires pairs of targets".to_string());
            }
            let pairs: Vec<(usize, usize)> = targets.chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            Ok(vec![StimGate::Depolarize2 { pairs, p: params[0] }])
        }
        "X_ERROR" => {
            if params.len() != 1 {
                return Err("X_ERROR requires probability parameter".to_string());
            }
            Ok(vec![StimGate::XError { targets, p: params[0] }])
        }
        "Z_ERROR" => {
            if params.len() != 1 {
                return Err("Z_ERROR requires probability parameter".to_string());
            }
            Ok(vec![StimGate::ZError { targets, p: params[0] }])
        }
        "OBSERVABLE_INCLUDE" => {
            if params.len() != 1 {
                return Err("OBSERVABLE_INCLUDE requires observable index".to_string());
            }
            Ok(vec![StimGate::ObservableInclude {
                observable: params[0] as usize,
                targets,
            }])
        }
        _ => Err(format!("Unknown gate: {}", gate_type)),
    }
}

/// Convert Stim circuit to ReferenceFrame gates
pub fn to_reference_frame_gates(circuit: &StimCircuit) -> Vec<crate::reference_frame::FrameGate> {
    use crate::reference_frame::FrameGate;

    let mut gates = Vec::new();
    let mut measurement_idx = 0;

    for gate in &circuit.gates {
        match gate {
            StimGate::H { targets } => {
                for &q in targets {
                    gates.push(FrameGate::H { qubit: q });
                }
            }
            StimGate::S { targets } => {
                for &q in targets {
                    gates.push(FrameGate::S { qubit: q });
                }
            }
            StimGate::X { targets } => {
                for &q in targets {
                    gates.push(FrameGate::X { qubit: q });
                }
            }
            StimGate::Y { targets } => {
                for &q in targets {
                    gates.push(FrameGate::Y { qubit: q });
                }
            }
            StimGate::Z { targets } => {
                for &q in targets {
                    gates.push(FrameGate::Z { qubit: q });
                }
            }
            StimGate::CX { pairs } => {
                for &(c, t) in pairs {
                    gates.push(FrameGate::CX { control: c, target: t });
                }
            }
            StimGate::CZ { pairs } => {
                for &(a, b) in pairs {
                    gates.push(FrameGate::CZ { a, b });
                }
            }
            StimGate::M { targets } => {
                for &q in targets {
                    gates.push(FrameGate::M { qubit: q, result_idx: measurement_idx });
                    measurement_idx += 1;
                }
            }
            StimGate::R { targets } => {
                for &q in targets {
                    gates.push(FrameGate::R { qubit: q });
                }
            }
            _ => {
                // Skip non-Clifford and noise gates for now
            }
        }
    }

    gates
}

/// Create example circuits
pub fn bell_state_circuit() -> StimCircuit {
    let mut circuit = StimCircuit::new();
    circuit.add_gate(StimGate::H { targets: vec![0] });
    circuit.add_gate(StimGate::CX { pairs: vec![(0, 1)] });
    circuit.add_gate(StimGate::M { targets: vec![0, 1] });
    circuit
}

/// Create repetition code circuit
pub fn repetition_code_circuit(n: usize, rounds: usize, p: f64) -> StimCircuit {
    let mut circuit = StimCircuit::with_qubits(n);

    // Reset all qubits
    circuit.add_gate(StimGate::R { targets: (0..n).collect() });

    for _ in 0..rounds {
        // Apply CNOTs between neighbors
        for i in 0..(n - 1) {
            circuit.add_gate(StimGate::CX { pairs: vec![(i, i + 1)] });
        }

        // Add noise
        if p > 0.0 {
            circuit.add_gate(StimGate::Depolarize1 {
                targets: (0..n).collect(),
                p,
            });
        }

        circuit.add_gate(StimGate::Tick);
    }

    // Final measurement
    circuit.add_gate(StimGate::M { targets: (0..n).collect() });

    circuit
}

/// Create surface code cycle circuit
pub fn surface_code_cycle(distance: usize, p: f64) -> StimCircuit {
    let n_data = distance * distance;
    let n_ancilla = (distance - 1) * (distance - 1) * 2;
    let n_total = n_data + n_ancilla;

    let mut circuit = StimCircuit::with_qubits(n_total);

    // Reset all qubits
    circuit.add_gate(StimGate::R { targets: (0..n_total).collect() });

    // Prepare ancillas in |+⟩
    circuit.add_gate(StimGate::H { targets: (n_data..n_total).collect() });

    // Apply CNOTs for X stabilizers
    // (Simplified - real surface code has specific connectivity)
    for i in 0..n_ancilla {
        let anc = n_data + i;
        let data1 = i % n_data;
        let data2 = (i + 1) % n_data;
        let data3 = (i + distance) % n_data;
        let data4 = (i + distance + 1) % n_data;

        circuit.add_gate(StimGate::CX { pairs: vec![(anc, data1), (anc, data2), (anc, data3), (anc, data4)] });
    }

    // Add noise
    if p > 0.0 {
        circuit.add_gate(StimGate::Depolarize1 {
            targets: (0..n_total).collect(),
            p,
        });
    }

    // Measure ancillas in X basis
    circuit.add_gate(StimGate::H { targets: (n_data..n_total).collect() });
    circuit.add_gate(StimGate::M { targets: (n_data..n_total).collect() });

    // Measure data in Z basis
    circuit.add_gate(StimGate::M { targets: (0..n_data).collect() });

    circuit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_display() {
        let gate = StimGate::H { targets: vec![0, 1, 2] };
        assert_eq!(format!("{}", gate), "H 0 1 2");

        let gate = StimGate::CX { pairs: vec![(0, 1), (2, 3)] };
        assert_eq!(format!("{}", gate), "CX 0 1 2 3");
    }

    #[test]
    fn test_parse_simple() {
        let circuit = StimCircuit::parse("H 0\nCX 0 1\nM 0 1").unwrap();
        assert_eq!(circuit.gates.len(), 3);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.num_measurements, 2);
    }

    #[test]
    fn test_parse_comments() {
        let circuit = StimCircuit::parse("# This is a comment\nH 0  # inline comment").unwrap();
        assert_eq!(circuit.gates.len(), 1);
    }

    #[test]
    fn test_parse_noise() {
        let circuit = StimCircuit::parse("DEPOLARIZE1(0.001) 0 1 2").unwrap();
        assert_eq!(circuit.gates.len(), 1);

        match &circuit.gates[0] {
            StimGate::Depolarize1 { targets, p } => {
                assert_eq!(targets, &[0, 1, 2]);
                assert!((p - 0.001).abs() < 1e-10);
            }
            _ => panic!("Expected DEPOLARIZE1"),
        }
    }

    #[test]
    fn test_roundtrip() {
        let original = "H 0\nCX 0 1\nM 0 1";
        let circuit = StimCircuit::parse(original).unwrap();
        let exported = circuit.to_stim();

        // Re-parse and compare gate count
        let reparsed = StimCircuit::parse(&exported).unwrap();
        assert_eq!(circuit.gates.len(), reparsed.gates.len());
    }

    #[test]
    fn test_bell_state_circuit() {
        let circuit = bell_state_circuit();
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 3);
    }

    #[test]
    fn test_gate_counts() {
        let mut circuit = StimCircuit::new();
        circuit.add_gate(StimGate::H { targets: vec![0, 1] });
        circuit.add_gate(StimGate::CX { pairs: vec![(0, 1)] });
        circuit.add_gate(StimGate::M { targets: vec![0, 1] });

        let counts = circuit.gate_counts();
        assert_eq!(*counts.get("H").unwrap_or(&0), 1);
        assert_eq!(*counts.get("CX").unwrap_or(&0), 1);
        assert_eq!(*counts.get("M").unwrap_or(&0), 1);
    }

    #[test]
    fn test_repetition_code() {
        let circuit = repetition_code_circuit(5, 3, 0.001);
        assert!(circuit.num_qubits >= 5);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_to_reference_frame() {
        let circuit = bell_state_circuit();
        let gates = to_reference_frame_gates(&circuit);
        // H(0) = 1 gate, CX(0,1) = 1 gate, M(0,1) = 2 gates = 4 total
        assert_eq!(gates.len(), 4);
    }
}
