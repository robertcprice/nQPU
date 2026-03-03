//! QASM (OpenQASM 2.0 Subset) Import/Export
//!
//! Basic OpenQASM 2.0 subset parser. Supports common gates (H, X, Y, Z, CX, RX,
//! RY, RZ, etc.) and measurements. This is NOT a full OpenQASM 2.0 or 3.0
//! implementation — it covers the most common circuit patterns.
//! For OpenQASM 3.0 syntax support, see [`crate::qasm3`].

use crate::QuantumSimulator;
use std::collections::HashMap;

// ============================================================
// QASM IMPORT
// ============================================================

/// QASM circuit representation
#[derive(Clone, Debug)]
pub struct QASMCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_cbits: usize,
    /// Circuit name
    pub name: String,
    /// Gates in the circuit
    pub gates: Vec<QASMGate>,
}

/// A single gate in QASM format
#[derive(Clone, Debug)]
pub struct QASMGate {
    /// Gate name (e.g., "h", "cx", "rx")
    pub name: String,
    /// Target qubits
    pub qubits: Vec<usize>,
    /// Classical bit targets (for measurement)
    pub cbits: Option<Vec<usize>>,
    /// Parameters (e.g., rotation angles)
    pub params: Option<Vec<f64>>,
}

/// Parse OpenQASM 2.0 format
///
/// Supports a subset of OpenQASM 2.0:
/// - qubit declarations: qreg q[n];
/// - classical bit declarations: creg c[n];
/// - single-qubit gates: h, x, y, z, s, t, rx, ry, rz, u1, u2, u3
/// - two-qubit gates: cx, cz, swap, crx, cry, crz
/// - measurement: measure q -> c
/// - comments: // ...
///
/// # Example
/// ```ignore
/// let qasm = r#"
/// OPENQASM 2.0;
/// include "qelib1.inc";
/// qreg q[2];
/// creg c[2];
/// h q[0];
/// cx q[0],q[1];
/// measure q -> c;
/// "#;
///
/// let circuit = parse_qasm(qasm).unwrap();
/// ```
pub fn parse_qasm(input: &str) -> Result<QASMCircuit, String> {
    let mut num_qubits = 0;
    let mut num_cbits = 0;
    let mut gates = Vec::new();
    let mut qubit_map: HashMap<String, usize> = HashMap::new();
    let mut cbit_map: HashMap<String, usize> = HashMap::new();
    let mut qubit_counter = 0;
    let mut cbit_counter = 0;

    for line in input.lines() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Skip version declaration and includes
        if line.starts_with("OPENQASM") || line.starts_with("include") {
            continue;
        }

        // Qubit declaration
        if line.contains("qreg") {
            if let Some(rest) = line.split("qreg").nth(1) {
                let rest = rest.trim();
                if let Some(name) = rest.split("[").next() {
                    let size_str = rest.split("[").nth(1).unwrap_or("0]");
                    let size_str = size_str.trim_end_matches("];");
                    let size: usize = size_str.parse().unwrap_or(0);
                    let name = name.trim();

                    for i in 0..size {
                        let key = format!("{}[{}]", name, i);
                        qubit_map.insert(key, qubit_counter);
                        qubit_counter += 1;
                    }
                    num_qubits = qubit_counter;
                }
            }
            continue;
        }

        // Classical bit declaration
        if line.contains("creg") {
            if let Some(rest) = line.split("creg").nth(1) {
                let rest = rest.trim();
                if let Some(name) = rest.split("[").next() {
                    let size_str = rest.split("[").nth(1).unwrap_or("0]");
                    let size_str = size_str.trim_end_matches("];");
                    let size: usize = size_str.parse().unwrap_or(0);
                    let name = name.trim();

                    for i in 0..size {
                        cbit_map.insert(format!("{}[{}]", name, i), cbit_counter);
                        cbit_counter += 1;
                    }
                    num_cbits = cbit_counter;
                }
            }
            continue;
        }

        // Parse gate declarations
        if line.contains("measure") {
            // Measurement: measure q -> c or measure q[i] -> c[j]
            // Format: "measure q[i] -> c[j];" or "measure q -> c;"
            let line_without_measure = line.replace("measure", "").trim().to_string();
            let parts: Vec<&str> = line_without_measure.split("->").map(|s| s.trim()).collect();

            if parts.len() >= 2 {
                let qubit_ref = parts[0].trim_end_matches(';');
                let cbit_ref = parts[1].trim_end_matches(';');

                // Check if this is a register-level measurement (measure q -> c)
                // or a single-qubit measurement (measure q[i] -> c[j])
                if qubit_ref.contains('[') && cbit_ref.contains('[') {
                    // Single qubit to single cbit: measure q[i] -> c[j]
                    if let Some(&q) = qubit_map.get(qubit_ref) {
                        gates.push(QASMGate {
                            name: "measure".to_string(),
                            qubits: vec![q],
                            cbits: Some(vec![cbit_map.get(cbit_ref).copied().unwrap_or(0)]),
                            params: None,
                        });
                    }
                } else if !qubit_ref.contains('[') && !cbit_ref.contains('[') {
                    // Register-level: measure q -> c means measure all qubits
                    // Find all qubits with this prefix and create individual measurements
                    let qubit_prefix = qubit_ref;
                    let cbit_prefix = cbit_ref;

                    for (key, &qubit_idx) in qubit_map.iter() {
                        if key.starts_with(&format!("{}[", qubit_prefix)) {
                            // Extract the index from the key
                            if let Some(idx_str) = key.split('[').nth(1) {
                                if let Some(idx) =
                                    idx_str.trim_end_matches(']').parse::<usize>().ok()
                                {
                                    let cbit_key = format!("{}[{}]", cbit_prefix, idx);
                                    let cbit_idx = cbit_map.get(&cbit_key).copied().unwrap_or(0);
                                    gates.push(QASMGate {
                                        name: "measure".to_string(),
                                        qubits: vec![qubit_idx],
                                        cbits: Some(vec![cbit_idx]),
                                        params: None,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Regular gates
            if !line.is_empty() {
                let (gate_name, rest) = line.split_once(' ').unwrap_or((line, ""));

                // Parse targets
                let mut qubits = Vec::new();
                let mut params = None;

                // Check for parameters in parentheses
                if let Some((name_with_params, targets)) = gate_name.split_once('(') {
                    if let Some((param_str, _rest)) = targets.split_once(')') {
                        // Parse parameters
                        params = Some(
                            param_str
                                .split(',')
                                .map(|p: &str| p.trim().parse::<f64>().unwrap_or(0.0))
                                .collect(),
                        );
                    }
                    let name = name_with_params.trim();
                    parse_qasm_targets(
                        name,
                        targets.split(')').next().unwrap_or(""),
                        &mut qubits,
                        &qubit_map,
                    );
                } else {
                    parse_qasm_targets(gate_name, rest, &mut qubits, &qubit_map);
                }

                if !qubits.is_empty() {
                    gates.push(QASMGate {
                        name: gate_name.trim().to_string(),
                        qubits,
                        cbits: None,
                        params,
                    });
                }
            }
        }
    }

    Ok(QASMCircuit {
        num_qubits,
        num_cbits,
        name: "imported".to_string(),
        gates,
    })
}

/// Helper function to parse QASM target qubits
fn parse_qasm_targets(
    _gate_name: &str,
    targets: &str,
    qubits: &mut Vec<usize>,
    qubit_map: &HashMap<String, usize>,
) {
    // Clean up targets (remove trailing semicolon, whitespace)
    let targets = targets.trim_end_matches(';').trim();

    // Split by comma for multiple targets
    for target in targets.split(',') {
        let target = target.trim();

        // Handle array indexing like q[0]
        if let Some(name) = target.split('[').next() {
            let index_str = target.split('[').nth(1).unwrap_or("0");
            let index: usize = index_str.trim_end_matches(']').parse().unwrap_or(0);
            let key = format!("{}[{}]", name.trim(), index);
            if let Some(&q) = qubit_map.get(&key) {
                qubits.push(q);
            }
        } else if let Some(&q) = qubit_map.get(target) {
            qubits.push(q);
        }
    }
}

/// Execute a QASM circuit on a simulator
///
/// # Arguments
/// * `circuit` - Parsed QASM circuit
/// * `simulator` - Simulator to execute on
///
/// # Returns
/// Measurement result (as integer)
pub fn execute_qasm(circuit: &QASMCircuit, simulator: &mut QuantumSimulator) -> usize {
    let mut measurement_result = 0;

    for gate in &circuit.gates {
        match gate.name.as_str() {
            "measure" => {
                let qubit = gate.qubits[0];
                let (bit, _) = simulator.measure_qubit(qubit);
                if let Some(cbits) = &gate.cbits {
                    measurement_result |= bit << cbits[0];
                }
            }
            "h" => {
                for &q in &gate.qubits {
                    simulator.h(q);
                }
            }
            "x" => {
                for &q in &gate.qubits {
                    simulator.x(q);
                }
            }
            "y" => {
                for &q in &gate.qubits {
                    simulator.y(q);
                }
            }
            "z" => {
                for &q in &gate.qubits {
                    simulator.z(q);
                }
            }
            "s" => {
                for &q in &gate.qubits {
                    simulator.s(q);
                }
            }
            "t" => {
                for &q in &gate.qubits {
                    simulator.t(q);
                }
            }
            "cx" => {
                if gate.qubits.len() >= 2 {
                    simulator.cnot(gate.qubits[0], gate.qubits[1]);
                }
            }
            "cz" => {
                if gate.qubits.len() >= 2 {
                    simulator.cz(gate.qubits[0], gate.qubits[1]);
                }
            }
            "swap" => {
                if gate.qubits.len() >= 2 {
                    simulator.swap(gate.qubits[0], gate.qubits[1]);
                }
            }
            "rx" => {
                if let Some(params) = &gate.params {
                    if !params.is_empty() && !gate.qubits.is_empty() {
                        simulator.rx(gate.qubits[0], params[0]);
                    }
                }
            }
            "ry" => {
                if let Some(params) = &gate.params {
                    if !params.is_empty() && !gate.qubits.is_empty() {
                        simulator.ry(gate.qubits[0], params[0]);
                    }
                }
            }
            "rz" => {
                if let Some(params) = &gate.params {
                    if !params.is_empty() && !gate.qubits.is_empty() {
                        simulator.rz(gate.qubits[0], params[0]);
                    }
                }
            }
            _ => {
                // Unknown gate - skip
            }
        }
    }

    measurement_result
}

// ============================================================
// QASM EXPORT
// ============================================================

/// Export a circuit function to OpenQASM format
///
/// # Arguments
/// * `name` - Circuit name
/// * `num_qubits` - Number of qubits
/// * `num_cbits` - Number of classical bits
/// * `generate_circuit` - Function that generates the circuit
pub fn export_qasm<F>(
    name: &str,
    num_qubits: usize,
    num_cbits: usize,
    generate_circuit: F,
) -> String
where
    F: Fn(&mut QASMWriter),
{
    let mut writer = QASMWriter::new(num_qubits, num_cbits);
    generate_circuit(&mut writer);
    writer.to_string(name)
}

/// QASM writer for constructing circuits
pub struct QASMWriter {
    num_qubits: usize,
    num_cbits: usize,
    statements: Vec<String>,
}

impl QASMWriter {
    /// Create a new QASM writer
    pub fn new(num_qubits: usize, num_cbits: usize) -> Self {
        QASMWriter {
            num_qubits,
            num_cbits,
            statements: Vec::new(),
        }
    }

    /// Add a Hadamard gate
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("h q[{}];", qubit));
        self
    }

    /// Add an X gate
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("x q[{}];", qubit));
        self
    }

    /// Add a Y gate
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("y q[{}];", qubit));
        self
    }

    /// Add a Z gate
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("z q[{}];", qubit));
        self
    }

    /// Add an S gate
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("s q[{}];", qubit));
        self
    }

    /// Add a T gate
    pub fn t(&mut self, qubit: usize) -> &mut Self {
        self.statements.push(format!("t q[{}];", qubit));
        self
    }

    /// Add an RX gate
    pub fn rx(&mut self, qubit: usize, angle: f64) -> &mut Self {
        self.statements.push(format!("rx({}) q[{}];", angle, qubit));
        self
    }

    /// Add an RY gate
    pub fn ry(&mut self, qubit: usize, angle: f64) -> &mut Self {
        self.statements.push(format!("ry({}) q[{}];", angle, qubit));
        self
    }

    /// Add an RZ gate
    pub fn rz(&mut self, qubit: usize, angle: f64) -> &mut Self {
        self.statements.push(format!("rz({}) q[{}];", angle, qubit));
        self
    }

    /// Add a CNOT gate
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.statements
            .push(format!("cx q[{}],q[{}];", control, target));
        self
    }

    /// Add a CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.statements
            .push(format!("cz q[{}],q[{}];", control, target));
        self
    }

    /// Add a SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        self.statements
            .push(format!("swap q[{}],q[{}];", qubit1, qubit2));
        self
    }

    /// Add a measurement
    pub fn measure(&mut self, qubit: usize, cbit: usize) -> &mut Self {
        self.statements
            .push(format!("measure q[{}] -> c[{}];", qubit, cbit));
        self
    }

    /// Add a custom gate
    pub fn gate(&mut self, name: &str, qubits: &[usize]) -> &mut Self {
        let qubit_str = qubits
            .iter()
            .map(|q| format!("q[{}]", q))
            .collect::<Vec<_>>()
            .join(",");
        self.statements.push(format!("{} {};", name, qubit_str));
        self
    }

    /// Add a comment
    pub fn comment(&mut self, comment: &str) -> &mut Self {
        self.statements.push(format!("// {}", comment));
        self
    }

    /// Convert to QASM string
    pub fn to_string(&self, _name: &str) -> String {
        let mut output = String::new();

        output.push_str("OPENQASM 2.0;\n");
        output.push_str("include \"qelib1.inc\";\n");
        output.push_str(&format!("qreg q[{}];\n", self.num_qubits));
        output.push_str(&format!("creg c[{}];\n", self.num_cbits));

        for statement in &self.statements {
            output.push_str(statement);
            output.push('\n');
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qasm_export() {
        let qasm = export_qasm("test_circuit", 2, 2, |w| {
            w.h(0);
            w.cx(0, 1);
            w.measure(0, 0);
            w.measure(1, 1);
        });

        assert!(qasm.contains("OPENQASM 2.0"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0],q[1];"));
    }

    #[test]
    fn test_qasm_import() {
        let qasm = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"#;

        let circuit = parse_qasm(qasm).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 4); // h, cx, measure q[0], measure q[1]
    }

    #[test]
    fn test_execute_qasm() {
        let qasm = r#"
OPENQASM 2.0;
qreg q[2];
creg c[2];
h q[0];
h q[1];
cx q[0],q[1];
"#;

        let circuit = parse_qasm(qasm).unwrap();
        let mut sim = QuantumSimulator::new(circuit.num_qubits);
        let result = execute_qasm(&circuit, &mut sim);

        // Result should be deterministic for this simple circuit
        // (|00⟩ → H⊗H → CNOT → |00⟩ + |11⟩ / √2)
        assert!(result == 0 || result == 3);
    }

    #[test]
    fn test_qasm_writer_rotation() {
        let qasm = export_qasm("rotation", 1, 1, |w| {
            w.rx(0, std::f64::consts::PI / 4.0);
            w.ry(0, std::f64::consts::PI / 2.0);
        });

        assert!(qasm.contains("rx("));
        assert!(qasm.contains("ry("));
    }
}
