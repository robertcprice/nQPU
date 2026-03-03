//! Circuit validation against backend constraints.

use crate::qpu::error::{QPUError, ValidationError};
use crate::qpu::job::{BackendInfo, ValidationReport};
use crate::qpu::QPUCircuit;

/// Validates a circuit against a specific backend's constraints.
pub struct CircuitValidator<'a> {
    backend: &'a BackendInfo,
}

impl<'a> CircuitValidator<'a> {
    pub fn new(backend: &'a BackendInfo) -> Self {
        Self { backend }
    }

    /// Run all validation checks and produce a report.
    pub fn validate(&self, circuit: &QPUCircuit) -> ValidationReport {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check empty circuit
        if circuit.gates.is_empty() {
            errors.push("Circuit has no gates".into());
        }

        // Check measurements
        if circuit.measurements.is_empty() {
            warnings.push("Circuit has no measurements — results will be empty".into());
        }

        // Check qubit count
        if circuit.num_qubits > self.backend.num_qubits {
            errors.push(format!(
                "Circuit requires {} qubits but {} has only {}",
                circuit.num_qubits, self.backend.name, self.backend.num_qubits
            ));
        }

        // Check gate support
        let unsupported = self.check_gate_support(circuit);
        if !unsupported.is_empty() {
            warnings.push(format!(
                "Circuit uses gates not in basis set (will be transpiled): {}",
                unsupported.join(", ")
            ));
        }

        // Check connectivity
        if let Some(ref coupling_map) = self.backend.coupling_map {
            let violations = self.check_connectivity(circuit, coupling_map);
            if !violations.is_empty() {
                warnings.push(format!(
                    "{} two-qubit gates violate connectivity (will need routing)",
                    violations.len()
                ));
            }
        }

        // Estimate fidelity
        let estimated_fidelity = self.estimate_fidelity(circuit);

        ValidationReport {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            estimated_fidelity: Some(estimated_fidelity),
        }
    }

    fn check_gate_support(&self, circuit: &QPUCircuit) -> Vec<String> {
        let mut unsupported = Vec::new();
        let basis: std::collections::HashSet<&str> = self
            .backend
            .basis_gates
            .iter()
            .map(|s| s.as_str())
            .collect();

        for gate in &circuit.gates {
            let gate_name = gate.name();
            if !basis.contains(gate_name) && !unsupported.contains(&gate_name.to_string()) {
                unsupported.push(gate_name.to_string());
            }
        }
        unsupported
    }

    fn check_connectivity(
        &self,
        circuit: &QPUCircuit,
        coupling_map: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let mut violations = Vec::new();
        let edges: std::collections::HashSet<(usize, usize)> =
            coupling_map.iter().copied().collect();

        for gate in &circuit.gates {
            if let Some((q0, q1)) = gate.two_qubit_operands() {
                if !edges.contains(&(q0, q1)) && !edges.contains(&(q1, q0)) {
                    violations.push((q0, q1));
                }
            }
        }
        violations
    }

    fn estimate_fidelity(&self, circuit: &QPUCircuit) -> f64 {
        let mut fidelity = 1.0;

        // Gate errors
        let num_1q = circuit
            .gates
            .iter()
            .filter(|g| g.two_qubit_operands().is_none())
            .count();
        let num_2q = circuit
            .gates
            .iter()
            .filter(|g| g.two_qubit_operands().is_some())
            .count();

        if let Some(err_1q) = self.backend.avg_gate_error_1q {
            fidelity *= (1.0 - err_1q).powi(num_1q as i32);
        }
        if let Some(err_2q) = self.backend.avg_gate_error_2q {
            fidelity *= (1.0 - err_2q).powi(num_2q as i32);
        }

        // Readout errors
        let num_measurements = circuit.measurements.len();
        if let Some(err_ro) = self.backend.avg_readout_error {
            fidelity *= (1.0 - err_ro).powi(num_measurements as i32);
        }

        fidelity.max(0.0)
    }
}
