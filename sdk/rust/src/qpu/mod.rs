//! Real QPU (Quantum Processing Unit) connectivity.
//!
//! This module provides a unified interface for submitting circuits to
//! actual quantum hardware from multiple providers:
//!
//! - **IBM Quantum** — Superconducting processors (Eagle, Heron, etc.)
//! - **Amazon Braket** — Multi-provider access (IonQ, Rigetti, OQC, QuEra)
//! - **Azure Quantum** — Multi-provider access (IonQ, Quantinuum) via QIR
//! - **IonQ Direct** — Trapped-ion processors (Aria, Forte)
//! - **Google Quantum AI** — Sycamore processors
//!
//! All providers implement the [`QPUProvider`] trait for a unified API.
//!
//! # Feature Gates
//!
//! Enable with `--features qpu` (all providers) or individual features:
//! - `qpu-ibm` — IBM Quantum only
//! - `qpu-braket` — Amazon Braket only
//! - `qpu-azure` — Azure Quantum only
//! - `qpu-ionq` — IonQ Direct only
//! - `qpu-google` — Google Quantum AI only
//!
//! # Example
//!
//! ```no_run
//! use nqpu_metal::qpu::*;
//! use std::time::Duration;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Connect to IBM Quantum
//! let provider = IBMProvider::from_env()?;
//!
//! // List available backends
//! let backends = provider.list_backends().await?;
//! for b in &backends {
//!     println!("{}: {} qubits", b.name, b.num_qubits);
//! }
//!
//! // Create a Bell state circuit
//! let mut circuit = QPUCircuit::new(2, 2);
//! circuit.h(0);
//! circuit.cx(0, 1);
//! circuit.measure(0, 0);
//! circuit.measure(1, 1);
//!
//! // Submit to hardware
//! let config = JobConfig { shots: 1024, ..Default::default() };
//! let job = provider.submit_circuit(&circuit, "ibm_brisbane", &config).await?;
//!
//! // Wait for results
//! let result = job.wait_for_completion(
//!     Duration::from_secs(300),
//!     Duration::from_secs(5),
//! ).await?;
//!
//! println!("Results: {:?}", result.counts);
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod job;
pub mod auth;
pub mod provider;
pub mod validation;
pub mod mock;

#[cfg(feature = "qpu-ibm")]
pub mod ibm;

#[cfg(feature = "qpu-braket")]
pub mod braket;

#[cfg(feature = "qpu-azure")]
pub mod azure;

#[cfg(feature = "qpu-ionq")]
pub mod ionq;

#[cfg(feature = "qpu-google")]
pub mod google;

// Re-exports
pub use error::{QPUError, ValidationError};
pub use job::{BackendInfo, BackendStatus, CostEstimate, JobConfig, JobResult, JobStatus, ValidationReport};
pub use auth::AuthConfig;
pub use provider::{QPUProvider, QPUJob};
pub use mock::MockProvider;

#[cfg(feature = "qpu-ibm")]
pub use ibm::IBMProvider;

#[cfg(feature = "qpu-braket")]
pub use braket::BraketProvider;

#[cfg(feature = "qpu-azure")]
pub use azure::AzureProvider;

#[cfg(feature = "qpu-ionq")]
pub use ionq::IonQProvider;

#[cfg(feature = "qpu-google")]
pub use google::GoogleProvider;

use std::collections::HashMap;

// ============================================================
// QPUCircuit — Universal circuit representation for QPU submission
// ============================================================

/// A quantum circuit ready for submission to hardware.
///
/// This is a simplified, provider-agnostic circuit representation.
/// It can be converted to OpenQASM 2.0, OpenQASM 3.0, QIR, or
/// native gate formats as needed by each provider.
#[derive(Debug, Clone)]
pub struct QPUCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_clbits: usize,
    /// Gates in application order
    pub gates: Vec<QPUGate>,
    /// Measurements: (qubit_index, classical_bit_index)
    pub measurements: Vec<(usize, usize)>,
    /// Circuit metadata
    pub metadata: HashMap<String, String>,
}

/// A gate in a QPU circuit.
#[derive(Debug, Clone)]
pub enum QPUGate {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    SX(usize),
    Rz(usize, f64),
    Rx(usize, f64),
    Ry(usize, f64),
    U3(usize, f64, f64, f64),
    // Two-qubit gates
    CX(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    ECR(usize, usize),
    Rzz(usize, usize, f64),
    Rxx(usize, usize, f64),
    // IonQ native gates
    GPI(usize, f64),
    GPI2(usize, f64),
    MS(usize, usize, f64, f64),
    // Google native gates
    SycamoreGate(usize, usize),
    PhasedXZ(usize, f64, f64, f64),
    // Barrier (no-op but preserved for structure)
    Barrier(Vec<usize>),
}

impl QPUGate {
    /// Get the gate name as a string.
    pub fn name(&self) -> &str {
        match self {
            QPUGate::H(_) => "h",
            QPUGate::X(_) => "x",
            QPUGate::Y(_) => "y",
            QPUGate::Z(_) => "z",
            QPUGate::S(_) => "s",
            QPUGate::Sdg(_) => "sdg",
            QPUGate::T(_) => "t",
            QPUGate::Tdg(_) => "tdg",
            QPUGate::SX(_) => "sx",
            QPUGate::Rz(_, _) => "rz",
            QPUGate::Rx(_, _) => "rx",
            QPUGate::Ry(_, _) => "ry",
            QPUGate::U3(_, _, _, _) => "u3",
            QPUGate::CX(_, _) => "cx",
            QPUGate::CZ(_, _) => "cz",
            QPUGate::SWAP(_, _) => "swap",
            QPUGate::ECR(_, _) => "ecr",
            QPUGate::Rzz(_, _, _) => "rzz",
            QPUGate::Rxx(_, _, _) => "rxx",
            QPUGate::GPI(_, _) => "gpi",
            QPUGate::GPI2(_, _) => "gpi2",
            QPUGate::MS(_, _, _, _) => "ms",
            QPUGate::SycamoreGate(_, _) => "syc",
            QPUGate::PhasedXZ(_, _, _, _) => "phased_xz",
            QPUGate::Barrier(_) => "barrier",
        }
    }

    /// Return the two-qubit operands if this is a two-qubit gate.
    pub fn two_qubit_operands(&self) -> Option<(usize, usize)> {
        match self {
            QPUGate::CX(a, b) | QPUGate::CZ(a, b) | QPUGate::SWAP(a, b) | QPUGate::ECR(a, b) => {
                Some((*a, *b))
            }
            QPUGate::Rzz(a, b, _) | QPUGate::Rxx(a, b, _) => Some((*a, *b)),
            QPUGate::MS(a, b, _, _) | QPUGate::SycamoreGate(a, b) => Some((*a, *b)),
            _ => None,
        }
    }

    /// Return all qubit operands.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            QPUGate::H(q) | QPUGate::X(q) | QPUGate::Y(q) | QPUGate::Z(q) => vec![*q],
            QPUGate::S(q) | QPUGate::Sdg(q) | QPUGate::T(q) | QPUGate::Tdg(q) => vec![*q],
            QPUGate::SX(q) => vec![*q],
            QPUGate::Rz(q, _) | QPUGate::Rx(q, _) | QPUGate::Ry(q, _) => vec![*q],
            QPUGate::U3(q, _, _, _) => vec![*q],
            QPUGate::GPI(q, _) | QPUGate::GPI2(q, _) => vec![*q],
            QPUGate::PhasedXZ(q, _, _, _) => vec![*q],
            QPUGate::CX(a, b) | QPUGate::CZ(a, b) | QPUGate::SWAP(a, b) | QPUGate::ECR(a, b) => {
                vec![*a, *b]
            }
            QPUGate::Rzz(a, b, _) | QPUGate::Rxx(a, b, _) => vec![*a, *b],
            QPUGate::MS(a, b, _, _) | QPUGate::SycamoreGate(a, b) => vec![*a, *b],
            QPUGate::Barrier(qs) => qs.clone(),
        }
    }
}

impl QPUCircuit {
    /// Create a new empty circuit.
    pub fn new(num_qubits: usize, num_clbits: usize) -> Self {
        Self {
            num_qubits,
            num_clbits,
            gates: Vec::new(),
            measurements: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a Hadamard gate.
    pub fn h(&mut self, qubit: usize) {
        self.gates.push(QPUGate::H(qubit));
    }

    /// Add a CNOT (CX) gate.
    pub fn cx(&mut self, control: usize, target: usize) {
        self.gates.push(QPUGate::CX(control, target));
    }

    /// Add a CZ gate.
    pub fn cz(&mut self, q0: usize, q1: usize) {
        self.gates.push(QPUGate::CZ(q0, q1));
    }

    /// Add an X gate.
    pub fn x(&mut self, qubit: usize) {
        self.gates.push(QPUGate::X(qubit));
    }

    /// Add a Y gate.
    pub fn y(&mut self, qubit: usize) {
        self.gates.push(QPUGate::Y(qubit));
    }

    /// Add a Z gate.
    pub fn z(&mut self, qubit: usize) {
        self.gates.push(QPUGate::Z(qubit));
    }

    /// Add an Rz gate.
    pub fn rz(&mut self, qubit: usize, theta: f64) {
        self.gates.push(QPUGate::Rz(qubit, theta));
    }

    /// Add an Rx gate.
    pub fn rx(&mut self, qubit: usize, theta: f64) {
        self.gates.push(QPUGate::Rx(qubit, theta));
    }

    /// Add an Ry gate.
    pub fn ry(&mut self, qubit: usize, theta: f64) {
        self.gates.push(QPUGate::Ry(qubit, theta));
    }

    /// Add an SX gate.
    pub fn sx(&mut self, qubit: usize) {
        self.gates.push(QPUGate::SX(qubit));
    }

    /// Add a U3 gate.
    pub fn u3(&mut self, qubit: usize, theta: f64, phi: f64, lambda: f64) {
        self.gates.push(QPUGate::U3(qubit, theta, phi, lambda));
    }

    /// Add a measurement.
    pub fn measure(&mut self, qubit: usize, clbit: usize) {
        self.measurements.push((qubit, clbit));
    }

    /// Add measurements on all qubits.
    pub fn measure_all(&mut self) {
        for i in 0..self.num_qubits.min(self.num_clbits) {
            self.measurements.push((i, i));
        }
    }

    /// Get circuit depth (longest path through the circuit).
    pub fn depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }
        let mut qubit_depth = vec![0usize; self.num_qubits];
        for gate in &self.gates {
            let qs = gate.qubits();
            let max_depth = qs.iter().filter_map(|&q| qubit_depth.get(q)).max().copied().unwrap_or(0);
            for &q in &qs {
                if q < qubit_depth.len() {
                    qubit_depth[q] = max_depth + 1;
                }
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }

    /// Count total gates (excluding barriers).
    pub fn gate_count(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| !matches!(g, QPUGate::Barrier(_)))
            .count()
    }

    /// Convert to OpenQASM 2.0 string.
    pub fn to_qasm2(&self) -> String {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 2.0;\n");
        qasm.push_str("include \"qelib1.inc\";\n");
        qasm.push_str(&format!("qreg q[{}];\n", self.num_qubits));
        qasm.push_str(&format!("creg c[{}];\n", self.num_clbits));

        for gate in &self.gates {
            match gate {
                QPUGate::H(q) => qasm.push_str(&format!("h q[{}];\n", q)),
                QPUGate::X(q) => qasm.push_str(&format!("x q[{}];\n", q)),
                QPUGate::Y(q) => qasm.push_str(&format!("y q[{}];\n", q)),
                QPUGate::Z(q) => qasm.push_str(&format!("z q[{}];\n", q)),
                QPUGate::S(q) => qasm.push_str(&format!("s q[{}];\n", q)),
                QPUGate::Sdg(q) => qasm.push_str(&format!("sdg q[{}];\n", q)),
                QPUGate::T(q) => qasm.push_str(&format!("t q[{}];\n", q)),
                QPUGate::Tdg(q) => qasm.push_str(&format!("tdg q[{}];\n", q)),
                QPUGate::SX(q) => qasm.push_str(&format!("sx q[{}];\n", q)),
                QPUGate::Rz(q, theta) => qasm.push_str(&format!("rz({}) q[{}];\n", theta, q)),
                QPUGate::Rx(q, theta) => qasm.push_str(&format!("rx({}) q[{}];\n", theta, q)),
                QPUGate::Ry(q, theta) => qasm.push_str(&format!("ry({}) q[{}];\n", theta, q)),
                QPUGate::U3(q, theta, phi, lam) => {
                    qasm.push_str(&format!("u3({},{},{}) q[{}];\n", theta, phi, lam, q))
                }
                QPUGate::CX(c, t) => qasm.push_str(&format!("cx q[{}],q[{}];\n", c, t)),
                QPUGate::CZ(a, b) => qasm.push_str(&format!("cz q[{}],q[{}];\n", a, b)),
                QPUGate::SWAP(a, b) => qasm.push_str(&format!("swap q[{}],q[{}];\n", a, b)),
                QPUGate::ECR(a, b) => qasm.push_str(&format!("ecr q[{}],q[{}];\n", a, b)),
                QPUGate::Rzz(a, b, theta) => {
                    qasm.push_str(&format!("rzz({}) q[{}],q[{}];\n", theta, a, b))
                }
                QPUGate::Rxx(a, b, theta) => {
                    qasm.push_str(&format!("rxx({}) q[{}],q[{}];\n", theta, a, b))
                }
                QPUGate::Barrier(qs) => {
                    let args: Vec<String> = qs.iter().map(|q| format!("q[{}]", q)).collect();
                    qasm.push_str(&format!("barrier {};\n", args.join(",")));
                }
                // Native gates — decompose to standard gates for QASM 2.0
                QPUGate::GPI(q, phi) => {
                    // GPI = Rz(-phi) * X * Rz(phi) approximately
                    qasm.push_str(&format!("u3(3.14159265358979,{},{}) q[{}];\n", phi, -phi, q));
                }
                QPUGate::GPI2(q, phi) => {
                    qasm.push_str(&format!(
                        "u3(1.5707963267949,{},{}) q[{}];\n",
                        phi - std::f64::consts::FRAC_PI_2,
                        std::f64::consts::FRAC_PI_2 - phi,
                        q
                    ));
                }
                QPUGate::MS(a, b, phi0, phi1) => {
                    // MS gate approximation via XX rotation
                    qasm.push_str(&format!("rxx(1.5707963267949) q[{}],q[{}];\n", a, b));
                }
                QPUGate::SycamoreGate(a, b) => {
                    // Sycamore ≈ iSWAP-like + CZ phase
                    qasm.push_str(&format!("cx q[{}],q[{}];\n", a, b));
                    qasm.push_str(&format!("cx q[{}],q[{}];\n", b, a));
                    qasm.push_str(&format!("cz q[{}],q[{}];\n", a, b));
                }
                QPUGate::PhasedXZ(q, x_exp, z_exp, axis_phase) => {
                    qasm.push_str(&format!(
                        "u3({},{},{}) q[{}];\n",
                        x_exp * std::f64::consts::PI,
                        axis_phase * std::f64::consts::PI,
                        z_exp * std::f64::consts::PI,
                        q
                    ));
                }
            }
        }

        for &(qubit, clbit) in &self.measurements {
            qasm.push_str(&format!("measure q[{}] -> c[{}];\n", qubit, clbit));
        }

        qasm
    }

    /// Convert to OpenQASM 3.0 string (for Amazon Braket).
    pub fn to_qasm3(&self) -> String {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 3.0;\n");
        qasm.push_str("include \"stdgates.inc\";\n");
        qasm.push_str(&format!("qubit[{}] q;\n", self.num_qubits));
        qasm.push_str(&format!("bit[{}] c;\n", self.num_clbits));

        for gate in &self.gates {
            match gate {
                QPUGate::H(q) => qasm.push_str(&format!("h q[{}];\n", q)),
                QPUGate::X(q) => qasm.push_str(&format!("x q[{}];\n", q)),
                QPUGate::Y(q) => qasm.push_str(&format!("y q[{}];\n", q)),
                QPUGate::Z(q) => qasm.push_str(&format!("z q[{}];\n", q)),
                QPUGate::S(q) => qasm.push_str(&format!("s q[{}];\n", q)),
                QPUGate::Sdg(q) => qasm.push_str(&format!("sdg q[{}];\n", q)),
                QPUGate::T(q) => qasm.push_str(&format!("t q[{}];\n", q)),
                QPUGate::Tdg(q) => qasm.push_str(&format!("tdg q[{}];\n", q)),
                QPUGate::SX(q) => qasm.push_str(&format!("sx q[{}];\n", q)),
                QPUGate::Rz(q, theta) => qasm.push_str(&format!("rz({}) q[{}];\n", theta, q)),
                QPUGate::Rx(q, theta) => qasm.push_str(&format!("rx({}) q[{}];\n", theta, q)),
                QPUGate::Ry(q, theta) => qasm.push_str(&format!("ry({}) q[{}];\n", theta, q)),
                QPUGate::U3(q, theta, phi, lam) => {
                    // QASM 3 doesn't have u3, decompose to rz/ry/rz
                    qasm.push_str(&format!("rz({}) q[{}];\n", lam, q));
                    qasm.push_str(&format!("ry({}) q[{}];\n", theta, q));
                    qasm.push_str(&format!("rz({}) q[{}];\n", phi, q));
                }
                QPUGate::CX(c, t) => qasm.push_str(&format!("cnot q[{}], q[{}];\n", c, t)),
                QPUGate::CZ(a, b) => qasm.push_str(&format!("cz q[{}], q[{}];\n", a, b)),
                QPUGate::SWAP(a, b) => qasm.push_str(&format!("swap q[{}], q[{}];\n", a, b)),
                QPUGate::ECR(a, b) => qasm.push_str(&format!("ecr q[{}], q[{}];\n", a, b)),
                QPUGate::Rzz(a, b, theta) => {
                    qasm.push_str(&format!("rzz({}) q[{}], q[{}];\n", theta, a, b))
                }
                QPUGate::Rxx(a, b, theta) => {
                    qasm.push_str(&format!("rxx({}) q[{}], q[{}];\n", theta, a, b))
                }
                QPUGate::Barrier(qs) => {
                    let args: Vec<String> = qs.iter().map(|q| format!("q[{}]", q)).collect();
                    qasm.push_str(&format!("barrier {};\n", args.join(", ")));
                }
                _ => {
                    // For native gates, decompose via QASM 2 logic
                    // This is a simplified fallback
                    match gate {
                        QPUGate::GPI(q, phi) => {
                            qasm.push_str(&format!(
                                "rz({}) q[{}];\nrx({}) q[{}];\nrz({}) q[{}];\n",
                                phi, q, std::f64::consts::PI, q, -phi, q
                            ));
                        }
                        QPUGate::GPI2(q, phi) => {
                            qasm.push_str(&format!(
                                "rz({}) q[{}];\nrx({}) q[{}];\nrz({}) q[{}];\n",
                                phi, q, std::f64::consts::FRAC_PI_2, q, -phi, q
                            ));
                        }
                        QPUGate::MS(a, b, _, _) => {
                            qasm.push_str(&format!(
                                "rxx({}) q[{}], q[{}];\n",
                                std::f64::consts::FRAC_PI_2,
                                a,
                                b
                            ));
                        }
                        QPUGate::SycamoreGate(a, b) => {
                            qasm.push_str(&format!("cnot q[{}], q[{}];\n", a, b));
                            qasm.push_str(&format!("cnot q[{}], q[{}];\n", b, a));
                            qasm.push_str(&format!("cz q[{}], q[{}];\n", a, b));
                        }
                        QPUGate::PhasedXZ(q, x, z, ax) => {
                            qasm.push_str(&format!("rz({}) q[{}];\n", z * std::f64::consts::PI, q));
                            qasm.push_str(&format!("rx({}) q[{}];\n", x * std::f64::consts::PI, q));
                        }
                        _ => {}
                    }
                }
            }
        }

        for &(qubit, clbit) in &self.measurements {
            qasm.push_str(&format!("c[{}] = measure q[{}];\n", clbit, qubit));
        }

        qasm
    }

    /// Convert to IonQ native gate JSON format.
    pub fn to_ionq_json(&self) -> serde_json::Value {
        let gates: Vec<serde_json::Value> = self
            .gates
            .iter()
            .filter_map(|gate| match gate {
                QPUGate::GPI(q, phi) => Some(serde_json::json!({
                    "gate": "gpi",
                    "target": q,
                    "phase": phi,
                })),
                QPUGate::GPI2(q, phi) => Some(serde_json::json!({
                    "gate": "gpi2",
                    "target": q,
                    "phase": phi,
                })),
                QPUGate::MS(a, b, phi0, phi1) => Some(serde_json::json!({
                    "gate": "ms",
                    "targets": [a, b],
                    "phases": [phi0, phi1],
                    "angle": 0.25,
                })),
                // Standard gates — IonQ also accepts these
                QPUGate::H(q) => Some(serde_json::json!({"gate": "h", "target": q})),
                QPUGate::X(q) => Some(serde_json::json!({"gate": "x", "target": q})),
                QPUGate::Y(q) => Some(serde_json::json!({"gate": "y", "target": q})),
                QPUGate::Z(q) => Some(serde_json::json!({"gate": "z", "target": q})),
                QPUGate::S(q) => Some(serde_json::json!({"gate": "s", "target": q})),
                QPUGate::Sdg(q) => Some(serde_json::json!({"gate": "si", "target": q})),
                QPUGate::T(q) => Some(serde_json::json!({"gate": "t", "target": q})),
                QPUGate::Tdg(q) => Some(serde_json::json!({"gate": "ti", "target": q})),
                QPUGate::SX(q) => Some(serde_json::json!({"gate": "v", "target": q})),
                QPUGate::Rx(q, theta) => Some(serde_json::json!({
                    "gate": "rx",
                    "target": q,
                    "rotation": theta / std::f64::consts::PI,
                })),
                QPUGate::Ry(q, theta) => Some(serde_json::json!({
                    "gate": "ry",
                    "target": q,
                    "rotation": theta / std::f64::consts::PI,
                })),
                QPUGate::Rz(q, theta) => Some(serde_json::json!({
                    "gate": "rz",
                    "target": q,
                    "rotation": theta / std::f64::consts::PI,
                })),
                QPUGate::CX(c, t) => Some(serde_json::json!({
                    "gate": "cnot",
                    "control": c,
                    "target": t,
                })),
                QPUGate::SWAP(a, b) => Some(serde_json::json!({
                    "gate": "swap",
                    "targets": [a, b],
                })),
                QPUGate::Barrier(_) => None,
                _ => None,
            })
            .collect();

        serde_json::json!({
            "gateset": "native",
            "qubits": self.num_qubits,
            "circuit": gates,
        })
    }

    /// Create a Bell state circuit (for testing).
    pub fn bell_state() -> Self {
        let mut c = Self::new(2, 2);
        c.h(0);
        c.cx(0, 1);
        c.measure_all();
        c
    }

    /// Create a GHZ state circuit (for testing).
    pub fn ghz_state(n: usize) -> Self {
        let mut c = Self::new(n, n);
        c.h(0);
        for i in 0..n - 1 {
            c.cx(i, i + 1);
        }
        c.measure_all();
        c
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_circuit_qasm2() {
        let circuit = QPUCircuit::bell_state();
        let qasm = circuit.to_qasm2();
        assert!(qasm.contains("OPENQASM 2.0;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0],q[1];"));
        assert!(qasm.contains("measure q[0] -> c[0];"));
        assert!(qasm.contains("measure q[1] -> c[1];"));
    }

    #[test]
    fn test_bell_circuit_qasm3() {
        let circuit = QPUCircuit::bell_state();
        let qasm = circuit.to_qasm3();
        assert!(qasm.contains("OPENQASM 3.0;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cnot q[0], q[1];"));
        assert!(qasm.contains("c[0] = measure q[0];"));
    }

    #[test]
    fn test_ionq_json() {
        let circuit = QPUCircuit::bell_state();
        let json = circuit.to_ionq_json();
        assert_eq!(json["qubits"], 2);
        assert!(json["circuit"].as_array().unwrap().len() >= 2);
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = QPUCircuit::new(3, 3);
        circuit.h(0);
        circuit.h(1);
        circuit.h(2);
        circuit.cx(0, 1);
        circuit.cx(1, 2);
        assert_eq!(circuit.depth(), 3); // H layer + 2 CX layers
    }

    #[test]
    fn test_ghz_state() {
        let circuit = QPUCircuit::ghz_state(5);
        assert_eq!(circuit.num_qubits, 5);
        assert_eq!(circuit.num_clbits, 5);
        assert_eq!(circuit.gates.len(), 5); // 1 H + 4 CX
        assert_eq!(circuit.measurements.len(), 5);
    }

    #[test]
    fn test_gate_count() {
        let mut circuit = QPUCircuit::new(2, 2);
        circuit.h(0);
        circuit.gates.push(QPUGate::Barrier(vec![0, 1]));
        circuit.cx(0, 1);
        assert_eq!(circuit.gate_count(), 2); // barrier excluded
    }
}
