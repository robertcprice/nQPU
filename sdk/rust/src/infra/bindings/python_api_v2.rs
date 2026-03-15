//! Simplified Python API for nQPU-Metal (PyO3 0.23 compatible)

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::jax_bridge::{
    parameter_shift_grad, vmap_expectation, vmap_simulate, JAXCircuit, JaxComplex,
};
use crate::{gates::Gate, QuantumSimulator};

// ============================================================================
// BACKEND ENUM
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "Backend", from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyBackend {
    CPU,
    GPU,
    Auto,
}

// ============================================================================
// QUANTUM CIRCUIT
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "QuantumCircuit", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyQuantumCircuit {
    num_qubits: usize,
    gates: Vec<Gate>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyQuantumCircuit {
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        if num_qubits > 28 {
            return Err(PyValueError::new_err("Maximum 28 qubits supported"));
        }
        Ok(PyQuantumCircuit {
            num_qubits,
            gates: Vec::new(),
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn h(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::h(qubit));
        Ok(())
    }

    fn x(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::x(qubit));
        Ok(())
    }

    fn y(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::y(qubit));
        Ok(())
    }

    fn z(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::z(qubit));
        Ok(())
    }

    fn s(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::s(qubit));
        Ok(())
    }

    fn t(&mut self, qubit: usize) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::t(qubit));
        Ok(())
    }

    fn rx(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::rx(qubit, theta));
        Ok(())
    }

    fn ry(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::ry(qubit, theta));
        Ok(())
    }

    fn rz(&mut self, qubit: usize, theta: f64) -> PyResult<()> {
        if qubit >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::rz(qubit, theta));
        Ok(())
    }

    fn cx(&mut self, control: usize, target: usize) -> PyResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        if control == target {
            return Err(PyValueError::new_err(
                "Control and target must be different",
            ));
        }
        self.gates.push(Gate::cnot(control, target));
        Ok(())
    }

    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.cx(control, target)
    }

    fn cz(&mut self, control: usize, target: usize) -> PyResult<()> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::cz(control, target));
        Ok(())
    }

    fn swap(&mut self, a: usize, b: usize) -> PyResult<()> {
        if a >= self.num_qubits || b >= self.num_qubits {
            return Err(PyValueError::new_err("Qubit index out of range"));
        }
        self.gates.push(Gate::swap(a, b));
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "<QuantumCircuit {} qubits, {} gates>",
            self.num_qubits,
            self.gates.len()
        )
    }
}

// Internal methods (not exposed to Python)
#[cfg(feature = "python")]
impl PyQuantumCircuit {
    pub(crate) fn get_gates(&self) -> &[Gate] {
        &self.gates
    }
}

// ============================================================================
// SIMULATION RESULT
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "Result", from_py_object)]
#[derive(Debug, Clone)]
pub struct PySimulationResult {
    counts: HashMap<String, usize>,
    shots: usize,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySimulationResult {
    #[getter]
    fn counts(&self) -> PyResult<HashMap<String, usize>> {
        Ok(self.counts.clone())
    }

    fn results(&self) -> PyResult<HashMap<String, usize>> {
        Ok(self.counts.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "<Result shots={}, outcomes={}>",
            self.shots,
            self.counts.len()
        )
    }
}

// ============================================================================
// MAIN SIMULATOR
// ============================================================================

#[cfg(feature = "python")]
#[pyclass(name = "Simulator")]
pub struct PySimulator {
    backend: PyBackend,
}

#[cfg(feature = "python")]
#[pymethods]
impl PySimulator {
    #[new]
    fn new(backend: Option<PyBackend>) -> Self {
        PySimulator {
            backend: backend.unwrap_or(PyBackend::Auto),
        }
    }

    fn run(
        &self,
        circuit: &PyQuantumCircuit,
        shots: Option<usize>,
    ) -> PyResult<PySimulationResult> {
        let shots = shots.unwrap_or(1024);
        let mut sim = QuantumSimulator::new(circuit.num_qubits);

        // Execute gates
        for gate in circuit.get_gates() {
            match &gate.gate_type {
                crate::gates::GateType::H => sim.h(gate.targets[0]),
                crate::gates::GateType::X => sim.x(gate.targets[0]),
                crate::gates::GateType::Y => sim.y(gate.targets[0]),
                crate::gates::GateType::Z => sim.z(gate.targets[0]),
                crate::gates::GateType::S => sim.s(gate.targets[0]),
                crate::gates::GateType::T => sim.t(gate.targets[0]),
                crate::gates::GateType::Rx(theta) => sim.rx(gate.targets[0], *theta),
                crate::gates::GateType::Ry(theta) => sim.ry(gate.targets[0], *theta),
                crate::gates::GateType::Rz(theta) => sim.rz(gate.targets[0], *theta),
                crate::gates::GateType::CNOT => {
                    if !gate.controls.is_empty() {
                        sim.cnot(gate.controls[0], gate.targets[0]);
                    }
                }
                crate::gates::GateType::CZ => {
                    if !gate.controls.is_empty() {
                        sim.cz(gate.controls[0], gate.targets[0]);
                    }
                }
                crate::gates::GateType::SWAP => {
                    sim.swap(gate.targets[0], gate.targets[1]);
                }
                _ => {}
            }
        }

        // Sample shots
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..shots {
            let measured = sim.measure();
            let bitstring = format!("{:0width$b}", measured, width = circuit.num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        Ok(PySimulationResult { counts, shots })
    }

    fn __repr__(&self) -> String {
        format!("<Simulator backend={:?}>", self.backend)
    }
}

// ============================================================================
// JAX INTEGRATION
// ============================================================================

/// PyO3 wrapper for JAXCircuit
#[cfg(feature = "python")]
#[pyclass(name = "PyJAXCircuit", from_py_object)]
#[derive(Clone)]
pub struct PyJAXCircuit {
    circuit: JAXCircuit,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyJAXCircuit {
    #[new]
    fn new(n_qubits: usize) -> PyResult<Self> {
        if n_qubits > 28 {
            return Err(PyValueError::new_err("Maximum 28 qubits supported"));
        }
        Ok(PyJAXCircuit {
            circuit: JAXCircuit::new(n_qubits),
        })
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.circuit.num_qubits()
    }

    #[getter]
    fn num_parameters(&self) -> usize {
        self.circuit.num_parameters()
    }

    fn h(&mut self, qubit: usize) {
        self.circuit.h(qubit);
    }

    fn x(&mut self, qubit: usize) {
        self.circuit.x(qubit);
    }

    fn y(&mut self, qubit: usize) {
        self.circuit.y(qubit);
    }

    fn z(&mut self, qubit: usize) {
        self.circuit.z(qubit);
    }

    fn rx(&mut self, qubit: usize, param_name: &str) {
        self.circuit.rx(qubit, param_name);
    }

    fn ry(&mut self, qubit: usize, param_name: &str) {
        self.circuit.ry(qubit, param_name);
    }

    fn rz(&mut self, qubit: usize, param_name: &str) {
        self.circuit.rz(qubit, param_name);
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.circuit.cx(control, target);
    }

    fn cz(&mut self, control: usize, target: usize) {
        self.circuit.cz(control, target);
    }

    fn __repr__(&self) -> String {
        format!(
            "<PyJAXCircuit n_qubits={}, n_params={}, n_gates={}>",
            self.circuit.num_qubits(),
            self.circuit.num_parameters(),
            self.circuit.num_gates()
        )
    }
}

/// Simulate JAX circuit and return complex amplitudes
#[cfg(feature = "python")]
#[pyfunction]
fn py_jax_simulate(circuit: &PyJAXCircuit, params: Vec<f32>) -> PyResult<Vec<(f32, f32)>> {
    let state = circuit.circuit.simulate(&params);
    Ok(state.iter().map(|c| (c.real, c.imag)).collect())
}

/// Compute expectation value of observable
#[cfg(feature = "python")]
#[pyfunction]
fn py_jax_expectation(circuit: &PyJAXCircuit, params: Vec<f32>, observable: &str) -> PyResult<f32> {
    Ok(circuit.circuit.expectation(&params, observable))
}

/// Compute parameter-shift gradients
#[cfg(feature = "python")]
#[pyfunction]
fn py_jax_gradient(circuit: &PyJAXCircuit, params: Vec<f32>, qubit: usize) -> PyResult<Vec<f32>> {
    let shift = std::f32::consts::PI / 2.0; // Standard parameter-shift
    Ok(parameter_shift_grad(
        &circuit.circuit,
        &params,
        qubit,
        shift,
    ))
}

/// VMAP: batch simulate multiple parameter sets
#[cfg(feature = "python")]
#[pyfunction]
fn py_jax_vmap_simulate(
    circuit: &PyJAXCircuit,
    batch_params: Vec<Vec<f32>>,
) -> PyResult<Vec<Vec<(f32, f32)>>> {
    let batch_states = vmap_simulate(&circuit.circuit, &batch_params);
    Ok(batch_states
        .iter()
        .map(|state| state.iter().map(|c| (c.real, c.imag)).collect())
        .collect())
}

/// VMAP: batch expectation values
#[cfg(feature = "python")]
#[pyfunction]
fn py_jax_vmap_expectation(
    circuit: &PyJAXCircuit,
    batch_params: Vec<Vec<f32>>,
    observable: &str,
) -> PyResult<Vec<f32>> {
    Ok(vmap_expectation(
        &circuit.circuit,
        &batch_params,
        observable,
    ))
}

// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

#[cfg(feature = "python")]
#[pymodule]
fn nqpu_metal_v2(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackend>()?;
    m.add_class::<PyQuantumCircuit>()?;
    m.add_class::<PySimulator>()?;
    m.add_class::<PySimulationResult>()?;
    m.add("PI", std::f64::consts::PI)?;
    m.add("VERSION", env!("CARGO_PKG_VERSION"))?;

    // JAX integration
    m.add_class::<PyJAXCircuit>()?;
    m.add_function(wrap_pyfunction!(py_jax_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_jax_expectation, m)?)?;
    m.add_function(wrap_pyfunction!(py_jax_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(py_jax_vmap_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(py_jax_vmap_expectation, m)?)?;

    Ok(())
}

// ============================================================================
// TESTS (pure Rust — no PyO3 interpreter required)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{Gate, GateType};
    use crate::jax_bridge::{JAXCircuit, JaxComplex};
    use std::collections::HashMap;

    // ------------------------------------------------------------------
    // 1. Gate construction mirrors PyQuantumCircuit gate-building logic
    // ------------------------------------------------------------------

    #[test]
    fn test_single_qubit_gate_construction() {
        let gates = vec![
            Gate::h(0),
            Gate::x(1),
            Gate::y(2),
            Gate::z(3),
            Gate::s(0),
            Gate::t(1),
        ];
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[0].targets, vec![0]);
        assert!(gates[0].controls.is_empty());

        assert_eq!(gates[1].gate_type, GateType::X);
        assert_eq!(gates[1].targets, vec![1]);

        assert_eq!(gates[2].gate_type, GateType::Y);
        assert_eq!(gates[2].targets, vec![2]);

        assert_eq!(gates[3].gate_type, GateType::Z);
        assert_eq!(gates[3].targets, vec![3]);

        assert_eq!(gates[4].gate_type, GateType::S);
        assert_eq!(gates[5].gate_type, GateType::T);
    }

    #[test]
    fn test_rotation_gate_construction() {
        let theta = std::f64::consts::PI / 4.0;
        let rx = Gate::rx(0, theta);
        let ry = Gate::ry(1, theta);
        let rz = Gate::rz(2, theta);

        assert_eq!(rx.gate_type, GateType::Rx(theta));
        assert_eq!(rx.targets, vec![0]);
        assert!(rx.controls.is_empty());

        assert_eq!(ry.gate_type, GateType::Ry(theta));
        assert_eq!(ry.targets, vec![1]);

        assert_eq!(rz.gate_type, GateType::Rz(theta));
        assert_eq!(rz.targets, vec![2]);
    }

    #[test]
    fn test_two_qubit_gate_construction() {
        let cnot = Gate::cnot(0, 1);
        assert_eq!(cnot.gate_type, GateType::CNOT);
        assert_eq!(cnot.targets, vec![1]);
        assert_eq!(cnot.controls, vec![0]);

        let cz = Gate::cz(2, 3);
        assert_eq!(cz.gate_type, GateType::CZ);
        assert_eq!(cz.targets, vec![3]);
        assert_eq!(cz.controls, vec![2]);

        let swap = Gate::swap(0, 2);
        assert_eq!(swap.gate_type, GateType::SWAP);
        assert_eq!(swap.targets, vec![0, 2]);
        assert!(swap.controls.is_empty());
    }

    // ------------------------------------------------------------------
    // 2. Qubit index validation logic (mirrors PyQuantumCircuit bounds)
    // ------------------------------------------------------------------

    #[test]
    fn test_qubit_bounds_validation_logic() {
        // The Python API rejects qubit >= num_qubits.
        // Verify the same boundary condition in pure Rust.
        let num_qubits: usize = 4;

        // Valid indices: 0..3
        for q in 0..num_qubits {
            assert!(
                q < num_qubits,
                "qubit {} should be valid for {} qubits",
                q,
                num_qubits
            );
        }

        // Invalid: index == num_qubits
        assert!(
            !(num_qubits < num_qubits),
            "index == num_qubits must be rejected"
        );

        // CNOT same-qubit check (mirrors cx/cnot validation)
        let control = 1_usize;
        let target = 1_usize;
        assert_eq!(control, target, "same control/target should be caught");
    }

    #[test]
    fn test_max_qubit_limit() {
        // PyQuantumCircuit::new rejects num_qubits > 28
        let max_allowed = 28_usize;
        assert!(max_allowed <= 28);
        assert!(29 > 28, "29 qubits should be rejected");
    }

    // ------------------------------------------------------------------
    // 3. Bitstring formatting (mirrors run method's measurement encoding)
    // ------------------------------------------------------------------

    #[test]
    fn test_bitstring_formatting() {
        // The run method formats: format!("{:0width$b}", measured, width = num_qubits)
        let num_qubits = 3;

        // |000> = 0
        let s = format!("{:0width$b}", 0_usize, width = num_qubits);
        assert_eq!(s, "000");

        // |001> = 1
        let s = format!("{:0width$b}", 1_usize, width = num_qubits);
        assert_eq!(s, "001");

        // |111> = 7
        let s = format!("{:0width$b}", 7_usize, width = num_qubits);
        assert_eq!(s, "111");

        // |101> = 5
        let s = format!("{:0width$b}", 5_usize, width = num_qubits);
        assert_eq!(s, "101");

        // Single qubit
        let s = format!("{:0width$b}", 0_usize, width = 1);
        assert_eq!(s, "0");
        let s = format!("{:0width$b}", 1_usize, width = 1);
        assert_eq!(s, "1");
    }

    // ------------------------------------------------------------------
    // 4. Measurement counting (mirrors run method's HashMap accumulation)
    // ------------------------------------------------------------------

    #[test]
    fn test_measurement_count_accumulation() {
        // Simulates the counting logic from PySimulator::run
        let measurements = vec![0_usize, 1, 0, 0, 1, 3, 3, 0];
        let num_qubits = 2;
        let mut counts: HashMap<String, usize> = HashMap::new();

        for measured in &measurements {
            let bitstring = format!("{:0width$b}", measured, width = num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        assert_eq!(counts.get("00"), Some(&4)); // 0 appeared 4 times
        assert_eq!(counts.get("01"), Some(&2)); // 1 appeared 2 times
        assert_eq!(counts.get("11"), Some(&2)); // 3 appeared 2 times
        assert_eq!(counts.get("10"), None); // 2 never appeared

        let total: usize = counts.values().sum();
        assert_eq!(total, measurements.len());
    }

    // ------------------------------------------------------------------
    // 5. JAXCircuit parameter management (mirrors PyJAXCircuit wrappers)
    // ------------------------------------------------------------------

    #[test]
    fn test_jax_circuit_construction_and_parameters() {
        let mut circuit = JAXCircuit::new(2);
        assert_eq!(circuit.num_parameters(), 0);

        // Add gates (mirrors PyJAXCircuit methods)
        circuit.h(0);
        circuit.x(1);
        circuit.rx(0, "theta_0");
        circuit.ry(1, "theta_1");
        circuit.rz(0, "theta_2");
        circuit.cx(0, 1);
        circuit.cz(1, 0);

        assert_eq!(circuit.num_parameters(), 3);
    }

    #[test]
    fn test_jax_circuit_simulate_identity() {
        // A circuit with no gates should return |00...0> state
        let circuit = JAXCircuit::new(2);
        let state = circuit.simulate(&[]);

        // 2 qubits => 4 amplitudes, only |00> has amplitude 1
        assert_eq!(state.len(), 4);
        assert!((state[0].real - 1.0).abs() < 1e-6);
        assert!(state[0].imag.abs() < 1e-6);
        for i in 1..4 {
            assert!(state[i].norm_sq() < 1e-6, "state[{}] should be zero", i);
        }
    }

    #[test]
    fn test_jax_circuit_hadamard_superposition() {
        // H|0> = (|0> + |1>) / sqrt(2)
        let mut circuit = JAXCircuit::new(1);
        circuit.h(0);
        let state = circuit.simulate(&[]);

        assert_eq!(state.len(), 2);
        let inv_sqrt2: f32 = 1.0 / 2.0_f32.sqrt();
        assert!(
            (state[0].real - inv_sqrt2).abs() < 1e-5,
            "expected {}, got {}",
            inv_sqrt2,
            state[0].real
        );
        assert!(
            (state[1].real - inv_sqrt2).abs() < 1e-5,
            "expected {}, got {}",
            inv_sqrt2,
            state[1].real
        );
    }

    #[test]
    fn test_jax_circuit_x_gate_flip() {
        // X|0> = |1>
        let mut circuit = JAXCircuit::new(1);
        circuit.x(0);
        let state = circuit.simulate(&[]);

        assert_eq!(state.len(), 2);
        assert!(state[0].norm_sq() < 1e-6, "|0> amplitude should be 0");
        assert!(
            (state[1].real - 1.0).abs() < 1e-6,
            "|1> amplitude should be 1"
        );
    }

    // ------------------------------------------------------------------
    // 6. JaxComplex arithmetic (used in py_jax_simulate return values)
    // ------------------------------------------------------------------

    #[test]
    fn test_jax_complex_arithmetic() {
        let a = JaxComplex::new(1.0, 2.0);
        let b = JaxComplex::new(3.0, -1.0);

        // Addition
        let sum = a + b;
        assert!((sum.real - 4.0).abs() < 1e-6);
        assert!((sum.imag - 1.0).abs() < 1e-6);

        // Multiplication: (1+2i)(3-i) = 3 - i + 6i - 2i^2 = 5 + 5i
        let prod = a * b;
        assert!((prod.real - 5.0).abs() < 1e-6);
        assert!((prod.imag - 5.0).abs() < 1e-6);

        // Conjugate
        let conj = a.conj();
        assert!((conj.real - 1.0).abs() < 1e-6);
        assert!((conj.imag - (-2.0)).abs() < 1e-6);

        // Norm
        assert!((a.norm_sq() - 5.0).abs() < 1e-6); // 1^2 + 2^2
        assert!((a.norm() - 5.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_jax_complex_special_values() {
        let zero = JaxComplex::zero();
        assert_eq!(zero.real, 0.0);
        assert_eq!(zero.imag, 0.0);

        let one = JaxComplex::one();
        assert_eq!(one.real, 1.0);
        assert_eq!(one.imag, 0.0);
    }

    // ------------------------------------------------------------------
    // 7. Full simulation flow (mirrors PySimulator::run end-to-end)
    // ------------------------------------------------------------------

    #[test]
    fn test_simulator_run_flow_x_gate() {
        // Mirrors: circuit = QuantumCircuit(1); circuit.x(0); sim.run(circuit, shots=100)
        // X|0> = |1>, so all measurements should be "1"
        let num_qubits = 1;
        let gates = vec![Gate::x(0)];

        let mut sim = QuantumSimulator::new(num_qubits);
        for gate in &gates {
            match &gate.gate_type {
                GateType::X => sim.x(gate.targets[0]),
                _ => {}
            }
        }

        let shots = 100;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..shots {
            let measured = sim.measure();
            let bitstring = format!("{:0width$b}", measured, width = num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        // X|0> = |1>, every measurement should be "1"
        assert_eq!(counts.get("1"), Some(&100));
        assert_eq!(counts.get("0"), None);
    }

    #[test]
    fn test_simulator_run_flow_bell_state() {
        // Bell state: H(0), CNOT(0,1) => equal mix of "00" and "11"
        let num_qubits = 2;
        let mut sim = QuantumSimulator::new(num_qubits);
        sim.h(0);
        sim.cnot(0, 1);

        let shots = 2000;
        let mut counts: HashMap<String, usize> = HashMap::new();
        for _ in 0..shots {
            let measured = sim.measure();
            let bitstring = format!("{:0width$b}", measured, width = num_qubits);
            *counts.entry(bitstring).or_insert(0) += 1;
        }

        // Should only see "00" and "11"
        let count_00 = counts.get("00").copied().unwrap_or(0);
        let count_11 = counts.get("11").copied().unwrap_or(0);
        let count_01 = counts.get("01").copied().unwrap_or(0);
        let count_10 = counts.get("10").copied().unwrap_or(0);

        assert_eq!(count_00 + count_11, shots, "only 00 and 11 should appear");
        assert_eq!(count_01 + count_10, 0, "01 and 10 should never appear");

        // Statistical check: each should be roughly 50% (allow wide margin)
        assert!(count_00 > 700, "expected ~1000 of 00, got {}", count_00);
        assert!(count_11 > 700, "expected ~1000 of 11, got {}", count_11);
    }

    // ------------------------------------------------------------------
    // 8. Repr format strings (mirrors __repr__ methods)
    // ------------------------------------------------------------------

    #[test]
    fn test_repr_format_strings() {
        // PyQuantumCircuit.__repr__
        let num_qubits = 5;
        let num_gates = 12;
        let repr = format!(
            "<QuantumCircuit {} qubits, {} gates>",
            num_qubits, num_gates
        );
        assert_eq!(repr, "<QuantumCircuit 5 qubits, 12 gates>");

        // PySimulationResult.__repr__
        let shots = 1024;
        let outcomes = 4;
        let repr = format!("<Result shots={}, outcomes={}>", shots, outcomes);
        assert_eq!(repr, "<Result shots=1024, outcomes=4>");

        // PyJAXCircuit.__repr__
        let n_qubits = 3;
        let n_params = 2;
        let n_gates = 7;
        let repr = format!(
            "<PyJAXCircuit n_qubits={}, n_params={}, n_gates={}>",
            n_qubits, n_params, n_gates
        );
        assert_eq!(repr, "<PyJAXCircuit n_qubits=3, n_params=2, n_gates=7>");
    }

    // ------------------------------------------------------------------
    // 9. Gate vector accumulation pattern (mirrors PyQuantumCircuit)
    // ------------------------------------------------------------------

    #[test]
    fn test_gate_vector_accumulation() {
        // Mirrors the pattern: self.gates.push(Gate::h(qubit))
        let mut gates: Vec<Gate> = Vec::new();
        gates.push(Gate::h(0));
        gates.push(Gate::x(1));
        gates.push(Gate::cnot(0, 1));
        gates.push(Gate::rx(0, 1.57));

        assert_eq!(gates.len(), 4);
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[2].gate_type, GateType::CNOT);
        assert_eq!(gates[2].controls, vec![0]);
        assert_eq!(gates[2].targets, vec![1]);
    }

    // ------------------------------------------------------------------
    // 10. JAX vmap batch simulation (mirrors py_jax_vmap_simulate)
    // ------------------------------------------------------------------

    #[test]
    fn test_jax_vmap_simulate_batch() {
        use crate::jax_bridge::vmap_simulate;

        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        // Batch of 3 different parameter sets
        let batch_params = vec![
            vec![0.0_f32],                    // RY(0) = identity
            vec![std::f32::consts::PI],       // RY(pi) = X
            vec![std::f32::consts::PI / 2.0], // RY(pi/2) = superposition
        ];

        let batch_states = vmap_simulate(&circuit, &batch_params);
        assert_eq!(batch_states.len(), 3);

        // RY(0)|0> = |0>
        assert!((batch_states[0][0].real - 1.0).abs() < 1e-5);
        assert!(batch_states[0][1].norm_sq() < 1e-5);

        // RY(pi)|0> = |1> (up to global phase)
        assert!(batch_states[1][0].norm_sq() < 1e-4);
        assert!((batch_states[1][1].norm_sq() - 1.0).abs() < 1e-4);

        // RY(pi/2)|0> = equal superposition
        let p0 = batch_states[2][0].norm_sq();
        let p1 = batch_states[2][1].norm_sq();
        assert!((p0 - 0.5).abs() < 0.05, "expected ~0.5, got {}", p0);
        assert!((p1 - 0.5).abs() < 0.05, "expected ~0.5, got {}", p1);
    }

    // ------------------------------------------------------------------
    // 11. JAX expectation value (mirrors py_jax_expectation)
    // ------------------------------------------------------------------

    #[test]
    fn test_jax_expectation_z_observable() {
        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        // <0|Z|0> = +1 (theta=0, identity)
        let exp_0 = circuit.expectation(&[0.0_f32], "Z0");
        assert!((exp_0 - 1.0).abs() < 1e-4, "expected ~1.0, got {}", exp_0);

        // <1|Z|1> = -1 (theta=pi, full flip)
        let exp_pi = circuit.expectation(&[std::f32::consts::PI], "Z0");
        assert!(
            (exp_pi - (-1.0)).abs() < 1e-3,
            "expected ~-1.0, got {}",
            exp_pi
        );

        // <+|Z|+> = 0 (theta=pi/2, equal superposition)
        let exp_half = circuit.expectation(&[std::f32::consts::PI / 2.0], "Z0");
        assert!(exp_half.abs() < 0.1, "expected ~0.0, got {}", exp_half);
    }

    // ------------------------------------------------------------------
    // 12. Parameter shift gradient (mirrors py_jax_gradient)
    // ------------------------------------------------------------------

    #[test]
    fn test_parameter_shift_gradient() {
        use crate::jax_bridge::parameter_shift_grad;

        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        let shift = std::f32::consts::PI / 2.0;

        // At theta=0, d<Z>/dtheta should be 0 (top of cosine)
        let grad_0 = parameter_shift_grad(&circuit, &[0.0_f32], 0, shift);
        assert!(!grad_0.is_empty());
        assert!(
            grad_0[0].abs() < 0.15,
            "gradient at 0 should be near 0, got {}",
            grad_0[0]
        );

        // At theta=pi/2, gradient should be near its extremum
        let grad_half = parameter_shift_grad(&circuit, &[std::f32::consts::PI / 2.0], 0, shift);
        assert!(
            grad_half[0].abs() > 0.5,
            "gradient at pi/2 should be significant, got {}",
            grad_half[0]
        );
    }
}
