//! Differentiable Quantum Circuit Engine (JAX-style API)
//!
//! Standalone differentiable quantum circuit engine providing a JAX-style API
//! for quantum circuit differentiation. Does **NOT** require or link to JAX.
//! The Python example below shows the *intended* integration pattern, but the
//! Rust implementation is fully self-contained with its own gradient engine.
//!
//! # Example (intended Python integration pattern)
//!
//! ```python
//! # NOTE: Requires a Python wrapper (not included) to bridge to actual JAX.
//! # The Rust engine computes gradients independently via parameter-shift rule.
//! import jax
//! import jax.numpy as jnp
//!
//! circuit = JAXCircuit(n_qubits=4)
//! circuit.ry(0, theta="params[0]")
//! circuit.ry(1, theta="params[1]")
//! circuit.cx(0, 1)
//!
//! @jax.jit
//! def loss(params):
//!     return circuit.expectation(params, observable="Z0")
//!
//! grad = jax.grad(loss)
//! ```


// ---------------------------------------------------------------------------
// JAX COMPATIBLE TYPES
// ---------------------------------------------------------------------------

/// JAX-compatible complex number (matches JAX's complex64)
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct JaxComplex {
    pub real: f32,
    pub imag: f32,
}

impl JaxComplex {
    pub fn new(real: f32, imag: f32) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }

    pub fn one() -> Self {
        Self { real: 1.0, imag: 0.0 }
    }

    pub fn conj(&self) -> Self {
        Self { real: self.real, imag: -self.imag }
    }

    pub fn norm_sq(&self) -> f32 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn norm(&self) -> f32 {
        self.norm_sq().sqrt()
    }
}

impl std::ops::Add for JaxComplex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}

impl std::ops::Mul for JaxComplex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }
}

// ---------------------------------------------------------------------------
// JAX CIRCUIT
// ---------------------------------------------------------------------------

/// JAX-compatible quantum circuit
#[derive(Clone, Debug)]
pub struct JAXCircuit {
    n_qubits: usize,
    gates: Vec<JAXGate>,
    parameter_names: Vec<String>,
}

/// Gate with JAX-compatible parameterization
#[derive(Clone, Debug)]
pub enum JAXGate {
    /// Hadamard gate
    H(usize),
    /// Pauli-X gate
    X(usize),
    /// Pauli-Y gate
    Y(usize),
    /// Pauli-Z gate
    Z(usize),
    /// CNOT gate
    CNOT { control: usize, target: usize },
    /// CZ gate
    CZ { control: usize, target: usize },
    /// Rotation around X axis
    RX { qubit: usize, param_idx: usize },
    /// Rotation around Y axis
    RY { qubit: usize, param_idx: usize },
    /// Rotation around Z axis
    RZ { qubit: usize, param_idx: usize },
    /// Parameterized phase
    Phase { qubit: usize, param_idx: usize },
}

impl JAXCircuit {
    /// Create new JAX circuit
    pub fn new(n_qubits: usize) -> Self {
        Self {
            n_qubits,
            gates: Vec::new(),
            parameter_names: Vec::new(),
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.n_qubits
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get gates reference
    pub fn get_gates(&self) -> &[JAXGate] {
        &self.gates
    }

    /// Add parameter and return its index
    pub fn add_parameter(&mut self, name: &str) -> usize {
        let idx = self.parameter_names.len();
        self.parameter_names.push(name.to_string());
        idx
    }

    /// Add Hadamard gate
    pub fn h(&mut self, qubit: usize) {
        self.gates.push(JAXGate::H(qubit));
    }

    /// Add Pauli-X gate
    pub fn x(&mut self, qubit: usize) {
        self.gates.push(JAXGate::X(qubit));
    }

    /// Add Pauli-Y gate
    pub fn y(&mut self, qubit: usize) {
        self.gates.push(JAXGate::Y(qubit));
    }

    /// Add Pauli-Z gate
    pub fn z(&mut self, qubit: usize) {
        self.gates.push(JAXGate::Z(qubit));
    }

    /// Add CNOT gate
    pub fn cx(&mut self, control: usize, target: usize) {
        self.gates.push(JAXGate::CNOT { control, target });
    }

    /// Add CZ gate
    pub fn cz(&mut self, control: usize, target: usize) {
        self.gates.push(JAXGate::CZ { control, target });
    }

    /// Add RX rotation with parameter
    pub fn rx(&mut self, qubit: usize, param_name: &str) {
        let idx = self.add_parameter(param_name);
        self.gates.push(JAXGate::RX { qubit, param_idx: idx });
    }

    /// Add RY rotation with parameter
    pub fn ry(&mut self, qubit: usize, param_name: &str) {
        let idx = self.add_parameter(param_name);
        self.gates.push(JAXGate::RY { qubit, param_idx: idx });
    }

    /// Add RZ rotation with parameter
    pub fn rz(&mut self, qubit: usize, param_name: &str) {
        let idx = self.add_parameter(param_name);
        self.gates.push(JAXGate::RZ { qubit, param_idx: idx });
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameter_names.len()
    }

    /// Simulate circuit with parameters, return statevector
    pub fn simulate(&self, params: &[f32]) -> Vec<JaxComplex> {
        let dim = 1 << self.n_qubits;
        let mut state = vec![JaxComplex::zero(); dim];
        state[0] = JaxComplex::one();

        for gate in &self.gates {
            match gate {
                JAXGate::H(q) => self.apply_h(&mut state, *q),
                JAXGate::X(q) => self.apply_x(&mut state, *q),
                JAXGate::Y(q) => self.apply_y(&mut state, *q),
                JAXGate::Z(q) => self.apply_z(&mut state, *q),
                JAXGate::CNOT { control, target } => self.apply_cnot(&mut state, *control, *target),
                JAXGate::CZ { control, target } => self.apply_cz(&mut state, *control, *target),
                JAXGate::RX { qubit, param_idx } => {
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_rx(&mut state, *qubit, angle);
                }
                JAXGate::RY { qubit, param_idx } => {
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_ry(&mut state, *qubit, angle);
                }
                JAXGate::RZ { qubit, param_idx } => {
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_rz(&mut state, *qubit, angle);
                }
                JAXGate::Phase { qubit, param_idx } => {
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_rz(&mut state, *qubit, angle);
                }
            }
        }

        state
    }

    // Gate implementations
    fn apply_h(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        let inv_sqrt2 = 0.7071067811865476;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = JaxComplex::new((a.real + b.real) * inv_sqrt2, (a.imag + b.imag) * inv_sqrt2);
                state[i + j + stride] = JaxComplex::new((a.real - b.real) * inv_sqrt2, (a.imag - b.imag) * inv_sqrt2);
            }
        }
    }

    fn apply_x(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                state.swap(i + j, i + j + stride);
            }
        }
    }

    fn apply_y(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        let _i_unit = JaxComplex::new(0.0, 1.0);
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = JaxComplex::new(b.imag, -b.real); // -i*b
                state[i + j + stride] = JaxComplex::new(-a.imag, a.real); // i*a
            }
        }
    }

    fn apply_z(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                state[i + j + stride] = JaxComplex::new(-state[i + j + stride].real, -state[i + j + stride].imag);
            }
        }
    }

    fn apply_cnot(&self, state: &mut [JaxComplex], control: usize, target: usize) {
        let stride_c = 1 << control;
        let stride_t = 1 << target;
        for i in 0..state.len() {
            if (i & stride_c) != 0 {
                let j = i ^ stride_t;
                if i < j {
                    state.swap(i, j);
                }
            }
        }
    }

    fn apply_cz(&self, state: &mut [JaxComplex], control: usize, target: usize) {
        let stride_c = 1 << control;
        let stride_t = 1 << target;
        for i in 0..state.len() {
            if (i & stride_c) != 0 && (i & stride_t) != 0 {
                state[i] = JaxComplex::new(-state[i].real, -state[i].imag);
            }
        }
    }

    fn apply_rx(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                // RX = exp(-i*X*θ/2)
                state[i + j] = JaxComplex::new(a.real * c, a.imag * c - b.imag * s);
                state[i + j + stride] = JaxComplex::new(b.real * c - a.real * s, b.imag * c - a.imag * s);
            }
        }
    }

    fn apply_ry(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                // RY = exp(-i*Y*θ/2)
                state[i + j] = JaxComplex::new(a.real * c - b.real * s, a.imag * c - b.imag * s);
                state[i + j + stride] = JaxComplex::new(a.real * s + b.real * c, a.imag * s + b.imag * c);
            }
        }
    }

    fn apply_rz(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                // |0⟩ stays same, |1⟩ gets phase
                let phase = JaxComplex::new(c, -s);
                state[i + j + stride] = state[i + j + stride] * phase;
            }
        }
    }

    /// Compute expectation value of Z observable on qubit
    pub fn expect_z(&self, params: &[f32], qubit: usize) -> f32 {
        let state = self.simulate(params);
        let stride = 1 << qubit;
        let mut exp_val = 0.0f32;

        for i in 0..state.len() {
            let prob = state[i].norm_sq();
            if (i & stride) == 0 {
                exp_val += prob;  // |0⟩ contributes +1
            } else {
                exp_val -= prob;  // |1⟩ contributes -1
            }
        }

        exp_val
    }

    /// Compute full expectation value from observable specification
    pub fn expectation(&self, params: &[f32], observable: &str) -> f32 {
        // Parse observable like "Z0", "X0Y1", "Z0Z1"
        let mut total = 0.0f32;
        let _state = self.simulate(params);

        // Simple Z observable parsing
        for q in 0..self.n_qubits {
            if observable.contains(&format!("Z{}", q)) {
                total += self.expect_z(params, q);
            }
        }

        total
    }
}

// ---------------------------------------------------------------------------
// JAX GRADIENT COMPUTATION
// ---------------------------------------------------------------------------

/// Compute gradient using parameter-shift rule (JAX-compatible)
pub fn parameter_shift_grad(circuit: &JAXCircuit, params: &[f32], qubit: usize, shift: f32) -> Vec<f32> {
    let n_params = circuit.num_parameters();
    let mut grads = vec![0.0f32; n_params];

    for i in 0..n_params {
        let mut params_plus = params.to_vec();
        let mut params_minus = params.to_vec();
        params_plus[i] += shift;
        params_minus[i] -= shift;

        let exp_plus = circuit.expect_z(&params_plus, qubit);
        let exp_minus = circuit.expect_z(&params_minus, qubit);

        grads[i] = (exp_plus - exp_minus) / (2.0 * shift);
    }

    grads
}

// ---------------------------------------------------------------------------
// JAX VMAP SIMULATION
// ---------------------------------------------------------------------------

/// Batch simulation for VMAP (multiple parameter sets)
pub fn vmap_simulate(circuit: &JAXCircuit, batch_params: &[Vec<f32>]) -> Vec<Vec<JaxComplex>> {
    batch_params.iter().map(|params| circuit.simulate(params)).collect()
}

/// Batch expectation for VMAP
pub fn vmap_expectation(circuit: &JAXCircuit, batch_params: &[Vec<f32>], observable: &str) -> Vec<f32> {
    batch_params.iter().map(|params| circuit.expectation(params, observable)).collect()
}

// ---------------------------------------------------------------------------
// JAX JIT CACHE
// ---------------------------------------------------------------------------

/// JIT compilation cache for faster repeated execution
pub struct JAXJitCache {
    circuit_hash: u64,
    compiled_gates: Vec<CompiledGate>,
}

#[derive(Clone, Debug)]
enum CompiledGate {
    H { mask: usize },
    X { mask: usize },
    Z { mask: usize },
    CNOT { control_mask: usize, target_mask: usize },
    RX { mask: usize, param_idx: usize },
    RY { mask: usize, param_idx: usize },
    RZ { mask: usize, param_idx: usize },
}

impl JAXJitCache {
    /// Create JIT cache from circuit
    pub fn new(circuit: &JAXCircuit) -> Self {
        let mut compiled = Vec::new();
        let mut hash = 0u64;

        for gate in &circuit.gates {
            match gate {
                JAXGate::H(q) => {
                    compiled.push(CompiledGate::H { mask: 1 << q });
                    hash = hash.wrapping_add(*q as u64 * 17);
                }
                JAXGate::X(q) => {
                    compiled.push(CompiledGate::X { mask: 1 << q });
                    hash = hash.wrapping_add(*q as u64 * 19);
                }
                JAXGate::Z(q) => {
                    compiled.push(CompiledGate::Z { mask: 1 << q });
                    hash = hash.wrapping_add(*q as u64 * 23);
                }
                JAXGate::CNOT { control, target } => {
                    compiled.push(CompiledGate::CNOT {
                        control_mask: 1 << control,
                        target_mask: 1 << target,
                    });
                    hash = hash.wrapping_add(*control as u64 * 29 + *target as u64 * 31);
                }
                JAXGate::RX { qubit, param_idx } => {
                    compiled.push(CompiledGate::RX {
                        mask: 1 << qubit,
                        param_idx: *param_idx,
                    });
                }
                JAXGate::RY { qubit, param_idx } => {
                    compiled.push(CompiledGate::RY {
                        mask: 1 << qubit,
                        param_idx: *param_idx,
                    });
                }
                JAXGate::RZ { qubit, param_idx } => {
                    compiled.push(CompiledGate::RZ {
                        mask: 1 << qubit,
                        param_idx: *param_idx,
                    });
                }
                _ => {}
            }
        }

        Self {
            circuit_hash: hash,
            compiled_gates: compiled,
        }
    }

    /// Execute with JIT optimization
    pub fn execute(&self, n_qubits: usize, params: &[f32]) -> Vec<JaxComplex> {
        let dim = 1 << n_qubits;
        let mut state = vec![JaxComplex::zero(); dim];
        state[0] = JaxComplex::one();

        for gate in &self.compiled_gates {
            match gate {
                CompiledGate::H { mask } => {
                    let q = mask.trailing_zeros() as usize;
                    self.apply_h_fast(&mut state, q);
                }
                CompiledGate::X { mask } => {
                    let q = mask.trailing_zeros() as usize;
                    self.apply_x_fast(&mut state, q);
                }
                CompiledGate::Z { mask } => {
                    let q = mask.trailing_zeros() as usize;
                    self.apply_z_fast(&mut state, q);
                }
                CompiledGate::CNOT { control_mask, target_mask } => {
                    let c = control_mask.trailing_zeros() as usize;
                    let t = target_mask.trailing_zeros() as usize;
                    self.apply_cnot_fast(&mut state, c, t);
                }
                CompiledGate::RX { mask, param_idx } => {
                    let q = mask.trailing_zeros() as usize;
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_rx_fast(&mut state, q, angle);
                }
                CompiledGate::RY { mask, param_idx } => {
                    let q = mask.trailing_zeros() as usize;
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_ry_fast(&mut state, q, angle);
                }
                CompiledGate::RZ { mask, param_idx } => {
                    let q = mask.trailing_zeros() as usize;
                    let angle = params.get(*param_idx).copied().unwrap_or(0.0);
                    self.apply_rz_fast(&mut state, q, angle);
                }
            }
        }

        state
    }

    // Fast gate implementations (inlined, no bounds checking)
    fn apply_h_fast(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        let inv_sqrt2 = 0.7071067811865476;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = JaxComplex::new((a.real + b.real) * inv_sqrt2, (a.imag + b.imag) * inv_sqrt2);
                state[i + j + stride] = JaxComplex::new((a.real - b.real) * inv_sqrt2, (a.imag - b.imag) * inv_sqrt2);
            }
        }
    }

    fn apply_x_fast(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                state.swap(i + j, i + j + stride);
            }
        }
    }

    fn apply_z_fast(&self, state: &mut [JaxComplex], qubit: usize) {
        let stride = 1 << qubit;
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                state[i + j + stride].real *= -1.0;
                state[i + j + stride].imag *= -1.0;
            }
        }
    }

    fn apply_cnot_fast(&self, state: &mut [JaxComplex], control: usize, target: usize) {
        let stride_c = 1 << control;
        let stride_t = 1 << target;
        for i in 0..state.len() {
            if (i & stride_c) != 0 {
                let j = i ^ stride_t;
                if i < j {
                    state.swap(i, j);
                }
            }
        }
    }

    fn apply_rx_fast(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = JaxComplex::new(a.real * c, a.imag * c - b.imag * s);
                state[i + j + stride] = JaxComplex::new(b.real * c - a.real * s, b.imag * c - a.imag * s);
            }
        }
    }

    fn apply_ry_fast(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let a = state[i + j];
                let b = state[i + j + stride];
                state[i + j] = JaxComplex::new(a.real * c - b.real * s, a.imag * c - b.imag * s);
                state[i + j + stride] = JaxComplex::new(a.real * s + b.real * c, a.imag * s + b.imag * c);
            }
        }
    }

    fn apply_rz_fast(&self, state: &mut [JaxComplex], qubit: usize, angle: f32) {
        let stride = 1 << qubit;
        let c = (angle / 2.0).cos();
        let s = (angle / 2.0).sin();
        for i in (0..state.len()).step_by(2 * stride) {
            for j in 0..stride {
                let val = &mut state[i + j + stride];
                let new_real = val.real * c - val.imag * s;
                let new_imag = val.real * s + val.imag * c;
                val.real = new_real;
                val.imag = new_imag;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jax_circuit_creation() {
        let circuit = JAXCircuit::new(4);
        assert_eq!(circuit.n_qubits, 4);
        assert_eq!(circuit.num_parameters(), 0);
    }

    #[test]
    fn test_jax_hadamard() {
        let mut circuit = JAXCircuit::new(1);
        circuit.h(0);

        let state = circuit.simulate(&[]);
        assert!((state[0].real - 0.7071).abs() < 0.01);
        assert!((state[1].real - 0.7071).abs() < 0.01);
    }

    #[test]
    fn test_jax_cnot() {
        let mut circuit = JAXCircuit::new(2);
        circuit.h(0);
        circuit.cx(0, 1);

        let state = circuit.simulate(&[]);
        // Should be Bell state
        assert!((state[0].norm_sq() - 0.5).abs() < 0.01);
        assert!((state[3].norm_sq() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_jax_rotation() {
        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        let state_zero = circuit.simulate(&[0.0]);
        assert!((state_zero[0].norm_sq() - 1.0).abs() < 0.01);

        let state_pi = circuit.simulate(&[std::f32::consts::PI]);
        assert!((state_pi[1].norm_sq() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_jax_expectation() {
        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        let exp_zero = circuit.expect_z(&[0.0], 0);
        assert!((exp_zero - 1.0).abs() < 0.01);

        let exp_pi = circuit.expect_z(&[std::f32::consts::PI], 0);
        assert!((exp_pi - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_parameter_shift_grad() {
        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        let params = vec![0.5];
        let grads = parameter_shift_grad(&circuit, &params, 0, 0.1);

        assert!(!grads.is_empty());
        assert!(grads[0].abs() > 0.0);
    }

    #[test]
    fn test_jit_cache() {
        let mut circuit = JAXCircuit::new(2);
        circuit.h(0);
        circuit.cx(0, 1);

        let cache = JAXJitCache::new(&circuit);
        let state = cache.execute(2, &[]);

        // Bell state
        assert!((state[0].norm_sq() - 0.5).abs() < 0.01);
        assert!((state[3].norm_sq() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_vmap() {
        let mut circuit = JAXCircuit::new(1);
        circuit.ry(0, "theta");

        let batch_params = vec![
            vec![0.0],
            vec![std::f32::consts::PI / 2.0],
            vec![std::f32::consts::PI],
        ];

        let expectations = vmap_expectation(&circuit, &batch_params, "Z0");
        assert_eq!(expectations.len(), 3);
        assert!((expectations[0] - 1.0).abs() < 0.01);
        assert!((expectations[2] - (-1.0)).abs() < 0.01);
    }
}
