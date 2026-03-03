//! Sampler/Estimator Primitives for nQPU-Metal
//!
//! Provides the standard Qiskit V2-style Sampler and Estimator abstractions
//! backed by a self-contained state-vector simulator. Libraries program
//! against these primitives rather than raw circuit execution.
//!
//! # Overview
//!
//! - **Sampler**: Execute circuits and return measurement counts / quasi-distributions
//! - **Estimator**: Compute expectation values of observables for given circuits
//! - **CircuitBuilder**: Fluent API for constructing quantum circuits
//! - **Observable**: Pauli-string observables with coefficients
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::primitives::*;
//!
//! // Build a Bell-state circuit
//! let circuit = CircuitBuilder::new(2)
//!     .h(0)
//!     .cx(0, 1)
//!     .measure_all()
//!     .build();
//!
//! // Sample 1024 shots
//! let sampler = Sampler::new(SamplerConfig::default());
//! let result = sampler.run_single(&circuit, 1024);
//! assert!(result.counts.contains_key(&0b00) || result.counts.contains_key(&0b11));
//!
//! // Estimate <ZZ>
//! let obs = Observable::from_string("ZZ");
//! let estimator = Estimator::new(EstimatorConfig::default());
//! let (value, _err) = estimator.run_single(&circuit, &obs);
//! assert!((value - 1.0).abs() < 0.01); // Bell state: <ZZ> = 1
//! ```

use num_complex::Complex64;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::time::Instant;

// ============================================================
// PAULI TYPES
// ============================================================

/// Single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

/// A weighted tensor product of Pauli operators acting on specific qubits.
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Real-valued coefficient.
    pub coefficient: f64,
    /// List of (qubit_index, pauli_operator) pairs.
    pub paulis: Vec<(usize, Pauli)>,
}

impl PauliTerm {
    /// Create a new Pauli term with given coefficient and Pauli operators.
    pub fn new(coefficient: f64, paulis: Vec<(usize, Pauli)>) -> Self {
        Self {
            coefficient,
            paulis,
        }
    }
}

// ============================================================
// OBSERVABLE
// ============================================================

/// A quantum observable expressed as a sum of Pauli terms.
///
/// `O = Σ c_i P_i` where each `P_i` is a tensor product of Pauli matrices.
#[derive(Debug, Clone)]
pub struct Observable {
    /// The Pauli terms comprising this observable.
    pub terms: Vec<PauliTerm>,
}

impl Observable {
    /// Create a single-qubit Z observable: Z_q.
    pub fn z(qubit: usize) -> Self {
        Self {
            terms: vec![PauliTerm::new(1.0, vec![(qubit, Pauli::Z)])],
        }
    }

    /// Create a two-qubit XX correlation observable: X_q0 X_q1.
    pub fn xx(q0: usize, q1: usize) -> Self {
        Self {
            terms: vec![PauliTerm::new(
                1.0,
                vec![(q0, Pauli::X), (q1, Pauli::X)],
            )],
        }
    }

    /// Parse an observable from a Pauli string like `"XYZII"`.
    ///
    /// The string is read left-to-right as qubit 0, 1, 2, ...
    /// Identity (`I`) qubits are omitted from the Pauli term.
    /// The coefficient is 1.0.
    pub fn from_string(s: &str) -> Self {
        let paulis: Vec<(usize, Pauli)> = s
            .chars()
            .enumerate()
            .filter_map(|(i, c)| {
                let p = match c {
                    'I' | 'i' => return None,
                    'X' | 'x' => Pauli::X,
                    'Y' | 'y' => Pauli::Y,
                    'Z' | 'z' => Pauli::Z,
                    _ => panic!("Unknown Pauli character: '{}'", c),
                };
                Some((i, p))
            })
            .collect();
        Self {
            terms: vec![PauliTerm::new(1.0, paulis)],
        }
    }

    /// Create an observable from multiple Pauli terms.
    pub fn from_terms(terms: Vec<PauliTerm>) -> Self {
        Self { terms }
    }
}

// ============================================================
// PRIMITIVE GATE
// ============================================================

/// A quantum gate in the primitive gate set.
#[derive(Debug, Clone)]
pub enum PrimitiveGate {
    /// Hadamard gate on qubit.
    H(usize),
    /// Pauli-X (NOT) gate on qubit.
    X(usize),
    /// Pauli-Y gate on qubit.
    Y(usize),
    /// Pauli-Z gate on qubit.
    Z(usize),
    /// S gate (√Z) on qubit.
    S(usize),
    /// T gate (√S) on qubit.
    T(usize),
    /// Rotation about X-axis by angle (radians).
    Rx(usize, f64),
    /// Rotation about Y-axis by angle (radians).
    Ry(usize, f64),
    /// Rotation about Z-axis by angle (radians).
    Rz(usize, f64),
    /// Controlled-X (CNOT) gate: control, target.
    CX(usize, usize),
    /// Controlled-Z gate: control, target.
    CZ(usize, usize),
    /// SWAP gate between two qubits.
    Swap(usize, usize),
    /// Barrier (no-op, for visualization/scheduling).
    Barrier,
    /// Parameterized rotation about X-axis: Rx(qubit, parameter_name).
    /// The angle is resolved at bind time via `ParametricCircuit::bind`.
    RxParam(usize, String),
    /// Parameterized rotation about Y-axis: Ry(qubit, parameter_name).
    RyParam(usize, String),
    /// Parameterized rotation about Z-axis: Rz(qubit, parameter_name).
    RzParam(usize, String),
}

// ============================================================
// CIRCUIT
// ============================================================

/// A named classical register that maps measurement outcomes to named bit groups.
///
/// In Qiskit V2, circuits can have multiple classical registers (e.g., "c0", "c1")
/// each mapping to a subset of measured qubits. The `BitArray` result type respects
/// these register boundaries.
#[derive(Debug, Clone)]
pub struct ClassicalRegister {
    /// Human-readable name for this register (e.g., "c0", "meas").
    pub name: String,
    /// Qubit indices that feed into this register, in order.
    /// The first qubit maps to bit 0 of the register, etc.
    pub qubits: Vec<usize>,
}

impl ClassicalRegister {
    /// Create a new classical register with a name and qubit mapping.
    pub fn new(name: &str, qubits: Vec<usize>) -> Self {
        Self {
            name: name.to_string(),
            qubits,
        }
    }

    /// Number of classical bits in this register.
    pub fn size(&self) -> usize {
        self.qubits.len()
    }
}

/// A quantum circuit consisting of gates and measurement specifications.
#[derive(Debug, Clone)]
pub struct Circuit {
    /// Ordered list of gates to apply.
    pub gates: Vec<PrimitiveGate>,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Qubits to measure (indices into the state vector).
    pub measurements: Vec<usize>,
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Classical registers for structured measurement output.
    /// If empty, all measurements go into a single implicit register.
    pub classical_registers: Vec<ClassicalRegister>,
    /// Names of unbound parameters (populated for parametric circuits).
    pub parameter_names: Vec<String>,
}

impl Circuit {
    /// Create a new empty circuit with the given number of qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            measurements: Vec::new(),
            name: None,
            classical_registers: Vec::new(),
            parameter_names: Vec::new(),
        }
    }

    /// Returns `true` if this circuit has unbound parameters.
    pub fn is_parametric(&self) -> bool {
        !self.parameter_names.is_empty()
    }

    /// Bind parameter values to a parametric circuit, producing a concrete circuit.
    ///
    /// `params` maps parameter names to their numeric values. All parameters
    /// referenced by `RxParam`/`RyParam`/`RzParam` gates must be present.
    ///
    /// # Errors
    ///
    /// Returns `Err` if a parameter name referenced by a gate is not found in `params`.
    pub fn bind_parameters(&self, params: &HashMap<String, f64>) -> Result<Circuit, String> {
        let mut bound_gates = Vec::with_capacity(self.gates.len());

        for gate in &self.gates {
            let bound = match gate {
                PrimitiveGate::RxParam(q, name) => {
                    let val = params
                        .get(name)
                        .ok_or_else(|| format!("Missing parameter: '{}'", name))?;
                    PrimitiveGate::Rx(*q, *val)
                }
                PrimitiveGate::RyParam(q, name) => {
                    let val = params
                        .get(name)
                        .ok_or_else(|| format!("Missing parameter: '{}'", name))?;
                    PrimitiveGate::Ry(*q, *val)
                }
                PrimitiveGate::RzParam(q, name) => {
                    let val = params
                        .get(name)
                        .ok_or_else(|| format!("Missing parameter: '{}'", name))?;
                    PrimitiveGate::Rz(*q, *val)
                }
                other => other.clone(),
            };
            bound_gates.push(bound);
        }

        Ok(Circuit {
            gates: bound_gates,
            num_qubits: self.num_qubits,
            measurements: self.measurements.clone(),
            name: self.name.clone(),
            classical_registers: self.classical_registers.clone(),
            parameter_names: Vec::new(), // All parameters are now bound
        })
    }
}

// ============================================================
// CIRCUIT BUILDER
// ============================================================

/// Fluent builder for constructing quantum circuits.
///
/// ```rust
/// use nqpu_metal::primitives::CircuitBuilder;
///
/// let circuit = CircuitBuilder::new(3)
///     .h(0)
///     .cx(0, 1)
///     .cx(1, 2)
///     .measure_all()
///     .build();
///
/// assert_eq!(circuit.num_qubits, 3);
/// assert_eq!(circuit.gates.len(), 3);
/// assert_eq!(circuit.measurements.len(), 3);
/// ```
pub struct CircuitBuilder {
    num_qubits: usize,
    gates: Vec<PrimitiveGate>,
    measurements: Vec<usize>,
    name: Option<String>,
    classical_registers: Vec<ClassicalRegister>,
    parameter_names: Vec<String>,
}

impl CircuitBuilder {
    /// Create a new circuit builder for the given number of qubits.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            measurements: Vec::new(),
            name: None,
            classical_registers: Vec::new(),
            parameter_names: Vec::new(),
        }
    }

    /// Set an optional name for the circuit.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Apply a Hadamard gate to qubit `q`.
    pub fn h(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::H(q));
        self
    }

    /// Apply a Pauli-X gate to qubit `q`.
    pub fn x(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::X(q));
        self
    }

    /// Apply a Pauli-Y gate to qubit `q`.
    pub fn y(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::Y(q));
        self
    }

    /// Apply a Pauli-Z gate to qubit `q`.
    pub fn z(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::Z(q));
        self
    }

    /// Apply an S gate to qubit `q`.
    pub fn s(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::S(q));
        self
    }

    /// Apply a T gate to qubit `q`.
    pub fn t(mut self, q: usize) -> Self {
        self.gates.push(PrimitiveGate::T(q));
        self
    }

    /// Apply an Rx rotation to qubit `q` with angle `theta` (radians).
    pub fn rx(mut self, q: usize, angle: f64) -> Self {
        self.gates.push(PrimitiveGate::Rx(q, angle));
        self
    }

    /// Apply an Ry rotation to qubit `q` with angle `theta` (radians).
    pub fn ry(mut self, q: usize, angle: f64) -> Self {
        self.gates.push(PrimitiveGate::Ry(q, angle));
        self
    }

    /// Apply an Rz rotation to qubit `q` with angle `theta` (radians).
    pub fn rz(mut self, q: usize, angle: f64) -> Self {
        self.gates.push(PrimitiveGate::Rz(q, angle));
        self
    }

    /// Apply a CNOT (CX) gate with control `c` and target `t`.
    pub fn cx(mut self, c: usize, t: usize) -> Self {
        self.gates.push(PrimitiveGate::CX(c, t));
        self
    }

    /// Apply a CZ gate between qubits `q0` and `q1`.
    pub fn cz(mut self, q0: usize, q1: usize) -> Self {
        self.gates.push(PrimitiveGate::CZ(q0, q1));
        self
    }

    /// Apply a SWAP gate between qubits `q0` and `q1`.
    pub fn swap(mut self, q0: usize, q1: usize) -> Self {
        self.gates.push(PrimitiveGate::Swap(q0, q1));
        self
    }

    /// Insert a barrier (no-op).
    pub fn barrier(mut self) -> Self {
        self.gates.push(PrimitiveGate::Barrier);
        self
    }

    /// Mark qubit `q` for measurement.
    pub fn measure(mut self, q: usize) -> Self {
        if !self.measurements.contains(&q) {
            self.measurements.push(q);
        }
        self
    }

    /// Mark all qubits for measurement.
    pub fn measure_all(mut self) -> Self {
        self.measurements = (0..self.num_qubits).collect();
        self
    }

    /// Add a classical register that maps specific qubits to a named bit group.
    pub fn add_classical_register(mut self, name: &str, qubits: Vec<usize>) -> Self {
        self.classical_registers
            .push(ClassicalRegister::new(name, qubits));
        self
    }

    /// Add a parametric Rx gate. The angle is resolved at bind time.
    pub fn rx_param(mut self, q: usize, param_name: &str) -> Self {
        self.gates
            .push(PrimitiveGate::RxParam(q, param_name.to_string()));
        if !self.parameter_names.contains(&param_name.to_string()) {
            self.parameter_names.push(param_name.to_string());
        }
        self
    }

    /// Add a parametric Ry gate. The angle is resolved at bind time.
    pub fn ry_param(mut self, q: usize, param_name: &str) -> Self {
        self.gates
            .push(PrimitiveGate::RyParam(q, param_name.to_string()));
        if !self.parameter_names.contains(&param_name.to_string()) {
            self.parameter_names.push(param_name.to_string());
        }
        self
    }

    /// Add a parametric Rz gate. The angle is resolved at bind time.
    pub fn rz_param(mut self, q: usize, param_name: &str) -> Self {
        self.gates
            .push(PrimitiveGate::RzParam(q, param_name.to_string()));
        if !self.parameter_names.contains(&param_name.to_string()) {
            self.parameter_names.push(param_name.to_string());
        }
        self
    }

    /// Consume the builder and produce the circuit.
    pub fn build(self) -> Circuit {
        Circuit {
            gates: self.gates,
            num_qubits: self.num_qubits,
            measurements: self.measurements,
            name: self.name,
            classical_registers: self.classical_registers,
            parameter_names: self.parameter_names,
        }
    }
}

// ============================================================
// STATE VECTOR SIMULATION
// ============================================================

/// Simulate a circuit starting from |0...0> and return the final state vector.
pub fn simulate_circuit(circuit: &Circuit) -> Vec<Complex64> {
    let dim = 1usize << circuit.num_qubits;
    let mut state = vec![Complex64::new(0.0, 0.0); dim];
    state[0] = Complex64::new(1.0, 0.0); // |0...0>

    for gate in &circuit.gates {
        apply_gate(&mut state, gate, circuit.num_qubits);
    }

    state
}

/// Apply a single gate to the state vector in-place.
///
/// The state vector has `2^n` amplitudes where `n = num_qubits`.
/// Qubit ordering: qubit 0 is the most-significant bit (MSB).
/// For a 2-qubit system, basis states are |q0 q1>:
///   index 0 = |00>, index 1 = |01>, index 2 = |10>, index 3 = |11>.
pub fn apply_gate(state: &mut Vec<Complex64>, gate: &PrimitiveGate, num_qubits: usize) {
    match gate {
        PrimitiveGate::H(q) => apply_single_qubit_gate(state, *q, num_qubits, |a, b| {
            let inv_sqrt2 = FRAC_1_SQRT_2;
            (a * inv_sqrt2 + b * inv_sqrt2, a * inv_sqrt2 - b * inv_sqrt2)
        }),

        PrimitiveGate::X(q) => {
            apply_single_qubit_gate(state, *q, num_qubits, |a, b| (b, a))
        }

        PrimitiveGate::Y(q) => apply_single_qubit_gate(state, *q, num_qubits, |a, b| {
            let i = Complex64::new(0.0, 1.0);
            (-i * b, i * a)
        }),

        PrimitiveGate::Z(q) => {
            apply_single_qubit_gate(state, *q, num_qubits, |a, b| (a, -b))
        }

        PrimitiveGate::S(q) => apply_single_qubit_gate(state, *q, num_qubits, |a, b| {
            (a, Complex64::new(0.0, 1.0) * b)
        }),

        PrimitiveGate::T(q) => apply_single_qubit_gate(state, *q, num_qubits, |a, b| {
            let t_phase = Complex64::from_polar(1.0, PI / 4.0);
            (a, t_phase * b)
        }),

        PrimitiveGate::Rx(q, theta) => {
            let half = theta / 2.0;
            let cos_h = half.cos();
            let sin_h = half.sin();
            apply_single_qubit_gate(state, *q, num_qubits, move |a, b| {
                let neg_i = Complex64::new(0.0, -1.0);
                (
                    a * cos_h + b * neg_i * sin_h,
                    a * neg_i * sin_h + b * cos_h,
                )
            })
        }

        PrimitiveGate::Ry(q, theta) => {
            let half = theta / 2.0;
            let cos_h = half.cos();
            let sin_h = half.sin();
            apply_single_qubit_gate(state, *q, num_qubits, move |a, b| {
                (a * cos_h - b * sin_h, a * sin_h + b * cos_h)
            })
        }

        PrimitiveGate::Rz(q, theta) => {
            let half = theta / 2.0;
            let phase_neg = Complex64::from_polar(1.0, -half);
            let phase_pos = Complex64::from_polar(1.0, half);
            apply_single_qubit_gate(state, *q, num_qubits, move |a, b| {
                (a * phase_neg, b * phase_pos)
            })
        }

        PrimitiveGate::CX(control, target) => {
            apply_controlled_gate(state, *control, *target, num_qubits, |a, b| (b, a))
        }

        PrimitiveGate::CZ(q0, q1) => {
            apply_controlled_gate(state, *q0, *q1, num_qubits, |a, b| (a, -b))
        }

        PrimitiveGate::Swap(q0, q1) => {
            // SWAP = CX(q0,q1) CX(q1,q0) CX(q0,q1)
            apply_controlled_gate(state, *q0, *q1, num_qubits, |a, b| (b, a));
            apply_controlled_gate(state, *q1, *q0, num_qubits, |a, b| (b, a));
            apply_controlled_gate(state, *q0, *q1, num_qubits, |a, b| (b, a));
        }

        PrimitiveGate::Barrier => { /* no-op */ }

        PrimitiveGate::RxParam(_, name)
        | PrimitiveGate::RyParam(_, name)
        | PrimitiveGate::RzParam(_, name) => {
            panic!(
                "Unbound parametric gate for parameter '{}'. \
                 Call Circuit::bind_parameters() before simulation.",
                name
            );
        }
    }
}

/// Apply a single-qubit gate to the state vector.
///
/// `gate_fn(|0>, |1>) -> (new|0>, new|1>)` defines the 2x2 unitary.
/// Qubit `q` means we iterate over pairs of basis states that differ only in bit `q`.
fn apply_single_qubit_gate<F>(
    state: &mut [Complex64],
    q: usize,
    num_qubits: usize,
    gate_fn: F,
) where
    F: Fn(Complex64, Complex64) -> (Complex64, Complex64),
{
    let bit = num_qubits - 1 - q; // Convert qubit index to bit position
    let mask = 1usize << bit;

    let dim = state.len();
    for i in 0..dim {
        if i & mask == 0 {
            let j = i | mask;
            let (new_a, new_b) = gate_fn(state[i], state[j]);
            state[i] = new_a;
            state[j] = new_b;
        }
    }
}

/// Apply a controlled single-qubit gate.
///
/// The `gate_fn` is applied to the target qubit only when the control qubit is |1>.
fn apply_controlled_gate<F>(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    num_qubits: usize,
    gate_fn: F,
) where
    F: Fn(Complex64, Complex64) -> (Complex64, Complex64),
{
    let ctrl_bit = num_qubits - 1 - control;
    let targ_bit = num_qubits - 1 - target;
    let ctrl_mask = 1usize << ctrl_bit;
    let targ_mask = 1usize << targ_bit;

    let dim = state.len();
    for i in 0..dim {
        // Only act when control is |1> and target is |0>
        if (i & ctrl_mask != 0) && (i & targ_mask == 0) {
            let j = i | targ_mask;
            let (new_a, new_b) = gate_fn(state[i], state[j]);
            state[i] = new_a;
            state[j] = new_b;
        }
    }
}

/// Compute the measurement probability distribution from a state vector.
///
/// Returns `|amplitude_i|²` for each basis state `i`.
pub fn measure_probabilities(state: &[Complex64]) -> Vec<f64> {
    state.iter().map(|a| a.norm_sqr()).collect()
}

/// Sample `shots` measurement outcomes from a probability distribution.
///
/// Returns a map from basis-state index to the number of times it was observed.
pub fn sample_distribution(
    probs: &[f64],
    shots: usize,
    rng: &mut impl Rng,
) -> HashMap<usize, usize> {
    let dist = WeightedIndex::new(probs);
    let mut counts: HashMap<usize, usize> = HashMap::new();

    match dist {
        Ok(d) => {
            for _ in 0..shots {
                let outcome = d.sample(rng);
                *counts.entry(outcome).or_insert(0) += 1;
            }
        }
        Err(_) => {
            // Degenerate distribution (all zeros): return index 0
            *counts.entry(0).or_insert(0) += shots;
        }
    }

    counts
}

use rand::distributions::WeightedIndex;

/// Compute the expectation value <ψ|P|ψ> for a single Pauli term.
///
/// For a term like `c * Z_0 X_2`, we compute `c * <ψ| (Z⊗I⊗X) |ψ>`.
///
/// Algorithm: for each basis state |k>, apply the Pauli string to get
/// `phase * |k'>`, then accumulate `conj(ψ[k']) * phase * ψ[k]`.
pub fn pauli_expectation(state: &[Complex64], pauli: &PauliTerm) -> f64 {
    let n = (state.len() as f64).log2() as usize;
    let dim = state.len();
    let mut result = Complex64::new(0.0, 0.0);

    for k in 0..dim {
        // Apply Pauli string to |k>: get (phase, k')
        let mut phase = Complex64::new(1.0, 0.0);
        let mut k_prime = k;

        for &(qubit, ref p) in &pauli.paulis {
            let bit = n - 1 - qubit;
            let bit_val = (k >> bit) & 1;

            match p {
                Pauli::I => {} // Identity: no change
                Pauli::X => {
                    // X|0> = |1>, X|1> = |0>
                    k_prime ^= 1 << bit;
                }
                Pauli::Y => {
                    // Y|0> = i|1>, Y|1> = -i|0>
                    k_prime ^= 1 << bit;
                    if bit_val == 0 {
                        phase *= Complex64::new(0.0, 1.0); // i
                    } else {
                        phase *= Complex64::new(0.0, -1.0); // -i
                    }
                }
                Pauli::Z => {
                    // Z|0> = |0>, Z|1> = -|1>
                    if bit_val == 1 {
                        phase *= Complex64::new(-1.0, 0.0);
                    }
                }
            }
        }

        // Accumulate <k'|phase|k> = conj(ψ[k']) * phase * ψ[k]
        result += state[k_prime].conj() * phase * state[k];
    }

    // The expectation value should be real for Hermitian operators.
    // Return the real part scaled by the coefficient.
    pauli.coefficient * result.re
}

// ============================================================
// SAMPLER
// ============================================================

/// Configuration for the Sampler primitive.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    /// Default number of shots per circuit.
    pub shots: usize,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            shots: 1024,
            seed: None,
        }
    }
}

impl SamplerConfig {
    /// Set the default number of shots.
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = shots;
        self
    }

    /// Set the RNG seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A Primitive Unified Bloc (PUB) for the Sampler.
///
/// Packages a circuit with optional per-PUB shot count override.
#[derive(Debug, Clone)]
pub struct SamplerPub {
    /// The quantum circuit to execute.
    pub circuit: Circuit,
    /// Optional per-PUB shot count (overrides sampler default).
    pub shots: Option<usize>,
}

impl SamplerPub {
    /// Create a new SamplerPub with the given circuit.
    pub fn new(circuit: Circuit) -> Self {
        Self {
            circuit,
            shots: None,
        }
    }

    /// Create a new SamplerPub with a specific shot count.
    pub fn with_shots(circuit: Circuit, shots: usize) -> Self {
        Self {
            circuit,
            shots: Some(shots),
        }
    }
}

/// Result from a Sampler execution.
#[derive(Debug, Clone)]
pub struct SamplerResult {
    /// Quasi-probability distribution (counts normalized by shots).
    pub quasi_distribution: HashMap<usize, f64>,
    /// Raw measurement counts by basis-state index.
    pub counts: HashMap<usize, usize>,
    /// Number of shots executed.
    pub shots: usize,
    /// Execution metadata.
    pub metadata: HashMap<String, String>,
}

/// The Sampler primitive: execute circuits and return measurement statistics.
///
/// The sampler runs each circuit through a state-vector simulation, computes
/// the probability distribution from the final state, and samples `shots`
/// bitstrings from it.
pub struct Sampler {
    config: SamplerConfig,
}

impl Sampler {
    /// Create a new Sampler with the given configuration.
    pub fn new(config: SamplerConfig) -> Self {
        Self { config }
    }

    /// Execute a batch of SamplerPubs.
    ///
    /// Returns one `SamplerResult` per PUB, in order.
    pub fn run(&self, pubs: &[SamplerPub]) -> Vec<SamplerResult> {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        pubs.iter()
            .map(|pub_item| {
                let shots = pub_item.shots.unwrap_or(self.config.shots);
                self.execute_circuit(&pub_item.circuit, shots, &mut rng)
            })
            .collect()
    }

    /// Execute a single circuit with the given number of shots.
    pub fn run_single(&self, circuit: &Circuit, shots: usize) -> SamplerResult {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        self.execute_circuit(circuit, shots, &mut rng)
    }

    fn execute_circuit(
        &self,
        circuit: &Circuit,
        shots: usize,
        rng: &mut impl Rng,
    ) -> SamplerResult {
        // Step 1: Simulate the circuit
        let state = simulate_circuit(circuit);

        // Step 2: Compute probability distribution
        let probs = measure_probabilities(&state);

        // Step 3: If measurements are specified, marginalize to measured qubits
        let (effective_probs, measured_qubits) = if circuit.measurements.is_empty() {
            // No measurements specified: measure all qubits
            (probs, (0..circuit.num_qubits).collect::<Vec<_>>())
        } else {
            // Marginalize to specified measurement qubits
            let mq = &circuit.measurements;
            let num_measured = mq.len();
            let out_dim = 1usize << num_measured;
            let mut marginal = vec![0.0; out_dim];

            for (idx, &p) in probs.iter().enumerate() {
                // Extract bits for measured qubits
                let mut out_idx = 0usize;
                for (m, &q) in mq.iter().enumerate() {
                    let bit = circuit.num_qubits - 1 - q;
                    let bit_val = (idx >> bit) & 1;
                    out_idx |= bit_val << (num_measured - 1 - m);
                }
                marginal[out_idx] += p;
            }

            (marginal, mq.clone())
        };

        // Step 4: Sample shots from the distribution
        let counts = sample_distribution(&effective_probs, shots, rng);

        // Step 5: Compute quasi-distribution
        let quasi_distribution: HashMap<usize, f64> = counts
            .iter()
            .map(|(&k, &v)| (k, v as f64 / shots as f64))
            .collect();

        // Build metadata
        let mut metadata = HashMap::new();
        metadata.insert("num_qubits".to_string(), circuit.num_qubits.to_string());
        metadata.insert("shots".to_string(), shots.to_string());
        metadata.insert(
            "measured_qubits".to_string(),
            format!("{:?}", measured_qubits),
        );
        if let Some(ref name) = circuit.name {
            metadata.insert("circuit_name".to_string(), name.clone());
        }

        SamplerResult {
            quasi_distribution,
            counts,
            shots,
            metadata,
        }
    }
}

// ============================================================
// ESTIMATOR
// ============================================================

/// Configuration for the Estimator primitive.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    /// Target precision for shot-based estimation.
    pub precision: f64,
    /// Number of shots. `None` means exact (statevector) computation.
    pub shots: Option<usize>,
    /// Optional RNG seed for shot-based estimation.
    pub seed: Option<u64>,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            precision: 0.01,
            shots: None,
            seed: None,
        }
    }
}

impl EstimatorConfig {
    /// Set target precision.
    pub fn with_precision(mut self, precision: f64) -> Self {
        self.precision = precision;
        self
    }

    /// Set shot count (enables shot-based estimation).
    pub fn with_shots(mut self, shots: usize) -> Self {
        self.shots = Some(shots);
        self
    }

    /// Set the RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A Primitive Unified Bloc (PUB) for the Estimator.
#[derive(Debug, Clone)]
pub struct EstimatorPub {
    /// The quantum circuit to execute.
    pub circuit: Circuit,
    /// Observables to estimate.
    pub observables: Vec<Observable>,
    /// Optional per-PUB precision override.
    pub precision: Option<f64>,
}

impl EstimatorPub {
    /// Create a new EstimatorPub.
    pub fn new(circuit: Circuit, observables: Vec<Observable>) -> Self {
        Self {
            circuit,
            observables,
            precision: None,
        }
    }

    /// Create a new EstimatorPub with a precision override.
    pub fn with_precision(circuit: Circuit, observables: Vec<Observable>, precision: f64) -> Self {
        Self {
            circuit,
            observables,
            precision: Some(precision),
        }
    }
}

/// Result from an Estimator execution.
#[derive(Debug, Clone)]
pub struct EstimatorResult {
    /// Expectation values, one per observable.
    pub values: Vec<f64>,
    /// Standard errors, one per observable.
    pub standard_errors: Vec<f64>,
    /// Execution metadata.
    pub metadata: HashMap<String, String>,
}

/// The Estimator primitive: compute expectation values of observables.
///
/// In exact mode (no shots), the estimator directly computes <ψ|O|ψ>
/// from the state vector. In shot-based mode, it rotates into the
/// measurement basis, samples, and computes statistics.
pub struct Estimator {
    config: EstimatorConfig,
}

impl Estimator {
    /// Create a new Estimator with the given configuration.
    pub fn new(config: EstimatorConfig) -> Self {
        Self { config }
    }

    /// Execute a batch of EstimatorPubs.
    ///
    /// Returns one `EstimatorResult` per PUB, in order.
    pub fn run(&self, pubs: &[EstimatorPub]) -> Vec<EstimatorResult> {
        pubs.iter()
            .map(|pub_item| {
                let state = simulate_circuit(&pub_item.circuit);
                let mut values = Vec::new();
                let mut standard_errors = Vec::new();

                for obs in &pub_item.observables {
                    let (val, err) = self.estimate_observable(&state, obs);
                    values.push(val);
                    standard_errors.push(err);
                }

                let mut metadata = HashMap::new();
                metadata.insert(
                    "num_qubits".to_string(),
                    pub_item.circuit.num_qubits.to_string(),
                );
                metadata.insert(
                    "num_observables".to_string(),
                    pub_item.observables.len().to_string(),
                );
                metadata.insert(
                    "mode".to_string(),
                    if self.config.shots.is_some() {
                        "shot-based"
                    } else {
                        "exact"
                    }
                    .to_string(),
                );

                EstimatorResult {
                    values,
                    standard_errors,
                    metadata,
                }
            })
            .collect()
    }

    /// Estimate a single observable for a single circuit.
    ///
    /// Returns `(expectation_value, standard_error)`.
    pub fn run_single(&self, circuit: &Circuit, observable: &Observable) -> (f64, f64) {
        let state = simulate_circuit(circuit);
        self.estimate_observable(&state, observable)
    }

    fn estimate_observable(&self, state: &[Complex64], observable: &Observable) -> (f64, f64) {
        match self.config.shots {
            None => {
                // Exact computation
                let value: f64 = observable
                    .terms
                    .iter()
                    .map(|term| pauli_expectation(state, term))
                    .sum();
                (value, 0.0) // No statistical error in exact mode
            }
            Some(shots) => {
                // Shot-based estimation
                self.shot_based_estimate(state, observable, shots)
            }
        }
    }

    /// Shot-based estimation of an observable.
    ///
    /// For each Pauli term, we measure in the corresponding basis
    /// and compute the expectation from measurement statistics.
    fn shot_based_estimate(
        &self,
        state: &[Complex64],
        observable: &Observable,
        shots: usize,
    ) -> (f64, f64) {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let n = (state.len() as f64).log2() as usize;
        let mut total_value = 0.0;

        // For each Pauli term, rotate to the computational basis and measure
        for term in &observable.terms {
            let mut rotated_state = state.to_vec();

            // Apply basis rotation for each non-identity Pauli
            for &(qubit, ref p) in &term.paulis {
                match p {
                    Pauli::X => {
                        // Rotate X eigenbasis to Z eigenbasis: apply H
                        apply_single_qubit_gate(&mut rotated_state, qubit, n, |a, b| {
                            let inv_sqrt2 = FRAC_1_SQRT_2;
                            (a * inv_sqrt2 + b * inv_sqrt2, a * inv_sqrt2 - b * inv_sqrt2)
                        });
                    }
                    Pauli::Y => {
                        // Rotate Y eigenbasis to Z eigenbasis: apply S^dag then H
                        apply_single_qubit_gate(&mut rotated_state, qubit, n, |a, b| {
                            (a, Complex64::new(0.0, -1.0) * b) // S^dag
                        });
                        apply_single_qubit_gate(&mut rotated_state, qubit, n, |a, b| {
                            let inv_sqrt2 = FRAC_1_SQRT_2;
                            (a * inv_sqrt2 + b * inv_sqrt2, a * inv_sqrt2 - b * inv_sqrt2)
                        });
                    }
                    Pauli::Z | Pauli::I => {
                        // Already in computational basis
                    }
                }
            }

            // Measure in computational basis
            let probs = measure_probabilities(&rotated_state);
            let counts = sample_distribution(&probs, shots, &mut rng);

            // Compute expectation: for each outcome, the eigenvalue is
            // (-1)^(parity of measured qubits in the Pauli term)
            let mut term_value = 0.0;
            for (&outcome, &count) in &counts {
                let mut parity = 0;
                for &(qubit, ref p) in &term.paulis {
                    if *p != Pauli::I {
                        let bit = n - 1 - qubit;
                        parity ^= (outcome >> bit) & 1;
                    }
                }
                let eigenvalue = if parity == 0 { 1.0 } else { -1.0 };
                term_value += eigenvalue * (count as f64);
            }
            term_value /= shots as f64;

            total_value += term.coefficient * term_value;
        }

        let std_error = 1.0 / (shots as f64).sqrt();
        (total_value, std_error)
    }
}

// ============================================================
// QISKIT V2 PRIMITIVES: BitArray, PubResult types, V2 API
// ============================================================

/// A `SparsePauliOp` entry: one term in a sparse Pauli operator.
///
/// Mirrors Qiskit's `SparsePauliOp` format where each term is specified
/// by a Pauli label string and a coefficient.
#[derive(Debug, Clone)]
pub struct SparsePauliOp {
    /// List of (pauli_label, coefficient) pairs.
    /// Each label is a string like `"XZIY"`, read left-to-right as qubit 0..n-1.
    pub terms: Vec<(String, f64)>,
}

impl SparsePauliOp {
    /// Create a `SparsePauliOp` from a list of `(label, coefficient)` pairs.
    ///
    /// ```rust
    /// use nqpu_metal::primitives::SparsePauliOp;
    ///
    /// let op = SparsePauliOp::new(vec![
    ///     ("ZZ".to_string(), 0.5),
    ///     ("XX".to_string(), 0.3),
    ///     ("II".to_string(), 0.2),
    /// ]);
    /// assert_eq!(op.terms.len(), 3);
    /// ```
    pub fn new(terms: Vec<(String, f64)>) -> Self {
        Self { terms }
    }

    /// Create a single-term `SparsePauliOp` from a Pauli label string with coefficient 1.0.
    pub fn from_label(label: &str) -> Self {
        Self {
            terms: vec![(label.to_string(), 1.0)],
        }
    }

    /// Convert this `SparsePauliOp` into an `Observable`.
    ///
    /// Each `(label, coeff)` pair becomes a `PauliTerm` in the observable.
    pub fn to_observable(&self) -> Observable {
        let terms: Vec<PauliTerm> = self
            .terms
            .iter()
            .map(|(label, coeff)| {
                let paulis: Vec<(usize, Pauli)> = label
                    .chars()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        let p = match c {
                            'I' | 'i' => return None,
                            'X' | 'x' => Pauli::X,
                            'Y' | 'y' => Pauli::Y,
                            'Z' | 'z' => Pauli::Z,
                            _ => panic!("Unknown Pauli character in SparsePauliOp: '{}'", c),
                        };
                        Some((i, p))
                    })
                    .collect();
                PauliTerm::new(*coeff, paulis)
            })
            .collect();
        Observable::from_terms(terms)
    }
}

/// Coerce various observable representations into `Observable`.
///
/// Accepts:
/// - A Pauli string: `"ZZ"`, `"XIYZ"`
/// - A `SparsePauliOp` (converted via `to_observable`)
/// - An existing `Observable`
pub enum ObservableInput {
    /// A single Pauli string with implicit coefficient 1.0.
    PauliString(String),
    /// A sparse Pauli operator (multiple weighted terms).
    Sparse(SparsePauliOp),
    /// An already-constructed observable.
    Observable(Observable),
}

impl ObservableInput {
    /// Resolve this input into a concrete `Observable`.
    pub fn into_observable(self) -> Observable {
        match self {
            ObservableInput::PauliString(s) => Observable::from_string(&s),
            ObservableInput::Sparse(sp) => sp.to_observable(),
            ObservableInput::Observable(obs) => obs,
        }
    }
}

impl From<&str> for ObservableInput {
    fn from(s: &str) -> Self {
        ObservableInput::PauliString(s.to_string())
    }
}

impl From<SparsePauliOp> for ObservableInput {
    fn from(sp: SparsePauliOp) -> Self {
        ObservableInput::Sparse(sp)
    }
}

impl From<Observable> for ObservableInput {
    fn from(obs: Observable) -> Self {
        ObservableInput::Observable(obs)
    }
}

// ============================================================
// BITARRAY
// ============================================================

/// Compact per-shot bitstring storage for Qiskit V2 `SamplerV2` results.
///
/// Each measurement shot produces a bitstring of length `num_bits`. These are
/// stored contiguously in a flat byte buffer using one bit per classical bit,
/// packed MSB-first within each byte.
///
/// For a circuit measuring 3 qubits over 1024 shots, the `BitArray` uses
/// `1024 * ceil(3/8)` = 1024 bytes, compared to 1024 `usize` values in the
/// old counts-based representation.
#[derive(Debug, Clone)]
pub struct BitArray {
    /// Packed bit data. Each shot occupies `bytes_per_shot` consecutive bytes.
    data: Vec<u8>,
    /// Number of classical bits per shot (i.e., number of measured qubits).
    num_bits: usize,
    /// Number of shots stored.
    num_shots: usize,
}

impl BitArray {
    /// Number of bytes needed to store `num_bits` classical bits.
    fn bytes_per_shot(num_bits: usize) -> usize {
        (num_bits + 7) / 8
    }

    /// Create a new, zero-initialized `BitArray`.
    pub fn new(num_bits: usize, num_shots: usize) -> Self {
        let total_bytes = Self::bytes_per_shot(num_bits) * num_shots;
        Self {
            data: vec![0u8; total_bytes],
            num_bits,
            num_shots,
        }
    }

    /// Create a `BitArray` from a vector of integer bitstrings.
    ///
    /// Each element of `bitstrings` is the integer representation of one shot's
    /// measurement outcome (e.g., `0b101` for a 3-qubit "101" result).
    pub fn from_bitstrings(num_bits: usize, bitstrings: &[usize]) -> Self {
        let num_shots = bitstrings.len();
        let bps = Self::bytes_per_shot(num_bits);
        let mut data = vec![0u8; bps * num_shots];

        for (shot, &value) in bitstrings.iter().enumerate() {
            let offset = shot * bps;
            // Pack the integer value into bytes, MSB-first
            for byte_idx in 0..bps {
                let shift = (bps - 1 - byte_idx) * 8;
                data[offset + byte_idx] = ((value >> shift) & 0xFF) as u8;
            }
        }

        Self {
            data,
            num_bits,
            num_shots,
        }
    }

    /// Number of classical bits per shot.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Number of shots stored.
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Retrieve the bitstring for a specific shot as an integer.
    ///
    /// # Panics
    ///
    /// Panics if `shot >= self.num_shots`.
    pub fn get_bitstring(&self, shot: usize) -> usize {
        assert!(shot < self.num_shots, "Shot index {} out of range", shot);
        let bps = Self::bytes_per_shot(self.num_bits);
        let offset = shot * bps;
        let mut value = 0usize;
        for byte_idx in 0..bps {
            value = (value << 8) | (self.data[offset + byte_idx] as usize);
        }
        // Mask to num_bits to discard padding bits
        if self.num_bits < std::mem::size_of::<usize>() * 8 {
            value &= (1usize << self.num_bits) - 1;
        }
        value
    }

    /// Get a specific bit for a specific shot.
    ///
    /// Bit 0 is the most-significant measured qubit (matching Qiskit convention).
    ///
    /// # Panics
    ///
    /// Panics if `shot >= num_shots` or `bit >= num_bits`.
    pub fn get_bit(&self, shot: usize, bit: usize) -> bool {
        assert!(shot < self.num_shots, "Shot index {} out of range", shot);
        assert!(bit < self.num_bits, "Bit index {} out of range", bit);
        let bps = Self::bytes_per_shot(self.num_bits);
        let offset = shot * bps;
        // bit 0 is MSB of the bitstring
        let total_bit_offset = (bps * 8 - self.num_bits) + bit;
        let byte_idx = total_bit_offset / 8;
        let bit_in_byte = 7 - (total_bit_offset % 8);
        (self.data[offset + byte_idx] >> bit_in_byte) & 1 == 1
    }

    /// Compute measurement counts from the stored bitstrings.
    ///
    /// Returns a `HashMap<usize, usize>` mapping bitstring integer values to
    /// the number of times they were observed, identical to the V1 counts format.
    pub fn get_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for shot in 0..self.num_shots {
            let bs = self.get_bitstring(shot);
            *counts.entry(bs).or_insert(0) += 1;
        }
        counts
    }

    /// Extract a sub-`BitArray` for a specific classical register.
    ///
    /// Given a register that maps to qubit indices within the measured qubits,
    /// extract just those bits for each shot.
    pub fn slice_register(&self, register_bits: &[usize]) -> BitArray {
        let new_num_bits = register_bits.len();
        let mut bitstrings = Vec::with_capacity(self.num_shots);

        for shot in 0..self.num_shots {
            let full_bs = self.get_bitstring(shot);
            let mut reg_value = 0usize;
            for (i, &bit_idx) in register_bits.iter().enumerate() {
                // Extract bit at position bit_idx from the full bitstring
                let bit_val =
                    (full_bs >> (self.num_bits - 1 - bit_idx)) & 1;
                reg_value |= bit_val << (new_num_bits - 1 - i);
            }
            bitstrings.push(reg_value);
        }

        BitArray::from_bitstrings(new_num_bits, &bitstrings)
    }

    /// Return raw byte data (useful for serialization or FFI).
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }
}

// ============================================================
// EXECUTION METADATA
// ============================================================

/// Structured execution metadata for V2 results.
///
/// Captures timing, backend identification, and execution parameters.
#[derive(Debug, Clone)]
pub struct ExecutionMetadata {
    /// Total wall-clock time for this PUB execution (seconds).
    pub execution_time_secs: f64,
    /// Backend identifier string.
    pub backend_name: String,
    /// Backend version string.
    pub backend_version: String,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Number of shots actually executed.
    pub shots_executed: usize,
    /// Additional key-value metadata.
    pub extra: HashMap<String, String>,
}

impl ExecutionMetadata {
    /// Create a new metadata instance with timing and backend info.
    fn new_nqpu(num_qubits: usize, shots: usize, elapsed_secs: f64) -> Self {
        Self {
            execution_time_secs: elapsed_secs,
            backend_name: "nqpu-metal-statevector".to_string(),
            backend_version: env!("CARGO_PKG_VERSION").to_string(),
            num_qubits,
            shots_executed: shots,
            extra: HashMap::new(),
        }
    }

    /// Convert to a flat `HashMap<String, String>` for backward compatibility.
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("backend_name".to_string(), self.backend_name.clone());
        map.insert("backend_version".to_string(), self.backend_version.clone());
        map.insert("num_qubits".to_string(), self.num_qubits.to_string());
        map.insert("shots".to_string(), self.shots_executed.to_string());
        map.insert(
            "execution_time_secs".to_string(),
            format!("{:.6}", self.execution_time_secs),
        );
        for (k, v) in &self.extra {
            map.insert(k.clone(), v.clone());
        }
        map
    }
}

// ============================================================
// V2 SAMPLER PUB RESULT
// ============================================================

/// Result for a single Sampler V2 PUB execution.
///
/// This is the Qiskit V2 result type: instead of a quasi-distribution,
/// it stores raw per-shot bitstrings in a `BitArray`, with optional
/// per-register slicing.
#[derive(Debug, Clone)]
pub struct SamplerPubResult {
    /// Per-shot measurement bitstrings.
    pub data: BitArray,
    /// Per-register `BitArray` slices, keyed by register name.
    /// Empty if the circuit has no explicit classical registers.
    pub register_data: HashMap<String, BitArray>,
    /// Structured execution metadata.
    pub metadata: ExecutionMetadata,
}

impl SamplerPubResult {
    /// Convenience: get measurement counts from the BitArray.
    pub fn get_counts(&self) -> HashMap<usize, usize> {
        self.data.get_counts()
    }

    /// Convenience: get counts for a specific classical register.
    ///
    /// Returns `None` if the named register does not exist.
    pub fn get_register_counts(&self, name: &str) -> Option<HashMap<usize, usize>> {
        self.register_data.get(name).map(|ba| ba.get_counts())
    }

    /// Number of shots in this result.
    pub fn num_shots(&self) -> usize {
        self.data.num_shots()
    }

    /// Number of measured classical bits.
    pub fn num_bits(&self) -> usize {
        self.data.num_bits()
    }
}

// ============================================================
// V2 ESTIMATOR PUB RESULT
// ============================================================

/// Result for a single Estimator V2 PUB execution.
///
/// Contains expectation values and standard errors for each observable
/// in the PUB, plus structured metadata including precision achieved.
#[derive(Debug, Clone)]
pub struct EstimatorPubResult {
    /// Expectation values, one per observable in the PUB.
    pub values: Vec<f64>,
    /// Standard errors (1-sigma), one per observable.
    pub standard_errors: Vec<f64>,
    /// Structured execution metadata.
    pub metadata: ExecutionMetadata,
}

impl EstimatorPubResult {
    /// Check whether all standard errors are within the target precision.
    pub fn meets_precision(&self, target: f64) -> bool {
        self.standard_errors.iter().all(|&se| se <= target)
    }
}

// ============================================================
// V2 SAMPLER PUB (extended)
// ============================================================

/// A V2 Primitive Unified Bloc for the Sampler with parameter binding support.
///
/// Extends the basic `SamplerPub` with:
/// - `parameter_values`: bindings for parametric circuits
/// - Full `SamplerPubResult` (BitArray) output
#[derive(Debug, Clone)]
pub struct SamplerPubV2 {
    /// The quantum circuit to execute (may be parametric).
    pub circuit: Circuit,
    /// Parameter bindings for parametric circuits.
    /// Each `HashMap` maps parameter names to values, representing one
    /// set of bindings. Multiple maps enable batched parameter sweeps.
    pub parameter_values: Vec<HashMap<String, f64>>,
    /// Number of shots per parameter binding (overrides sampler default).
    pub shots: Option<usize>,
}

impl SamplerPubV2 {
    /// Create a V2 PUB for a non-parametric circuit.
    pub fn new(circuit: Circuit) -> Self {
        Self {
            circuit,
            parameter_values: Vec::new(),
            shots: None,
        }
    }

    /// Create a V2 PUB with a specific shot count.
    pub fn with_shots(circuit: Circuit, shots: usize) -> Self {
        Self {
            circuit,
            parameter_values: Vec::new(),
            shots: Some(shots),
        }
    }

    /// Create a V2 PUB with parameter bindings.
    pub fn with_parameters(
        circuit: Circuit,
        parameter_values: Vec<HashMap<String, f64>>,
    ) -> Self {
        Self {
            circuit,
            parameter_values,
            shots: None,
        }
    }

    /// Create a V2 PUB with parameter bindings and a shot count.
    pub fn with_parameters_and_shots(
        circuit: Circuit,
        parameter_values: Vec<HashMap<String, f64>>,
        shots: usize,
    ) -> Self {
        Self {
            circuit,
            parameter_values,
            shots: Some(shots),
        }
    }
}

// ============================================================
// V2 ESTIMATOR PUB (extended)
// ============================================================

/// A V2 Primitive Unified Bloc for the Estimator with parameter binding
/// and observable coercion support.
#[derive(Debug, Clone)]
pub struct EstimatorPubV2 {
    /// The quantum circuit to execute (may be parametric).
    pub circuit: Circuit,
    /// Observables to estimate.
    pub observables: Vec<Observable>,
    /// Parameter bindings for parametric circuits.
    pub parameter_values: Vec<HashMap<String, f64>>,
    /// Optional per-PUB precision target (overrides estimator default).
    pub precision: Option<f64>,
}

impl EstimatorPubV2 {
    /// Create a V2 EstimatorPub from a circuit and observables.
    pub fn new(circuit: Circuit, observables: Vec<Observable>) -> Self {
        Self {
            circuit,
            observables,
            parameter_values: Vec::new(),
            precision: None,
        }
    }

    /// Create a V2 EstimatorPub with a precision target.
    pub fn with_precision(
        circuit: Circuit,
        observables: Vec<Observable>,
        precision: f64,
    ) -> Self {
        Self {
            circuit,
            observables,
            parameter_values: Vec::new(),
            precision: Some(precision),
        }
    }

    /// Create a V2 EstimatorPub with parameter bindings.
    pub fn with_parameters(
        circuit: Circuit,
        observables: Vec<Observable>,
        parameter_values: Vec<HashMap<String, f64>>,
    ) -> Self {
        Self {
            circuit,
            observables,
            parameter_values,
            precision: None,
        }
    }

    /// Create from observable inputs (coerces Pauli strings / SparsePauliOp).
    pub fn from_inputs(circuit: Circuit, inputs: Vec<ObservableInput>) -> Self {
        let observables: Vec<Observable> =
            inputs.into_iter().map(|i| i.into_observable()).collect();
        Self::new(circuit, observables)
    }
}

// ============================================================
// V2 SAMPLER IMPLEMENTATION
// ============================================================

impl Sampler {
    /// Execute a batch of V2 SamplerPubs, returning V2 results with `BitArray`.
    ///
    /// For parametric circuits, each parameter binding set produces its own
    /// `SamplerPubResult`. Non-parametric circuits produce exactly one result.
    ///
    /// This is the Qiskit V2 `SamplerV2.run()` equivalent.
    pub fn run_v2(&self, pubs: &[SamplerPubV2]) -> Vec<Vec<SamplerPubResult>> {
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        pubs.iter()
            .map(|pub_item| {
                let shots = pub_item.shots.unwrap_or(self.config.shots);

                if pub_item.parameter_values.is_empty() {
                    // Non-parametric: single execution
                    vec![self.execute_circuit_v2(&pub_item.circuit, shots, &mut rng)]
                } else {
                    // Parametric: one execution per parameter binding
                    pub_item
                        .parameter_values
                        .iter()
                        .map(|params| {
                            let bound = pub_item
                                .circuit
                                .bind_parameters(params)
                                .expect("Parameter binding failed in SamplerV2");
                            self.execute_circuit_v2(&bound, shots, &mut rng)
                        })
                        .collect()
                }
            })
            .collect()
    }

    /// Internal: execute a single bound circuit and produce a V2 result.
    fn execute_circuit_v2(
        &self,
        circuit: &Circuit,
        shots: usize,
        rng: &mut impl Rng,
    ) -> SamplerPubResult {
        let start = Instant::now();

        // Step 1: Simulate
        let state = simulate_circuit(circuit);

        // Step 2: Probability distribution
        let probs = measure_probabilities(&state);

        // Step 3: Marginalize to measured qubits
        let measured_qubits = if circuit.measurements.is_empty() {
            (0..circuit.num_qubits).collect::<Vec<_>>()
        } else {
            circuit.measurements.clone()
        };
        let num_measured = measured_qubits.len();

        let (effective_probs, _) = if circuit.measurements.is_empty() {
            (probs, measured_qubits.clone())
        } else {
            let out_dim = 1usize << num_measured;
            let mut marginal = vec![0.0; out_dim];
            for (idx, &p) in probs.iter().enumerate() {
                let mut out_idx = 0usize;
                for (m, &q) in measured_qubits.iter().enumerate() {
                    let bit = circuit.num_qubits - 1 - q;
                    let bit_val = (idx >> bit) & 1;
                    out_idx |= bit_val << (num_measured - 1 - m);
                }
                marginal[out_idx] += p;
            }
            (marginal, measured_qubits.clone())
        };

        // Step 4: Sample shots and store as BitArray
        let dist = WeightedIndex::new(&effective_probs);
        let mut bitstrings = Vec::with_capacity(shots);

        match dist {
            Ok(d) => {
                for _ in 0..shots {
                    bitstrings.push(d.sample(rng));
                }
            }
            Err(_) => {
                // Degenerate distribution
                for _ in 0..shots {
                    bitstrings.push(0);
                }
            }
        }

        let bit_array = BitArray::from_bitstrings(num_measured, &bitstrings);

        // Step 5: Slice into classical registers if defined
        let mut register_data = HashMap::new();
        for reg in &circuit.classical_registers {
            // Map register qubit indices to positions within measured_qubits
            let reg_bit_positions: Vec<usize> = reg
                .qubits
                .iter()
                .filter_map(|q| measured_qubits.iter().position(|mq| mq == q))
                .collect();
            if !reg_bit_positions.is_empty() {
                register_data.insert(
                    reg.name.clone(),
                    bit_array.slice_register(&reg_bit_positions),
                );
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let metadata = ExecutionMetadata::new_nqpu(circuit.num_qubits, shots, elapsed);

        SamplerPubResult {
            data: bit_array,
            register_data,
            metadata,
        }
    }
}

// ============================================================
// V2 ESTIMATOR IMPLEMENTATION
// ============================================================

impl Estimator {
    /// Execute a batch of V2 EstimatorPubs, returning V2 results.
    ///
    /// For parametric circuits, each parameter binding set produces its own
    /// `EstimatorPubResult`. Non-parametric circuits produce exactly one result.
    ///
    /// When a precision target is set and `config.shots` is `None` (exact mode),
    /// the estimator uses exact state-vector computation (zero standard error).
    /// When `config.shots` is set, the estimator uses adaptive shot allocation
    /// to meet the precision target.
    ///
    /// This is the Qiskit V2 `EstimatorV2.run()` equivalent.
    pub fn run_v2(&self, pubs: &[EstimatorPubV2]) -> Vec<Vec<EstimatorPubResult>> {
        pubs.iter()
            .map(|pub_item| {
                if pub_item.parameter_values.is_empty() {
                    vec![self.execute_estimator_v2(
                        &pub_item.circuit,
                        &pub_item.observables,
                        pub_item.precision,
                    )]
                } else {
                    pub_item
                        .parameter_values
                        .iter()
                        .map(|params| {
                            let bound = pub_item
                                .circuit
                                .bind_parameters(params)
                                .expect("Parameter binding failed in EstimatorV2");
                            self.execute_estimator_v2(
                                &bound,
                                &pub_item.observables,
                                pub_item.precision,
                            )
                        })
                        .collect()
                }
            })
            .collect()
    }

    /// Internal: execute estimator for a single bound circuit.
    fn execute_estimator_v2(
        &self,
        circuit: &Circuit,
        observables: &[Observable],
        precision_override: Option<f64>,
    ) -> EstimatorPubResult {
        let start = Instant::now();
        let state = simulate_circuit(circuit);

        let target_precision = precision_override.unwrap_or(self.config.precision);

        let (values, standard_errors) = match self.config.shots {
            None => {
                // Exact computation: zero standard error
                let vals: Vec<f64> = observables
                    .iter()
                    .map(|obs| {
                        obs.terms
                            .iter()
                            .map(|term| pauli_expectation(&state, term))
                            .sum()
                    })
                    .collect();
                let errs = vec![0.0; vals.len()];
                (vals, errs)
            }
            Some(base_shots) => {
                // Shot-based with adaptive precision targeting.
                // Compute the number of shots needed to achieve the target precision:
                // std_error = 1/sqrt(N)  =>  N = 1/precision^2
                let precision_shots =
                    (1.0 / (target_precision * target_precision)).ceil() as usize;
                let shots = precision_shots.max(base_shots);

                let mut vals = Vec::with_capacity(observables.len());
                let mut errs = Vec::with_capacity(observables.len());

                for obs in observables {
                    let (val, err) = self.shot_based_estimate(&state, obs, shots);
                    vals.push(val);
                    errs.push(err);
                }
                (vals, errs)
            }
        };

        let elapsed = start.elapsed().as_secs_f64();
        let shots_used = match self.config.shots {
            None => 0,
            Some(base) => {
                let precision_shots =
                    (1.0 / (target_precision * target_precision)).ceil() as usize;
                precision_shots.max(base)
            }
        };
        let mut metadata =
            ExecutionMetadata::new_nqpu(circuit.num_qubits, shots_used, elapsed);
        metadata
            .extra
            .insert("target_precision".to_string(), format!("{:.6}", target_precision));
        metadata.extra.insert(
            "mode".to_string(),
            if self.config.shots.is_some() {
                "shot-based".to_string()
            } else {
                "exact".to_string()
            },
        );

        EstimatorPubResult {
            values,
            standard_errors,
            metadata,
        }
    }
}

// ============================================================
// BACKWARD COMPATIBILITY: V1 <-> V2 CONVERSIONS
// ============================================================

impl SamplerPub {
    /// Convert a V1 `SamplerPub` to a V2 `SamplerPubV2`.
    pub fn to_v2(&self) -> SamplerPubV2 {
        SamplerPubV2 {
            circuit: self.circuit.clone(),
            parameter_values: Vec::new(),
            shots: self.shots,
        }
    }
}

impl EstimatorPub {
    /// Convert a V1 `EstimatorPub` to a V2 `EstimatorPubV2`.
    pub fn to_v2(&self) -> EstimatorPubV2 {
        EstimatorPubV2 {
            circuit: self.circuit.clone(),
            observables: self.observables.clone(),
            parameter_values: Vec::new(),
            precision: self.precision,
        }
    }
}

impl SamplerPubResult {
    /// Convert a V2 result to V1 `SamplerResult` for backward compatibility.
    pub fn to_v1(&self) -> SamplerResult {
        let counts = self.data.get_counts();
        let shots = self.data.num_shots();
        let quasi_distribution: HashMap<usize, f64> = counts
            .iter()
            .map(|(&k, &v)| (k, v as f64 / shots as f64))
            .collect();

        SamplerResult {
            quasi_distribution,
            counts,
            shots,
            metadata: self.metadata.to_map(),
        }
    }
}

impl EstimatorPubResult {
    /// Convert a V2 result to V1 `EstimatorResult` for backward compatibility.
    pub fn to_v1(&self) -> EstimatorResult {
        EstimatorResult {
            values: self.values.clone(),
            standard_errors: self.standard_errors.clone(),
            metadata: self.metadata.to_map(),
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: CircuitBuilder creates valid circuit
    #[test]
    fn test_circuit_builder_creates_valid_circuit() {
        let circuit = CircuitBuilder::new(3)
            .h(0)
            .cx(0, 1)
            .x(2)
            .measure(0)
            .measure(1)
            .build();

        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.gates.len(), 3);
        assert_eq!(circuit.measurements.len(), 2);
        assert_eq!(circuit.measurements[0], 0);
        assert_eq!(circuit.measurements[1], 1);
    }

    // Test 2: Simulate H gate gives equal superposition
    #[test]
    fn test_simulate_h_gate_equal_superposition() {
        let circuit = CircuitBuilder::new(1).h(0).build();
        let state = simulate_circuit(&circuit);

        let inv_sqrt2 = FRAC_1_SQRT_2;
        assert!((state[0].re - inv_sqrt2).abs() < 1e-10);
        assert!((state[1].re - inv_sqrt2).abs() < 1e-10);
        assert!(state[0].im.abs() < 1e-10);
        assert!(state[1].im.abs() < 1e-10);
    }

    // Test 3: Simulate Bell state: 50/50 |00> + |11>
    #[test]
    fn test_simulate_bell_state() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let state = simulate_circuit(&circuit);
        // |00> has amplitude 1/√2
        assert!((state[0b00].norm_sqr() - 0.5).abs() < 1e-10);
        // |01> has amplitude 0
        assert!(state[0b01].norm_sqr() < 1e-10);
        // |10> has amplitude 0
        assert!(state[0b10].norm_sqr() < 1e-10);
        // |11> has amplitude 1/√2
        assert!((state[0b11].norm_sqr() - 0.5).abs() < 1e-10);
    }

    // Test 4: Sampler H gate gives ~50/50 counts
    #[test]
    fn test_sampler_h_gate_roughly_even() {
        let circuit = CircuitBuilder::new(1).h(0).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(10000).with_seed(42));
        let result = sampler.run_single(&circuit, 10000);

        assert_eq!(result.shots, 10000);
        let count_0 = *result.counts.get(&0).unwrap_or(&0);
        let count_1 = *result.counts.get(&1).unwrap_or(&0);
        assert_eq!(count_0 + count_1, 10000);

        // Should be roughly 50/50 within statistical tolerance
        let ratio = count_0 as f64 / 10000.0;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "Expected ~50% |0>, got {:.1}%",
            ratio * 100.0
        );
    }

    // Test 5: Sampler X gate gives 100% |1>
    #[test]
    fn test_sampler_x_gate_all_ones() {
        let circuit = CircuitBuilder::new(1).x(0).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));
        let result = sampler.run_single(&circuit, 1024);

        assert_eq!(*result.counts.get(&1).unwrap_or(&0), 1024);
        assert_eq!(*result.counts.get(&0).unwrap_or(&0), 0);
    }

    // Test 6: Sampler Bell state only has |00> and |11>
    #[test]
    fn test_sampler_bell_state_only_00_and_11() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(4096).with_seed(42));
        let result = sampler.run_single(&circuit, 4096);

        // Should only have |00> (0) and |11> (3)
        assert_eq!(*result.counts.get(&0b01).unwrap_or(&0), 0);
        assert_eq!(*result.counts.get(&0b10).unwrap_or(&0), 0);

        let count_00 = *result.counts.get(&0b00).unwrap_or(&0);
        let count_11 = *result.counts.get(&0b11).unwrap_or(&0);
        assert_eq!(count_00 + count_11, 4096);
        assert!(count_00 > 1500, "Expected significant |00> counts");
        assert!(count_11 > 1500, "Expected significant |11> counts");
    }

    // Test 7: Sampler batch: multiple circuits executed
    #[test]
    fn test_sampler_batch_execution() {
        let circuit_h = CircuitBuilder::new(1).h(0).measure_all().build();
        let circuit_x = CircuitBuilder::new(1).x(0).measure_all().build();

        let pubs = vec![
            SamplerPub::new(circuit_h),
            SamplerPub::new(circuit_x),
        ];

        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));
        let results = sampler.run(&pubs);

        assert_eq!(results.len(), 2);

        // First circuit (H): should have both 0 and 1
        assert!(results[0].counts.len() >= 1);
        assert_eq!(
            results[0].counts.values().sum::<usize>(),
            1024
        );

        // Second circuit (X): should only have 1
        assert_eq!(*results[1].counts.get(&1).unwrap_or(&0), 1024);
    }

    // Test 8: Estimator exact: <Z> = 1 for |0>
    #[test]
    fn test_estimator_z_expectation_zero_state() {
        let circuit = CircuitBuilder::new(1).build(); // |0>
        let obs = Observable::z(0);
        let estimator = Estimator::new(EstimatorConfig::default());
        let (value, err) = estimator.run_single(&circuit, &obs);

        assert!(
            (value - 1.0).abs() < 1e-10,
            "Expected <Z> = 1 for |0>, got {}",
            value
        );
        assert!(err.abs() < 1e-10, "Exact mode should have zero error");
    }

    // Test 9: Estimator exact: <Z> = -1 for |1> (X|0>)
    #[test]
    fn test_estimator_z_expectation_one_state() {
        let circuit = CircuitBuilder::new(1).x(0).build(); // X|0> = |1>
        let obs = Observable::z(0);
        let estimator = Estimator::new(EstimatorConfig::default());
        let (value, err) = estimator.run_single(&circuit, &obs);

        assert!(
            (value - (-1.0)).abs() < 1e-10,
            "Expected <Z> = -1 for |1>, got {}",
            value
        );
        assert!(err.abs() < 1e-10);
    }

    // Test 10: Estimator exact: <XX> for Bell state = 1
    #[test]
    fn test_estimator_xx_bell_state() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let obs = Observable::xx(0, 1);
        let estimator = Estimator::new(EstimatorConfig::default());
        let (value, err) = estimator.run_single(&circuit, &obs);

        assert!(
            (value - 1.0).abs() < 1e-10,
            "Expected <XX> = 1 for Bell state, got {}",
            value
        );
        assert!(err.abs() < 1e-10);
    }

    // Test 11: Estimator shot-based matches exact within 3-sigma
    #[test]
    fn test_estimator_shot_based_matches_exact() {
        let circuit = CircuitBuilder::new(1).h(0).build(); // (|0> + |1>)/√2
        let obs = Observable::z(0);

        // Exact: <Z> should be 0 for equal superposition
        let exact_estimator = Estimator::new(EstimatorConfig::default());
        let (exact_val, _) = exact_estimator.run_single(&circuit, &obs);

        // Shot-based
        let shot_estimator = Estimator::new(
            EstimatorConfig::default()
                .with_shots(100_000)
                .with_seed(42),
        );
        let (shot_val, std_err) = shot_estimator.run_single(&circuit, &obs);

        assert!(
            (shot_val - exact_val).abs() < 3.0 * std_err + 0.01,
            "Shot-based ({}) should match exact ({}) within 3*sigma ({})",
            shot_val,
            exact_val,
            3.0 * std_err
        );
    }

    // Test 12: Observable from string "ZZI" has correct qubits
    #[test]
    fn test_observable_from_string_zzi() {
        let obs = Observable::from_string("ZZI");
        assert_eq!(obs.terms.len(), 1);

        let term = &obs.terms[0];
        assert!((term.coefficient - 1.0).abs() < 1e-10);
        assert_eq!(term.paulis.len(), 2); // I is filtered out
        assert_eq!(term.paulis[0], (0, Pauli::Z));
        assert_eq!(term.paulis[1], (1, Pauli::Z));
    }

    // Test 13: Per-PUB shots override default
    #[test]
    fn test_per_pub_shots_override() {
        let circuit = CircuitBuilder::new(1).h(0).measure_all().build();

        let pub_default = SamplerPub::new(circuit.clone());
        let pub_override = SamplerPub::with_shots(circuit.clone(), 500);

        let sampler = Sampler::new(SamplerConfig::default().with_shots(1024).with_seed(42));
        let results = sampler.run(&[pub_default, pub_override]);

        assert_eq!(results[0].shots, 1024);
        assert_eq!(
            results[0].counts.values().sum::<usize>(),
            1024
        );

        assert_eq!(results[1].shots, 500);
        assert_eq!(
            results[1].counts.values().sum::<usize>(),
            500
        );
    }

    // Test 14: Empty circuit returns |0...0>
    #[test]
    fn test_empty_circuit_returns_zero_state() {
        let circuit = CircuitBuilder::new(3).build();
        let state = simulate_circuit(&circuit);

        assert_eq!(state.len(), 8); // 2^3
        assert!((state[0].re - 1.0).abs() < 1e-10);
        assert!(state[0].im.abs() < 1e-10);
        for i in 1..8 {
            assert!(state[i].norm_sqr() < 1e-10);
        }
    }

    // Additional tests for completeness

    // Test 15: ZZ expectation for Bell state = 1
    #[test]
    fn test_estimator_zz_bell_state() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let obs = Observable::from_string("ZZ");
        let estimator = Estimator::new(EstimatorConfig::default());
        let (value, _) = estimator.run_single(&circuit, &obs);

        assert!(
            (value - 1.0).abs() < 1e-10,
            "Expected <ZZ> = 1 for Bell state, got {}",
            value
        );
    }

    // Test 16: Rotation gates produce correct state
    #[test]
    fn test_rotation_gates() {
        // Ry(pi)|0> = |1>
        let circuit = CircuitBuilder::new(1).ry(0, PI).build();
        let state = simulate_circuit(&circuit);
        assert!(state[0].norm_sqr() < 1e-10, "Expected |0> amplitude ~ 0");
        assert!(
            (state[1].norm_sqr() - 1.0).abs() < 1e-10,
            "Expected |1> amplitude ~ 1"
        );
    }

    // Test 17: measure_all with CircuitBuilder
    #[test]
    fn test_measure_all() {
        let circuit = CircuitBuilder::new(4).measure_all().build();
        assert_eq!(circuit.measurements, vec![0, 1, 2, 3]);
    }

    // Test 18: Sampler quasi-distribution sums to 1
    #[test]
    fn test_quasi_distribution_sums_to_one() {
        let circuit = CircuitBuilder::new(2).h(0).h(1).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));
        let result = sampler.run_single(&circuit, 1024);

        let total: f64 = result.quasi_distribution.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "Quasi-distribution should sum to 1, got {}",
            total
        );
    }

    // Test 19: Estimator batch execution
    #[test]
    fn test_estimator_batch() {
        let circuit_0 = CircuitBuilder::new(1).build();       // |0>
        let circuit_1 = CircuitBuilder::new(1).x(0).build();  // |1>

        let obs_z = Observable::z(0);

        let pubs = vec![
            EstimatorPub::new(circuit_0, vec![obs_z.clone()]),
            EstimatorPub::new(circuit_1, vec![obs_z]),
        ];

        let estimator = Estimator::new(EstimatorConfig::default());
        let results = estimator.run(&pubs);

        assert_eq!(results.len(), 2);
        assert!((results[0].values[0] - 1.0).abs() < 1e-10);  // <Z> = 1 for |0>
        assert!((results[1].values[0] + 1.0).abs() < 1e-10);  // <Z> = -1 for |1>
    }

    // ================================================================
    // V2 PRIMITIVES TESTS
    // ================================================================

    // V2 Test 1: SamplerV2 single PUB with BitArray result
    #[test]
    fn test_sampler_v2_single_pub() {
        let circuit = CircuitBuilder::new(1).x(0).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(512).with_seed(42));

        let pubs = vec![SamplerPubV2::new(circuit)];
        let results = sampler.run_v2(&pubs);

        assert_eq!(results.len(), 1); // One PUB
        assert_eq!(results[0].len(), 1); // Non-parametric: one result per PUB

        let result = &results[0][0];
        assert_eq!(result.data.num_shots(), 512);
        assert_eq!(result.data.num_bits(), 1);

        // X|0> = |1>, so every shot should be 1
        let counts = result.get_counts();
        assert_eq!(*counts.get(&1).unwrap_or(&0), 512);
        assert_eq!(*counts.get(&0).unwrap_or(&0), 0);

        // Metadata should be populated
        assert_eq!(result.metadata.num_qubits, 1);
        assert_eq!(result.metadata.shots_executed, 512);
        assert!(result.metadata.execution_time_secs >= 0.0);
        assert!(result.metadata.backend_name.contains("nqpu"));
    }

    // V2 Test 2: SamplerV2 multiple PUBs
    #[test]
    fn test_sampler_v2_multiple_pubs() {
        let circuit_h = CircuitBuilder::new(1).h(0).measure_all().build();
        let circuit_x = CircuitBuilder::new(1).x(0).measure_all().build();
        let circuit_id = CircuitBuilder::new(1).measure_all().build(); // |0>

        let sampler = Sampler::new(SamplerConfig::default().with_shots(1024).with_seed(42));
        let pubs = vec![
            SamplerPubV2::new(circuit_h),
            SamplerPubV2::new(circuit_x),
            SamplerPubV2::new(circuit_id),
        ];
        let results = sampler.run_v2(&pubs);

        assert_eq!(results.len(), 3);

        // PUB 0: H|0> = superposition
        let counts_h = results[0][0].get_counts();
        let total_h: usize = counts_h.values().sum();
        assert_eq!(total_h, 1024);
        assert!(counts_h.len() >= 2, "Hadamard should produce both 0 and 1");

        // PUB 1: X|0> = |1>
        let counts_x = results[1][0].get_counts();
        assert_eq!(*counts_x.get(&1).unwrap_or(&0), 1024);

        // PUB 2: |0>
        let counts_id = results[2][0].get_counts();
        assert_eq!(*counts_id.get(&0).unwrap_or(&0), 1024);
    }

    // V2 Test 3: SamplerV2 with parametric circuit (parameter binding)
    #[test]
    fn test_sampler_v2_parametric_circuit() {
        // Build a parametric circuit: Ry(theta)|0>
        // With theta=pi, should get |1> deterministically
        let circuit = CircuitBuilder::new(1)
            .ry_param(0, "theta")
            .measure_all()
            .build();

        assert!(circuit.is_parametric());
        assert_eq!(circuit.parameter_names, vec!["theta".to_string()]);

        let mut params = HashMap::new();
        params.insert("theta".to_string(), PI);

        let pubs = vec![SamplerPubV2::with_parameters_and_shots(
            circuit,
            vec![params],
            1024,
        )];

        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));
        let results = sampler.run_v2(&pubs);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1); // One parameter binding

        let counts = results[0][0].get_counts();
        assert_eq!(
            *counts.get(&1).unwrap_or(&0),
            1024,
            "Ry(pi)|0> should give |1> deterministically"
        );
    }

    // V2 Test 4: BitArray individual bit and bitstring access
    #[test]
    fn test_sampler_v2_bitarray_access() {
        // 2-qubit Bell state: should get only |00> and |11>
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(2048).with_seed(42));

        let pubs = vec![SamplerPubV2::new(circuit)];
        let results = sampler.run_v2(&pubs);

        let bit_array = &results[0][0].data;
        assert_eq!(bit_array.num_bits(), 2);
        assert_eq!(bit_array.num_shots(), 2048);

        // Verify each shot is either |00> (0) or |11> (3)
        for shot in 0..bit_array.num_shots() {
            let bs = bit_array.get_bitstring(shot);
            assert!(
                bs == 0b00 || bs == 0b11,
                "Bell state shot {} gave bitstring {}, expected 0 or 3",
                shot,
                bs
            );

            // For |00>: bit 0 and bit 1 should both be false
            // For |11>: bit 0 and bit 1 should both be true
            let bit0 = bit_array.get_bit(shot, 0);
            let bit1 = bit_array.get_bit(shot, 1);
            assert_eq!(bit0, bit1, "Bell state bits must be correlated");
        }
    }

    // V2 Test 5: SamplerV2 with classical registers
    #[test]
    fn test_sampler_v2_classical_registers() {
        // 3-qubit circuit: X on q0 and q2, identity on q1
        // Register "r0" maps to q0, q1 (should see "10")
        // Register "r1" maps to q2 (should see "1")
        let circuit = CircuitBuilder::new(3)
            .x(0)
            .x(2)
            .measure_all()
            .add_classical_register("r0", vec![0, 1])
            .add_classical_register("r1", vec![2])
            .build();

        let sampler = Sampler::new(SamplerConfig::default().with_shots(512).with_seed(42));
        let pubs = vec![SamplerPubV2::new(circuit)];
        let results = sampler.run_v2(&pubs);

        let result = &results[0][0];

        // Full result should be |101> = 5
        let full_counts = result.get_counts();
        assert_eq!(*full_counts.get(&0b101).unwrap_or(&0), 512);

        // Register r0 (q0, q1): should be |10> = 2
        let r0_counts = result.get_register_counts("r0").expect("r0 register missing");
        assert_eq!(
            *r0_counts.get(&0b10).unwrap_or(&0),
            512,
            "Register r0 (q0=1, q1=0) should be 0b10=2"
        );

        // Register r1 (q2): should be |1> = 1
        let r1_counts = result.get_register_counts("r1").expect("r1 register missing");
        assert_eq!(
            *r1_counts.get(&1).unwrap_or(&0),
            512,
            "Register r1 (q2=1) should be 1"
        );
    }

    // V2 Test 6: EstimatorV2 single observable
    #[test]
    fn test_estimator_v2_single_observable() {
        let circuit = CircuitBuilder::new(1).build(); // |0>
        let obs = Observable::z(0);

        let estimator = Estimator::new(EstimatorConfig::default());
        let pubs = vec![EstimatorPubV2::new(circuit, vec![obs])];
        let results = estimator.run_v2(&pubs);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 1);

        let result = &results[0][0];
        assert!((result.values[0] - 1.0).abs() < 1e-10, "Expected <Z>=1 for |0>");
        assert!(result.standard_errors[0].abs() < 1e-10, "Exact mode: zero error");
        assert!(result.metadata.extra.get("mode").unwrap() == "exact");
    }

    // V2 Test 7: EstimatorV2 multiple observables per circuit
    #[test]
    fn test_estimator_v2_multiple_observables() {
        // Bell state: <ZZ>=1, <XX>=1, <ZI>=0
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build();
        let obs_zz = Observable::from_string("ZZ");
        let obs_xx = Observable::xx(0, 1);
        let obs_zi = Observable::z(0); // <Z_0> for Bell state = 0

        let estimator = Estimator::new(EstimatorConfig::default());
        let pubs = vec![EstimatorPubV2::new(
            circuit,
            vec![obs_zz, obs_xx, obs_zi],
        )];
        let results = estimator.run_v2(&pubs);

        let result = &results[0][0];
        assert_eq!(result.values.len(), 3);
        assert_eq!(result.standard_errors.len(), 3);

        assert!(
            (result.values[0] - 1.0).abs() < 1e-10,
            "Expected <ZZ>=1, got {}",
            result.values[0]
        );
        assert!(
            (result.values[1] - 1.0).abs() < 1e-10,
            "Expected <XX>=1, got {}",
            result.values[1]
        );
        assert!(
            result.values[2].abs() < 1e-10,
            "Expected <Z_0>=0 for Bell, got {}",
            result.values[2]
        );
    }

    // V2 Test 8: EstimatorV2 precision targeting with adaptive shots
    #[test]
    fn test_estimator_v2_precision_targeting() {
        let circuit = CircuitBuilder::new(1).h(0).build(); // |+>
        let obs = Observable::z(0); // <Z>=0 for |+>

        // Request high precision (0.01) with shot-based estimation
        let estimator = Estimator::new(
            EstimatorConfig::default()
                .with_shots(100)
                .with_seed(42),
        );
        let pubs = vec![EstimatorPubV2::with_precision(
            circuit,
            vec![obs],
            0.01, // Target: std_error <= 0.01 => needs 10000 shots
        )];
        let results = estimator.run_v2(&pubs);

        let result = &results[0][0];
        // With precision targeting, the estimator should have used enough shots
        // that the standard error is <= 0.01
        assert!(
            result.standard_errors[0] <= 0.011, // Slight tolerance
            "Expected std_error <= 0.01, got {}",
            result.standard_errors[0]
        );
        // Result should be close to 0
        assert!(
            result.values[0].abs() < 0.1,
            "Expected <Z>~0 for |+>, got {}",
            result.values[0]
        );

        // Verify metadata reports the precision target
        assert!(result.metadata.extra.contains_key("target_precision"));
    }

    // V2 Test 9: EstimatorV2 with Pauli string input via SparsePauliOp
    #[test]
    fn test_estimator_v2_pauli_string() {
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build(); // Bell

        // Create observable from SparsePauliOp
        let sparse_op = SparsePauliOp::new(vec![
            ("ZZ".to_string(), 0.5),
            ("XX".to_string(), 0.5),
        ]);
        let obs = sparse_op.to_observable();

        let estimator = Estimator::new(EstimatorConfig::default());
        let pubs = vec![EstimatorPubV2::new(circuit, vec![obs])];
        let results = estimator.run_v2(&pubs);

        let result = &results[0][0];
        // 0.5*<ZZ> + 0.5*<XX> = 0.5*1 + 0.5*1 = 1.0 for Bell state
        assert!(
            (result.values[0] - 1.0).abs() < 1e-10,
            "Expected 0.5*<ZZ> + 0.5*<XX> = 1.0, got {}",
            result.values[0]
        );
    }

    // V2 Test 10: EstimatorV2 with parametric circuit
    #[test]
    fn test_estimator_v2_parametric() {
        // Ry(theta)|0>: <Z> = cos(theta)
        let circuit = CircuitBuilder::new(1)
            .ry_param(0, "theta")
            .build();

        let obs = Observable::z(0);

        // Sweep: theta=0 -> <Z>=1, theta=pi -> <Z>=-1, theta=pi/2 -> <Z>=0
        let params_list = vec![
            {
                let mut m = HashMap::new();
                m.insert("theta".to_string(), 0.0);
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("theta".to_string(), PI);
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("theta".to_string(), PI / 2.0);
                m
            },
        ];

        let estimator = Estimator::new(EstimatorConfig::default());
        let pubs = vec![EstimatorPubV2 {
            circuit,
            observables: vec![obs],
            parameter_values: params_list,
            precision: None,
        }];
        let results = estimator.run_v2(&pubs);

        assert_eq!(results.len(), 1); // One PUB
        assert_eq!(results[0].len(), 3); // Three parameter bindings

        // theta=0: <Z>=cos(0)=1
        assert!(
            (results[0][0].values[0] - 1.0).abs() < 1e-10,
            "theta=0: expected <Z>=1, got {}",
            results[0][0].values[0]
        );
        // theta=pi: <Z>=cos(pi)=-1
        assert!(
            (results[0][1].values[0] - (-1.0)).abs() < 1e-10,
            "theta=pi: expected <Z>=-1, got {}",
            results[0][1].values[0]
        );
        // theta=pi/2: <Z>=cos(pi/2)=0
        assert!(
            results[0][2].values[0].abs() < 1e-10,
            "theta=pi/2: expected <Z>=0, got {}",
            results[0][2].values[0]
        );
    }

    // V2 Test 11: PUB construction and field validation
    #[test]
    fn test_pub_construction() {
        // SamplerPubV2 constructors
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();

        let pub_basic = SamplerPubV2::new(circuit.clone());
        assert!(pub_basic.parameter_values.is_empty());
        assert!(pub_basic.shots.is_none());

        let pub_shots = SamplerPubV2::with_shots(circuit.clone(), 2048);
        assert_eq!(pub_shots.shots, Some(2048));

        let mut params = HashMap::new();
        params.insert("theta".to_string(), 1.5);
        let pub_params = SamplerPubV2::with_parameters(circuit.clone(), vec![params.clone()]);
        assert_eq!(pub_params.parameter_values.len(), 1);
        assert!(pub_params.shots.is_none());

        let pub_full =
            SamplerPubV2::with_parameters_and_shots(circuit.clone(), vec![params], 4096);
        assert_eq!(pub_full.parameter_values.len(), 1);
        assert_eq!(pub_full.shots, Some(4096));

        // EstimatorPubV2 constructors
        let obs = Observable::z(0);
        let epub_basic = EstimatorPubV2::new(circuit.clone(), vec![obs.clone()]);
        assert!(epub_basic.parameter_values.is_empty());
        assert!(epub_basic.precision.is_none());

        let epub_prec =
            EstimatorPubV2::with_precision(circuit.clone(), vec![obs.clone()], 0.001);
        assert_eq!(epub_prec.precision, Some(0.001));

        // from_inputs with observable coercion
        let epub_coerced = EstimatorPubV2::from_inputs(
            circuit,
            vec![
                ObservableInput::PauliString("ZZ".to_string()),
                ObservableInput::Sparse(SparsePauliOp::from_label("XX")),
                ObservableInput::Observable(obs),
            ],
        );
        assert_eq!(epub_coerced.observables.len(), 3);
    }

    // V2 Test 12: BitArray operations (construction, access, slicing)
    #[test]
    fn test_bitarray_operations() {
        // 3-bit bitstrings: [0b101, 0b011, 0b000, 0b111, 0b101]
        let bitstrings = vec![0b101, 0b011, 0b000, 0b111, 0b101];
        let ba = BitArray::from_bitstrings(3, &bitstrings);

        assert_eq!(ba.num_bits(), 3);
        assert_eq!(ba.num_shots(), 5);

        // Verify bitstring retrieval
        assert_eq!(ba.get_bitstring(0), 0b101);
        assert_eq!(ba.get_bitstring(1), 0b011);
        assert_eq!(ba.get_bitstring(2), 0b000);
        assert_eq!(ba.get_bitstring(3), 0b111);
        assert_eq!(ba.get_bitstring(4), 0b101);

        // Verify individual bit access
        // Shot 0: 101 -> bit0=1, bit1=0, bit2=1
        assert!(ba.get_bit(0, 0));
        assert!(!ba.get_bit(0, 1));
        assert!(ba.get_bit(0, 2));

        // Shot 1: 011 -> bit0=0, bit1=1, bit2=1
        assert!(!ba.get_bit(1, 0));
        assert!(ba.get_bit(1, 1));
        assert!(ba.get_bit(1, 2));

        // Verify counts
        let counts = ba.get_counts();
        assert_eq!(*counts.get(&0b101).unwrap_or(&0), 2);
        assert_eq!(*counts.get(&0b011).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&0b000).unwrap_or(&0), 1);
        assert_eq!(*counts.get(&0b111).unwrap_or(&0), 1);

        // Verify register slicing: extract bits [0, 2] (first and third)
        let sliced = ba.slice_register(&[0, 2]);
        assert_eq!(sliced.num_bits(), 2);
        assert_eq!(sliced.num_shots(), 5);
        // Shot 0: bits 0,2 of 101 -> 1,1 -> 0b11=3
        assert_eq!(sliced.get_bitstring(0), 0b11);
        // Shot 1: bits 0,2 of 011 -> 0,1 -> 0b01=1
        assert_eq!(sliced.get_bitstring(1), 0b01);
        // Shot 2: bits 0,2 of 000 -> 0,0 -> 0b00=0
        assert_eq!(sliced.get_bitstring(2), 0b00);
    }

    // V2 Test 13: V1-V2 backward compatibility conversion
    #[test]
    fn test_v2_backward_compatible() {
        // Run the same circuit through V1 and V2, compare results
        let circuit = CircuitBuilder::new(1).x(0).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(1024).with_seed(42));

        // V1 path
        let v1_result = sampler.run_single(&circuit, 1024);

        // V2 path -> convert to V1
        let v2_pubs = vec![SamplerPubV2::with_shots(circuit.clone(), 1024)];

        // Need a fresh sampler with same seed for deterministic comparison
        let sampler2 = Sampler::new(SamplerConfig::default().with_shots(1024).with_seed(42));
        let v2_results = sampler2.run_v2(&v2_pubs);
        let v2_as_v1 = v2_results[0][0].to_v1();

        // Both should have 1024 shots of |1>
        assert_eq!(v1_result.shots, v2_as_v1.shots);
        assert_eq!(
            *v1_result.counts.get(&1).unwrap_or(&0),
            *v2_as_v1.counts.get(&1).unwrap_or(&0),
        );

        // V1 PUB -> V2 PUB conversion
        let v1_pub = SamplerPub::with_shots(circuit.clone(), 512);
        let v2_pub = v1_pub.to_v2();
        assert_eq!(v2_pub.shots, Some(512));
        assert!(v2_pub.parameter_values.is_empty());

        // V1 EstimatorPub -> V2 conversion
        let obs = Observable::z(0);
        let v1_epub = EstimatorPub::with_precision(circuit, vec![obs], 0.005);
        let v2_epub = v1_epub.to_v2();
        assert_eq!(v2_epub.precision, Some(0.005));
        assert_eq!(v2_epub.observables.len(), 1);
    }

    // V2 Test 14: EstimatorV2 standard error validation
    #[test]
    fn test_estimator_v2_standard_error() {
        let circuit = CircuitBuilder::new(1).h(0).build(); // |+>
        let obs = Observable::z(0); // <Z>=0

        // Exact mode: zero standard error
        let exact_est = Estimator::new(EstimatorConfig::default());
        let pubs_exact = vec![EstimatorPubV2::new(circuit.clone(), vec![obs.clone()])];
        let results_exact = exact_est.run_v2(&pubs_exact);
        assert!(
            results_exact[0][0].standard_errors[0].abs() < 1e-15,
            "Exact mode must have zero standard error"
        );

        // Shot-based: standard error = 1/sqrt(N)
        let shot_est = Estimator::new(
            EstimatorConfig::default()
                .with_shots(10000)
                .with_seed(42),
        );
        let pubs_shot = vec![EstimatorPubV2::new(circuit, vec![obs])];
        let results_shot = shot_est.run_v2(&pubs_shot);
        let expected_se = 1.0 / (10000.0_f64).sqrt(); // 0.01
        assert!(
            (results_shot[0][0].standard_errors[0] - expected_se).abs() < 0.001,
            "Shot-based std_error should be ~{}, got {}",
            expected_se,
            results_shot[0][0].standard_errors[0]
        );

        // meets_precision check
        assert!(results_shot[0][0].meets_precision(0.02));
        assert!(!results_shot[0][0].meets_precision(0.005));
    }

    // V2 Test 15: SamplerV2 shot distribution validation
    #[test]
    fn test_sampler_v2_shots_distribution() {
        // H on qubit 0: should be roughly 50/50
        let circuit = CircuitBuilder::new(1).h(0).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));

        let pubs = vec![SamplerPubV2::with_shots(circuit, 10000)];
        let results = sampler.run_v2(&pubs);

        let result = &results[0][0];
        let counts = result.get_counts();
        let total: usize = counts.values().sum();
        assert_eq!(total, 10000);

        let count_0 = *counts.get(&0).unwrap_or(&0);
        let count_1 = *counts.get(&1).unwrap_or(&0);

        let ratio = count_0 as f64 / 10000.0;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "Expected ~50% |0> for H gate, got {:.1}%",
            ratio * 100.0
        );
        assert_eq!(count_0 + count_1, 10000);
    }

    // V2 Test 16: SparsePauliOp construction and conversion
    #[test]
    fn test_sparse_pauli_op() {
        // Single-label constructor
        let single = SparsePauliOp::from_label("XYZ");
        assert_eq!(single.terms.len(), 1);
        assert_eq!(single.terms[0].0, "XYZ");
        assert!((single.terms[0].1 - 1.0).abs() < 1e-15);

        // Multi-term constructor
        let multi = SparsePauliOp::new(vec![
            ("ZZ".to_string(), 0.4),
            ("XX".to_string(), 0.3),
            ("YY".to_string(), 0.2),
            ("II".to_string(), 0.1),
        ]);
        assert_eq!(multi.terms.len(), 4);

        // Conversion to Observable and evaluation
        let obs = multi.to_observable();
        assert_eq!(obs.terms.len(), 4);

        // The II term should have no non-identity paulis
        let ii_term = &obs.terms[3]; // Last term was "II"
        assert!(ii_term.paulis.is_empty(), "II should have no active paulis");
        assert!((ii_term.coefficient - 0.1).abs() < 1e-15);

        // Use the observable in an estimator for a concrete validation
        let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).build(); // Bell
        let estimator = Estimator::new(EstimatorConfig::default());
        let pubs = vec![EstimatorPubV2::new(circuit, vec![obs])];
        let results = estimator.run_v2(&pubs);
        // For Bell state: <ZZ>=1, <XX>=1, <YY>=-1, <II>=1
        // Observable = 0.4*1 + 0.3*1 + 0.2*(-1) + 0.1*1 = 0.4+0.3-0.2+0.1 = 0.6
        let expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.2 * (-1.0) + 0.1 * 1.0;
        assert!(
            (results[0][0].values[0] - expected).abs() < 1e-10,
            "SparsePauliOp: expected {}, got {}",
            expected,
            results[0][0].values[0]
        );
    }

    // V2 Test 17: Parameter binding with multiple parameters in one circuit
    #[test]
    fn test_parameter_binding_multi_param() {
        // Two parametric rotations on the same qubit
        let circuit = CircuitBuilder::new(1)
            .rx_param(0, "alpha")
            .rz_param(0, "beta")
            .build();

        assert!(circuit.is_parametric());
        assert_eq!(circuit.parameter_names.len(), 2);
        assert!(circuit.parameter_names.contains(&"alpha".to_string()));
        assert!(circuit.parameter_names.contains(&"beta".to_string()));

        // Bind both parameters
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), PI / 2.0);
        params.insert("beta".to_string(), 0.0);

        let bound = circuit.bind_parameters(&params).unwrap();
        assert!(!bound.is_parametric());
        assert!(bound.parameter_names.is_empty());

        // Missing parameter should error
        let mut incomplete = HashMap::new();
        incomplete.insert("alpha".to_string(), 1.0);
        let err = circuit.bind_parameters(&incomplete);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("beta"));
    }

    // V2 Test 18: SamplerV2 parameter sweep (multiple bindings per PUB)
    #[test]
    fn test_sampler_v2_parameter_sweep() {
        // Ry(theta)|0>: sweep theta from 0 to pi
        let circuit = CircuitBuilder::new(1)
            .ry_param(0, "theta")
            .measure_all()
            .build();

        let param_sets: Vec<HashMap<String, f64>> = vec![
            { let mut m = HashMap::new(); m.insert("theta".to_string(), 0.0); m },
            { let mut m = HashMap::new(); m.insert("theta".to_string(), PI); m },
        ];

        let pubs = vec![SamplerPubV2::with_parameters_and_shots(
            circuit,
            param_sets,
            2048,
        )];

        let sampler = Sampler::new(SamplerConfig::default().with_seed(42));
        let results = sampler.run_v2(&pubs);

        assert_eq!(results.len(), 1); // One PUB
        assert_eq!(results[0].len(), 2); // Two parameter bindings

        // theta=0: Ry(0)|0> = |0>
        let counts_0 = results[0][0].get_counts();
        assert_eq!(*counts_0.get(&0).unwrap_or(&0), 2048);

        // theta=pi: Ry(pi)|0> = |1>
        let counts_pi = results[0][1].get_counts();
        assert_eq!(*counts_pi.get(&1).unwrap_or(&0), 2048);
    }

    // V2 Test 19: Execution metadata populated correctly
    #[test]
    fn test_execution_metadata() {
        let circuit = CircuitBuilder::new(3).h(0).cx(0, 1).cx(1, 2).measure_all().build();
        let sampler = Sampler::new(SamplerConfig::default().with_shots(512).with_seed(42));

        let pubs = vec![SamplerPubV2::new(circuit)];
        let results = sampler.run_v2(&pubs);

        let meta = &results[0][0].metadata;
        assert_eq!(meta.num_qubits, 3);
        assert_eq!(meta.shots_executed, 512);
        assert!(meta.execution_time_secs >= 0.0);
        assert!(meta.backend_name.contains("nqpu"));
        assert!(!meta.backend_version.is_empty());

        // to_map conversion
        let map = meta.to_map();
        assert!(map.contains_key("backend_name"));
        assert!(map.contains_key("num_qubits"));
        assert!(map.contains_key("shots"));
        assert!(map.contains_key("execution_time_secs"));
    }

    // V2 Test 20: ObservableInput coercion from all types
    #[test]
    fn test_observable_coercion() {
        // From Pauli string
        let obs1 = ObservableInput::from("ZIZ").into_observable();
        assert_eq!(obs1.terms.len(), 1);
        assert_eq!(obs1.terms[0].paulis.len(), 2); // I filtered out

        // From SparsePauliOp
        let sp = SparsePauliOp::new(vec![("XY".to_string(), 2.5)]);
        let obs2 = ObservableInput::from(sp).into_observable();
        assert_eq!(obs2.terms.len(), 1);
        assert!((obs2.terms[0].coefficient - 2.5).abs() < 1e-15);

        // From Observable directly
        let raw = Observable::z(0);
        let obs3 = ObservableInput::from(raw).into_observable();
        assert_eq!(obs3.terms.len(), 1);
    }
}
