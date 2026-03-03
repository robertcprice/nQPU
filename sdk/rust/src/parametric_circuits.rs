//! Parametric Circuits for Variational Quantum Algorithms
//!
//! This module provides circuit templates with symbolic parameters that can be
//! bound to concrete values at execution time. Essential for VQE, QAOA, and all
//! variational algorithms.
//!
//! # Architecture
//!
//! - [`Parameter`]: Named symbolic parameter with optional bounds and initial value
//! - [`ParameterExpression`]: Arithmetic expression tree over parameters
//! - [`ParametricGate`]: Gate with parameter expressions for angles
//! - [`ParametricCircuit`]: Ordered list of parametric gates with a parameter registry
//! - [`BoundCircuit`]: Fully resolved circuit with all parameters replaced by f64 values
//! - [`GradientCircuit`]: Shifted circuits for parameter-shift rule gradient computation
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::parametric_circuits::*;
//!
//! let mut circuit = ParametricCircuit::new();
//! let theta = circuit.add_parameter(Parameter::new("theta")).unwrap();
//! circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));
//!
//! let binding = ParameterBinding::from([("theta".to_string(), 1.57)]);
//! let bound = circuit.bind(&binding).unwrap();
//! assert!((bound.gates[0].angle - 1.57).abs() < 1e-12);
//! ```

use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during parametric circuit operations.
#[derive(Clone, Debug, PartialEq)]
pub enum ParametricError {
    /// A parameter referenced in an expression has no bound value.
    UnboundParameter(String),
    /// Attempted to register a parameter with a name that already exists.
    DuplicateParameter(String),
    /// A bound value falls outside the parameter's declared bounds.
    OutOfBounds {
        name: String,
        value: f64,
        min: f64,
        max: f64,
    },
    /// An error occurred while evaluating a parameter expression.
    ExpressionError(String),
}

impl fmt::Display for ParametricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParametricError::UnboundParameter(name) => {
                write!(f, "parameter '{}' has no bound value", name)
            }
            ParametricError::DuplicateParameter(name) => {
                write!(f, "parameter '{}' is already registered", name)
            }
            ParametricError::OutOfBounds {
                name,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "value {} for parameter '{}' is outside bounds [{}, {}]",
                    value, name, min, max
                )
            }
            ParametricError::ExpressionError(msg) => {
                write!(f, "expression evaluation error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ParametricError {}

/// Result type alias for parametric circuit operations.
pub type Result<T> = std::result::Result<T, ParametricError>;

// ============================================================
// PARAMETER
// ============================================================

/// A named symbolic parameter with optional bounds and initial value.
///
/// Parameters represent free variables in a parametric circuit. They carry
/// metadata (bounds, initial value) that optimizers can use to constrain
/// the search space.
#[derive(Clone, Debug)]
pub struct Parameter {
    /// Unique name identifying this parameter.
    pub name: String,
    /// Lower bound for the parameter value (defaults to -infinity).
    pub min: f64,
    /// Upper bound for the parameter value (defaults to +infinity).
    pub max: f64,
    /// Initial value hint for optimizers (defaults to 0.0).
    pub initial_value: f64,
}

impl Parameter {
    /// Create a parameter with the given name and default bounds/initial value.
    pub fn new(name: &str) -> Self {
        Parameter {
            name: name.to_string(),
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
            initial_value: 0.0,
        }
    }

    /// Create a parameter with explicit bounds.
    pub fn with_bounds(name: &str, min: f64, max: f64) -> Self {
        Parameter {
            name: name.to_string(),
            min,
            max,
            initial_value: 0.0,
        }
    }

    /// Create a parameter with bounds and an initial value.
    pub fn with_bounds_and_initial(name: &str, min: f64, max: f64, initial_value: f64) -> Self {
        Parameter {
            name: name.to_string(),
            min,
            max,
            initial_value,
        }
    }

    /// Check whether a value is within this parameter's bounds.
    pub fn check_bounds(&self, value: f64) -> Result<()> {
        if value < self.min || value > self.max {
            Err(ParametricError::OutOfBounds {
                name: self.name.clone(),
                value,
                min: self.min,
                max: self.max,
            })
        } else {
            Ok(())
        }
    }
}

impl PartialEq for Parameter {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

// ============================================================
// PARAMETER EXPRESSION
// ============================================================

/// An arithmetic expression tree over symbolic parameters.
///
/// Supports literals, parameter references, and arithmetic/trigonometric
/// operations. Expressions are evaluated by supplying a binding that maps
/// parameter names to concrete f64 values.
#[derive(Clone, Debug)]
pub enum ParameterExpression {
    /// A literal floating-point constant.
    Literal(f64),
    /// A reference to a named parameter.
    Param(String),
    /// Sum of two sub-expressions.
    Add(Box<ParameterExpression>, Box<ParameterExpression>),
    /// Product of two sub-expressions.
    Mul(Box<ParameterExpression>, Box<ParameterExpression>),
    /// Negation of a sub-expression.
    Neg(Box<ParameterExpression>),
    /// Sine of a sub-expression.
    Sin(Box<ParameterExpression>),
    /// Cosine of a sub-expression.
    Cos(Box<ParameterExpression>),
}

impl ParameterExpression {
    /// Shorthand constructor for a literal value.
    pub fn literal(value: f64) -> Self {
        ParameterExpression::Literal(value)
    }

    /// Shorthand constructor for a parameter reference.
    pub fn param(name: &str) -> Self {
        ParameterExpression::Param(name.to_string())
    }

    /// Shorthand constructor for addition.
    pub fn add(lhs: ParameterExpression, rhs: ParameterExpression) -> Self {
        ParameterExpression::Add(Box::new(lhs), Box::new(rhs))
    }

    /// Shorthand constructor for multiplication.
    pub fn mul(lhs: ParameterExpression, rhs: ParameterExpression) -> Self {
        ParameterExpression::Mul(Box::new(lhs), Box::new(rhs))
    }

    /// Shorthand constructor for negation.
    pub fn neg(expr: ParameterExpression) -> Self {
        ParameterExpression::Neg(Box::new(expr))
    }

    /// Shorthand constructor for sine.
    pub fn sin(expr: ParameterExpression) -> Self {
        ParameterExpression::Sin(Box::new(expr))
    }

    /// Shorthand constructor for cosine.
    pub fn cos(expr: ParameterExpression) -> Self {
        ParameterExpression::Cos(Box::new(expr))
    }

    /// Evaluate this expression given a parameter binding.
    ///
    /// Returns `Err(UnboundParameter)` if any referenced parameter is missing
    /// from the binding.
    pub fn evaluate(&self, binding: &ParameterBinding) -> Result<f64> {
        match self {
            ParameterExpression::Literal(v) => Ok(*v),
            ParameterExpression::Param(name) => binding
                .get(name)
                .copied()
                .ok_or_else(|| ParametricError::UnboundParameter(name.clone())),
            ParameterExpression::Add(a, b) => {
                Ok(a.evaluate(binding)? + b.evaluate(binding)?)
            }
            ParameterExpression::Mul(a, b) => {
                Ok(a.evaluate(binding)? * b.evaluate(binding)?)
            }
            ParameterExpression::Neg(a) => Ok(-a.evaluate(binding)?),
            ParameterExpression::Sin(a) => Ok(a.evaluate(binding)?.sin()),
            ParameterExpression::Cos(a) => Ok(a.evaluate(binding)?.cos()),
        }
    }

    /// Collect all parameter names referenced by this expression.
    pub fn parameters(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        self.collect_parameters(&mut out);
        out
    }

    fn collect_parameters(&self, out: &mut HashSet<String>) {
        match self {
            ParameterExpression::Literal(_) => {}
            ParameterExpression::Param(name) => {
                out.insert(name.clone());
            }
            ParameterExpression::Add(a, b) | ParameterExpression::Mul(a, b) => {
                a.collect_parameters(out);
                b.collect_parameters(out);
            }
            ParameterExpression::Neg(a)
            | ParameterExpression::Sin(a)
            | ParameterExpression::Cos(a) => {
                a.collect_parameters(out);
            }
        }
    }

    /// Attempt partial evaluation: substitute bound parameters, leave unbound
    /// ones as symbolic `Param` nodes.
    pub fn partial_evaluate(&self, binding: &ParameterBinding) -> ParameterExpression {
        match self {
            ParameterExpression::Literal(v) => ParameterExpression::Literal(*v),
            ParameterExpression::Param(name) => {
                if let Some(&val) = binding.get(name) {
                    ParameterExpression::Literal(val)
                } else {
                    ParameterExpression::Param(name.clone())
                }
            }
            ParameterExpression::Add(a, b) => {
                let ea = a.partial_evaluate(binding);
                let eb = b.partial_evaluate(binding);
                // Fold if both sides are now literal
                if let (ParameterExpression::Literal(va), ParameterExpression::Literal(vb)) =
                    (&ea, &eb)
                {
                    ParameterExpression::Literal(va + vb)
                } else {
                    ParameterExpression::Add(Box::new(ea), Box::new(eb))
                }
            }
            ParameterExpression::Mul(a, b) => {
                let ea = a.partial_evaluate(binding);
                let eb = b.partial_evaluate(binding);
                if let (ParameterExpression::Literal(va), ParameterExpression::Literal(vb)) =
                    (&ea, &eb)
                {
                    ParameterExpression::Literal(va * vb)
                } else {
                    ParameterExpression::Mul(Box::new(ea), Box::new(eb))
                }
            }
            ParameterExpression::Neg(a) => {
                let ea = a.partial_evaluate(binding);
                if let ParameterExpression::Literal(va) = &ea {
                    ParameterExpression::Literal(-va)
                } else {
                    ParameterExpression::Neg(Box::new(ea))
                }
            }
            ParameterExpression::Sin(a) => {
                let ea = a.partial_evaluate(binding);
                if let ParameterExpression::Literal(va) = &ea {
                    ParameterExpression::Literal(va.sin())
                } else {
                    ParameterExpression::Sin(Box::new(ea))
                }
            }
            ParameterExpression::Cos(a) => {
                let ea = a.partial_evaluate(binding);
                if let ParameterExpression::Literal(va) = &ea {
                    ParameterExpression::Literal(va.cos())
                } else {
                    ParameterExpression::Cos(Box::new(ea))
                }
            }
        }
    }

    /// Returns true if this expression contains no symbolic parameters.
    pub fn is_fully_bound(&self) -> bool {
        self.parameters().is_empty()
    }
}

impl fmt::Display for ParameterExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParameterExpression::Literal(v) => write!(f, "{}", v),
            ParameterExpression::Param(name) => write!(f, "{}", name),
            ParameterExpression::Add(a, b) => write!(f, "({} + {})", a, b),
            ParameterExpression::Mul(a, b) => write!(f, "({} * {})", a, b),
            ParameterExpression::Neg(a) => write!(f, "(-{})", a),
            ParameterExpression::Sin(a) => write!(f, "sin({})", a),
            ParameterExpression::Cos(a) => write!(f, "cos({})", a),
        }
    }
}

// ============================================================
// PARAMETER BINDING
// ============================================================

/// A mapping from parameter names to concrete f64 values.
pub type ParameterBinding = HashMap<String, f64>;

// ============================================================
// GATE TYPES
// ============================================================

/// Identifies the kind of gate in a parametric circuit.
#[derive(Clone, Debug, PartialEq)]
pub enum ParametricGateKind {
    /// Rotation around Z axis by an angle.
    Rz,
    /// Rotation around Y axis by an angle.
    Ry,
    /// Rotation around X axis by an angle.
    Rx,
    /// Hadamard gate (no parameter).
    H,
    /// CNOT gate (no parameter).
    Cx,
    /// CZ gate (no parameter).
    Cz,
    /// General single-qubit rotation U(theta, phi, lambda).
    U3,
}

/// A gate within a parametric circuit, carrying parameter expressions for angles.
///
/// Fixed gates (H, CX, CZ) have empty `angles`. Rotation gates carry one
/// expression per angle. U3 carries three.
#[derive(Clone, Debug)]
pub struct ParametricGate {
    /// The kind of gate.
    pub kind: ParametricGateKind,
    /// Target qubit index (or control qubit for two-qubit gates).
    pub qubit: usize,
    /// Second qubit for two-qubit gates (None for single-qubit gates).
    pub qubit2: Option<usize>,
    /// Parameter expressions for the gate's angles.
    pub angles: Vec<ParameterExpression>,
}

impl ParametricGate {
    // -- Rotation gate constructors --

    /// Rz(angle_expr) on the given qubit.
    pub fn rz(qubit: usize, angle: ParameterExpression) -> Self {
        ParametricGate {
            kind: ParametricGateKind::Rz,
            qubit,
            qubit2: None,
            angles: vec![angle],
        }
    }

    /// Ry(angle_expr) on the given qubit.
    pub fn ry(qubit: usize, angle: ParameterExpression) -> Self {
        ParametricGate {
            kind: ParametricGateKind::Ry,
            qubit,
            qubit2: None,
            angles: vec![angle],
        }
    }

    /// Rx(angle_expr) on the given qubit.
    pub fn rx(qubit: usize, angle: ParameterExpression) -> Self {
        ParametricGate {
            kind: ParametricGateKind::Rx,
            qubit,
            qubit2: None,
            angles: vec![angle],
        }
    }

    /// Hadamard on the given qubit (no parameters).
    pub fn h(qubit: usize) -> Self {
        ParametricGate {
            kind: ParametricGateKind::H,
            qubit,
            qubit2: None,
            angles: vec![],
        }
    }

    /// CNOT gate: control = qubit, target = qubit2.
    pub fn cx(control: usize, target: usize) -> Self {
        ParametricGate {
            kind: ParametricGateKind::Cx,
            qubit: control,
            qubit2: Some(target),
            angles: vec![],
        }
    }

    /// CZ gate: qubit and qubit2.
    pub fn cz(qubit1: usize, qubit2: usize) -> Self {
        ParametricGate {
            kind: ParametricGateKind::Cz,
            qubit: qubit1,
            qubit2: Some(qubit2),
            angles: vec![],
        }
    }

    /// U3(theta, phi, lambda) on the given qubit.
    pub fn u3(
        qubit: usize,
        theta: ParameterExpression,
        phi: ParameterExpression,
        lambda: ParameterExpression,
    ) -> Self {
        ParametricGate {
            kind: ParametricGateKind::U3,
            qubit,
            qubit2: None,
            angles: vec![theta, phi, lambda],
        }
    }

    /// Collect all parameter names referenced by this gate's angle expressions.
    pub fn parameters(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        for expr in &self.angles {
            out.extend(expr.parameters());
        }
        out
    }
}

// ============================================================
// BOUND GATE / BOUND CIRCUIT
// ============================================================

/// A gate with all parameter expressions resolved to concrete f64 angles.
#[derive(Clone, Debug)]
pub struct BoundGate {
    /// The kind of gate.
    pub kind: ParametricGateKind,
    /// Target qubit (or control for two-qubit gates).
    pub qubit: usize,
    /// Second qubit for two-qubit gates.
    pub qubit2: Option<usize>,
    /// Resolved angle value. For single-angle gates this is the first element.
    /// For U3 there are three. For fixed gates this is 0.0 (unused).
    pub angle: f64,
    /// All resolved angle values (for U3 or future multi-angle gates).
    pub angles: Vec<f64>,
}

/// A fully resolved circuit: all parameters have been replaced by f64 values.
///
/// Ready for execution on any quantum backend.
#[derive(Clone, Debug)]
pub struct BoundCircuit {
    /// The resolved gates in order.
    pub gates: Vec<BoundGate>,
    /// Number of qubits the circuit operates on.
    pub num_qubits: usize,
}

// ============================================================
// PARAMETRIC CIRCUIT
// ============================================================

/// A circuit template with symbolic parameters.
///
/// Define a circuit once with named parameters, then bind to different
/// concrete values to produce a [`BoundCircuit`] each time.
#[derive(Clone, Debug)]
pub struct ParametricCircuit {
    /// Ordered list of gates in the circuit.
    pub gates: Vec<ParametricGate>,
    /// Registry of declared parameters, keyed by name.
    pub parameters: HashMap<String, Parameter>,
    /// Number of qubits the circuit operates on.
    pub num_qubits: usize,
}

impl ParametricCircuit {
    /// Create an empty parametric circuit.
    pub fn new() -> Self {
        ParametricCircuit {
            gates: Vec::new(),
            parameters: HashMap::new(),
            num_qubits: 0,
        }
    }

    /// Create a parametric circuit with a known number of qubits.
    pub fn with_num_qubits(num_qubits: usize) -> Self {
        ParametricCircuit {
            gates: Vec::new(),
            parameters: HashMap::new(),
            num_qubits,
        }
    }

    /// Register a parameter. Returns a reference to the stored parameter.
    ///
    /// Fails with `DuplicateParameter` if a parameter with the same name
    /// already exists.
    pub fn add_parameter(&mut self, param: Parameter) -> Result<&Parameter> {
        if self.parameters.contains_key(&param.name) {
            return Err(ParametricError::DuplicateParameter(param.name.clone()));
        }
        let name = param.name.clone();
        self.parameters.insert(name.clone(), param);
        Ok(self.parameters.get(&name).unwrap())
    }

    /// Append a gate to the circuit. Automatically tracks num_qubits.
    pub fn add_gate(&mut self, gate: ParametricGate) {
        let max_qubit = if let Some(q2) = gate.qubit2 {
            gate.qubit.max(q2)
        } else {
            gate.qubit
        };
        if max_qubit + 1 > self.num_qubits {
            self.num_qubits = max_qubit + 1;
        }
        self.gates.push(gate);
    }

    /// Return the set of all parameter names referenced by gates in this circuit.
    pub fn referenced_parameters(&self) -> HashSet<String> {
        let mut out = HashSet::new();
        for gate in &self.gates {
            out.extend(gate.parameters());
        }
        out
    }

    /// Return the names of all registered parameters.
    pub fn parameter_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.parameters.keys().cloned().collect();
        names.sort();
        names
    }

    /// Bind all parameters to produce a [`BoundCircuit`].
    ///
    /// Every parameter referenced by the circuit's gates must have a value
    /// in the binding. Values are checked against parameter bounds.
    pub fn bind(&self, binding: &ParameterBinding) -> Result<BoundCircuit> {
        // Validate bounds for all registered parameters that have a binding
        for (name, param) in &self.parameters {
            if let Some(&value) = binding.get(name) {
                param.check_bounds(value)?;
            }
        }

        let mut bound_gates = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let resolved_angles: Vec<f64> = gate
                .angles
                .iter()
                .map(|expr| expr.evaluate(binding))
                .collect::<Result<Vec<f64>>>()?;

            let primary_angle = resolved_angles.first().copied().unwrap_or(0.0);

            bound_gates.push(BoundGate {
                kind: gate.kind.clone(),
                qubit: gate.qubit,
                qubit2: gate.qubit2,
                angle: primary_angle,
                angles: resolved_angles,
            });
        }

        Ok(BoundCircuit {
            gates: bound_gates,
            num_qubits: self.num_qubits,
        })
    }

    /// Partially bind parameters, producing a new `ParametricCircuit` where
    /// bound parameters are replaced by literals and unbound ones remain symbolic.
    pub fn partial_bind(&self, binding: &ParameterBinding) -> Result<ParametricCircuit> {
        // Validate bounds for parameters being bound
        for (name, param) in &self.parameters {
            if let Some(&value) = binding.get(name) {
                param.check_bounds(value)?;
            }
        }

        let new_gates: Vec<ParametricGate> = self
            .gates
            .iter()
            .map(|gate| {
                let new_angles: Vec<ParameterExpression> = gate
                    .angles
                    .iter()
                    .map(|expr| expr.partial_evaluate(binding))
                    .collect();
                ParametricGate {
                    kind: gate.kind.clone(),
                    qubit: gate.qubit,
                    qubit2: gate.qubit2,
                    angles: new_angles,
                }
            })
            .collect();

        // Remaining parameters: those not in the binding
        let remaining_params: HashMap<String, Parameter> = self
            .parameters
            .iter()
            .filter(|(name, _)| !binding.contains_key(*name))
            .map(|(name, param)| (name.clone(), param.clone()))
            .collect();

        Ok(ParametricCircuit {
            gates: new_gates,
            parameters: remaining_params,
            num_qubits: self.num_qubits,
        })
    }

    /// Compose two parametric circuits sequentially. Gates from `other` are
    /// appended after this circuit's gates.
    ///
    /// On parameter name conflicts, the other circuit's parameters are
    /// automatically renamed with a `_2` suffix.
    pub fn compose(&self, other: &ParametricCircuit) -> Result<ParametricCircuit> {
        let mut combined = self.clone();

        // Detect conflicting parameter names
        let self_names: HashSet<String> = self.parameters.keys().cloned().collect();
        let mut rename_map: HashMap<String, String> = HashMap::new();

        for (name, param) in &other.parameters {
            if self_names.contains(name) {
                let mut new_name = format!("{}_2", name);
                let mut counter = 3;
                while combined.parameters.contains_key(&new_name) {
                    new_name = format!("{}_{}", name, counter);
                    counter += 1;
                }
                rename_map.insert(name.clone(), new_name.clone());
                let mut renamed_param = param.clone();
                renamed_param.name = new_name.clone();
                combined
                    .parameters
                    .insert(new_name, renamed_param);
            } else {
                combined
                    .parameters
                    .insert(name.clone(), param.clone());
            }
        }

        // Append gates with renamed parameters
        for gate in &other.gates {
            let new_angles: Vec<ParameterExpression> = gate
                .angles
                .iter()
                .map(|expr| rename_expression(expr, &rename_map))
                .collect();

            combined.add_gate(ParametricGate {
                kind: gate.kind.clone(),
                qubit: gate.qubit,
                qubit2: gate.qubit2,
                angles: new_angles,
            });
        }

        Ok(combined)
    }
}

impl Default for ParametricCircuit {
    fn default() -> Self {
        Self::new()
    }
}

/// Recursively rename parameter references in an expression.
fn rename_expression(
    expr: &ParameterExpression,
    rename_map: &HashMap<String, String>,
) -> ParameterExpression {
    match expr {
        ParameterExpression::Literal(v) => ParameterExpression::Literal(*v),
        ParameterExpression::Param(name) => {
            if let Some(new_name) = rename_map.get(name) {
                ParameterExpression::Param(new_name.clone())
            } else {
                ParameterExpression::Param(name.clone())
            }
        }
        ParameterExpression::Add(a, b) => ParameterExpression::Add(
            Box::new(rename_expression(a, rename_map)),
            Box::new(rename_expression(b, rename_map)),
        ),
        ParameterExpression::Mul(a, b) => ParameterExpression::Mul(
            Box::new(rename_expression(a, rename_map)),
            Box::new(rename_expression(b, rename_map)),
        ),
        ParameterExpression::Neg(a) => {
            ParameterExpression::Neg(Box::new(rename_expression(a, rename_map)))
        }
        ParameterExpression::Sin(a) => {
            ParameterExpression::Sin(Box::new(rename_expression(a, rename_map)))
        }
        ParameterExpression::Cos(a) => {
            ParameterExpression::Cos(Box::new(rename_expression(a, rename_map)))
        }
    }
}

// ============================================================
// GRADIENT CIRCUIT (PARAMETER-SHIFT RULE)
// ============================================================

/// A pair of circuits used for parameter-shift rule gradient computation.
///
/// For a circuit with parameter theta, the gradient with respect to theta is:
///   dE/d(theta) = (E(theta + pi/2) - E(theta - pi/2)) / 2
///
/// Each `GradientCircuit` holds the shifted bindings for plus and minus shifts.
#[derive(Clone, Debug)]
pub struct GradientCircuit {
    /// Name of the parameter this gradient is computed for.
    pub parameter_name: String,
    /// Binding with the parameter shifted by +pi/2.
    pub plus_shift: ParameterBinding,
    /// Binding with the parameter shifted by -pi/2.
    pub minus_shift: ParameterBinding,
}

/// Generate gradient circuits for all parameters using the parameter-shift rule.
///
/// Given a base binding, produces one [`GradientCircuit`] per parameter,
/// each containing +pi/2 and -pi/2 shifted bindings.
pub fn parameter_shift_gradients(
    circuit: &ParametricCircuit,
    base_binding: &ParameterBinding,
) -> Vec<GradientCircuit> {
    let shift = PI / 2.0;
    let mut gradients = Vec::new();

    let mut param_names: Vec<String> = circuit.parameters.keys().cloned().collect();
    param_names.sort(); // deterministic ordering

    for name in &param_names {
        if let Some(&base_val) = base_binding.get(name) {
            let mut plus = base_binding.clone();
            plus.insert(name.clone(), base_val + shift);

            let mut minus = base_binding.clone();
            minus.insert(name.clone(), base_val - shift);

            gradients.push(GradientCircuit {
                parameter_name: name.clone(),
                plus_shift: plus,
                minus_shift: minus,
            });
        }
    }

    gradients
}

// ============================================================
// HARDWARE-EFFICIENT ANSATZ TEMPLATES
// ============================================================

/// Entanglement strategy for two-qubit gates in ansatz templates.
#[derive(Clone, Debug, PartialEq)]
pub enum EntanglementStrategy {
    /// Linear chain: qubit i entangled with qubit i+1.
    Linear,
    /// Full connectivity: every pair of qubits entangled.
    Full,
    /// Circular: linear plus wrap-around from last to first qubit.
    Circular,
}

/// Build a RealAmplitudes ansatz circuit.
///
/// Structure per layer:
///   - Ry(theta_i) on each qubit
///   - CNOT chain according to the entanglement strategy
///
/// Total parameters = num_qubits * (reps + 1)
/// (each repetition has num_qubits Ry gates, plus a final Ry layer)
pub fn real_amplitudes(
    num_qubits: usize,
    reps: usize,
    entanglement: EntanglementStrategy,
) -> ParametricCircuit {
    let mut circuit = ParametricCircuit::with_num_qubits(num_qubits);
    let mut param_idx = 0;

    for layer in 0..=reps {
        // Rotation layer: Ry on each qubit
        for qubit in 0..num_qubits {
            let name = format!("theta_{}", param_idx);
            let _ = circuit.add_parameter(Parameter::new(&name));
            circuit.add_gate(ParametricGate::ry(qubit, ParameterExpression::param(&name)));
            param_idx += 1;
        }

        // Entanglement layer (not after the last rotation layer)
        if layer < reps {
            add_entanglement_layer(&mut circuit, num_qubits, &entanglement);
        }
    }

    circuit
}

/// Build an EfficientSU2 ansatz circuit.
///
/// Structure per layer:
///   - Ry(theta_i) on each qubit
///   - Rz(theta_j) on each qubit
///   - CNOT chain according to the entanglement strategy
///
/// Total parameters = 2 * num_qubits * (reps + 1)
pub fn efficient_su2(
    num_qubits: usize,
    reps: usize,
    entanglement: EntanglementStrategy,
) -> ParametricCircuit {
    let mut circuit = ParametricCircuit::with_num_qubits(num_qubits);
    let mut param_idx = 0;

    for layer in 0..=reps {
        // Ry layer
        for qubit in 0..num_qubits {
            let name = format!("theta_{}", param_idx);
            let _ = circuit.add_parameter(Parameter::new(&name));
            circuit.add_gate(ParametricGate::ry(qubit, ParameterExpression::param(&name)));
            param_idx += 1;
        }

        // Rz layer
        for qubit in 0..num_qubits {
            let name = format!("theta_{}", param_idx);
            let _ = circuit.add_parameter(Parameter::new(&name));
            circuit.add_gate(ParametricGate::rz(qubit, ParameterExpression::param(&name)));
            param_idx += 1;
        }

        // Entanglement layer (not after the last rotation layer)
        if layer < reps {
            add_entanglement_layer(&mut circuit, num_qubits, &entanglement);
        }
    }

    circuit
}

/// Build a TwoLocal ansatz circuit.
///
/// A general template with configurable rotation gate types and entanglement
/// gate types. The rotation and entanglement layers alternate.
///
/// - `rotation_gates`: gate kinds for each rotation layer (Rx, Ry, Rz)
/// - `entanglement_gate`: two-qubit gate kind (Cx or Cz)
///
/// Total parameters = len(rotation_gates) * num_qubits * (reps + 1)
pub fn two_local(
    num_qubits: usize,
    reps: usize,
    rotation_gates: &[ParametricGateKind],
    entanglement_gate: ParametricGateKind,
    entanglement: EntanglementStrategy,
) -> ParametricCircuit {
    let mut circuit = ParametricCircuit::with_num_qubits(num_qubits);
    let mut param_idx = 0;

    for layer in 0..=reps {
        // Rotation layer: apply each rotation gate type to each qubit
        for gate_kind in rotation_gates {
            for qubit in 0..num_qubits {
                let name = format!("theta_{}", param_idx);
                let _ = circuit.add_parameter(Parameter::new(&name));
                let gate = match gate_kind {
                    ParametricGateKind::Rx => {
                        ParametricGate::rx(qubit, ParameterExpression::param(&name))
                    }
                    ParametricGateKind::Ry => {
                        ParametricGate::ry(qubit, ParameterExpression::param(&name))
                    }
                    ParametricGateKind::Rz => {
                        ParametricGate::rz(qubit, ParameterExpression::param(&name))
                    }
                    _ => ParametricGate::ry(qubit, ParameterExpression::param(&name)),
                };
                circuit.add_gate(gate);
                param_idx += 1;
            }
        }

        // Entanglement layer (not after the last rotation layer)
        if layer < reps {
            add_entanglement_layer_with_gate(
                &mut circuit,
                num_qubits,
                &entanglement,
                &entanglement_gate,
            );
        }
    }

    circuit
}

/// Add CNOT entanglement gates according to the strategy.
fn add_entanglement_layer(
    circuit: &mut ParametricCircuit,
    num_qubits: usize,
    strategy: &EntanglementStrategy,
) {
    add_entanglement_layer_with_gate(circuit, num_qubits, strategy, &ParametricGateKind::Cx);
}

/// Add two-qubit entanglement gates of a specified kind.
fn add_entanglement_layer_with_gate(
    circuit: &mut ParametricCircuit,
    num_qubits: usize,
    strategy: &EntanglementStrategy,
    gate_kind: &ParametricGateKind,
) {
    let pairs: Vec<(usize, usize)> = match strategy {
        EntanglementStrategy::Linear => (0..num_qubits.saturating_sub(1))
            .map(|i| (i, i + 1))
            .collect(),
        EntanglementStrategy::Full => {
            let mut p = Vec::new();
            for i in 0..num_qubits {
                for j in (i + 1)..num_qubits {
                    p.push((i, j));
                }
            }
            p
        }
        EntanglementStrategy::Circular => {
            let mut p: Vec<(usize, usize)> = (0..num_qubits.saturating_sub(1))
                .map(|i| (i, i + 1))
                .collect();
            if num_qubits > 1 {
                p.push((num_qubits - 1, 0));
            }
            p
        }
    };

    for (q1, q2) in pairs {
        let gate = match gate_kind {
            ParametricGateKind::Cz => ParametricGate::cz(q1, q2),
            _ => ParametricGate::cx(q1, q2),
        };
        circuit.add_gate(gate);
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn parameter_creation() {
        let p = Parameter::new("theta");
        assert_eq!(p.name, "theta");
        assert_eq!(p.initial_value, 0.0);
        assert!(p.min.is_infinite() && p.min < 0.0);
        assert!(p.max.is_infinite() && p.max > 0.0);

        let p2 = Parameter::with_bounds("phi", -PI, PI);
        assert_eq!(p2.name, "phi");
        assert_eq!(p2.min, -PI);
        assert_eq!(p2.max, PI);
        assert_eq!(p2.initial_value, 0.0);

        let p3 = Parameter::with_bounds_and_initial("lambda", 0.0, 2.0 * PI, 0.5);
        assert_eq!(p3.initial_value, 0.5);
        assert!(p3.check_bounds(1.0).is_ok());
        assert!(p3.check_bounds(-0.1).is_err());
    }

    #[test]
    fn expression_evaluation() {
        let binding: ParameterBinding =
            HashMap::from([("x".to_string(), 2.0), ("y".to_string(), 3.0)]);

        // Literal
        let expr = ParameterExpression::literal(42.0);
        assert_eq!(expr.evaluate(&binding).unwrap(), 42.0);

        // Param
        let expr = ParameterExpression::param("x");
        assert_eq!(expr.evaluate(&binding).unwrap(), 2.0);

        // Unbound param
        let expr = ParameterExpression::param("z");
        assert!(matches!(
            expr.evaluate(&binding),
            Err(ParametricError::UnboundParameter(ref n)) if n == "z"
        ));

        // Arithmetic: 2*x + y = 2*2 + 3 = 7
        let expr = ParameterExpression::add(
            ParameterExpression::mul(
                ParameterExpression::literal(2.0),
                ParameterExpression::param("x"),
            ),
            ParameterExpression::param("y"),
        );
        assert!((expr.evaluate(&binding).unwrap() - 7.0).abs() < 1e-12);

        // Negation: -x = -2
        let expr = ParameterExpression::neg(ParameterExpression::param("x"));
        assert!((expr.evaluate(&binding).unwrap() - (-2.0)).abs() < 1e-12);
    }

    #[test]
    fn expression_composition() {
        let binding: ParameterBinding = HashMap::from([("theta".to_string(), PI / 4.0)]);

        // sin(2 * theta) = sin(pi/2) = 1.0
        let expr = ParameterExpression::sin(ParameterExpression::mul(
            ParameterExpression::literal(2.0),
            ParameterExpression::param("theta"),
        ));
        assert!((expr.evaluate(&binding).unwrap() - 1.0).abs() < 1e-12);

        // cos(theta) + sin(theta) = cos(pi/4) + sin(pi/4) = sqrt(2)
        let expr = ParameterExpression::add(
            ParameterExpression::cos(ParameterExpression::param("theta")),
            ParameterExpression::sin(ParameterExpression::param("theta")),
        );
        let expected = (PI / 4.0).cos() + (PI / 4.0).sin();
        assert!((expr.evaluate(&binding).unwrap() - expected).abs() < 1e-12);

        // Nested: sin(cos(theta))
        let expr = ParameterExpression::sin(ParameterExpression::cos(
            ParameterExpression::param("theta"),
        ));
        let expected = (PI / 4.0).cos().sin();
        assert!((expr.evaluate(&binding).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn parametric_gate() {
        let gate = ParametricGate::rz(0, ParameterExpression::param("theta"));
        assert_eq!(gate.kind, ParametricGateKind::Rz);
        assert_eq!(gate.qubit, 0);
        assert!(gate.qubit2.is_none());
        assert_eq!(gate.angles.len(), 1);
        assert!(gate.parameters().contains("theta"));
    }

    #[test]
    fn circuit_builder() {
        let mut circuit = ParametricCircuit::new();
        circuit
            .add_parameter(Parameter::new("theta"))
            .unwrap();
        circuit
            .add_parameter(Parameter::new("phi"))
            .unwrap();

        circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));
        circuit.add_gate(ParametricGate::ry(1, ParameterExpression::param("phi")));
        circuit.add_gate(ParametricGate::cx(0, 1));

        assert_eq!(circuit.gates.len(), 3);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.parameters.len(), 2);

        let ref_params = circuit.referenced_parameters();
        assert!(ref_params.contains("theta"));
        assert!(ref_params.contains("phi"));
    }

    #[test]
    fn parameter_binding() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("theta")).unwrap();
        circuit.add_parameter(Parameter::new("phi")).unwrap();

        circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));
        circuit.add_gate(ParametricGate::ry(1, ParameterExpression::param("phi")));
        circuit.add_gate(ParametricGate::cx(0, 1));

        let binding: ParameterBinding =
            HashMap::from([("theta".to_string(), 1.0), ("phi".to_string(), 2.0)]);

        let bound = circuit.bind(&binding).unwrap();
        assert_eq!(bound.gates.len(), 3);
        assert!((bound.gates[0].angle - 1.0).abs() < 1e-12);
        assert!((bound.gates[1].angle - 2.0).abs() < 1e-12);
        assert_eq!(bound.num_qubits, 2);
    }

    #[test]
    fn partial_binding() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("theta")).unwrap();
        circuit.add_parameter(Parameter::new("phi")).unwrap();

        circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));
        circuit.add_gate(ParametricGate::ry(
            1,
            ParameterExpression::add(
                ParameterExpression::param("phi"),
                ParameterExpression::param("theta"),
            ),
        ));

        // Bind only theta
        let partial: ParameterBinding = HashMap::from([("theta".to_string(), 1.5)]);
        let partially_bound = circuit.partial_bind(&partial).unwrap();

        // theta is now gone from the parameter registry
        assert_eq!(partially_bound.parameters.len(), 1);
        assert!(partially_bound.parameters.contains_key("phi"));
        assert!(!partially_bound.parameters.contains_key("theta"));

        // First gate's angle should be a literal now
        assert!(partially_bound.gates[0].angles[0].is_fully_bound());

        // Second gate still references phi
        assert!(!partially_bound.gates[1].angles[0].is_fully_bound());

        // Now bind phi to get fully bound
        let full: ParameterBinding = HashMap::from([("phi".to_string(), 0.5)]);
        let bound = partially_bound.bind(&full).unwrap();
        // Second gate: phi + theta = 0.5 + 1.5 = 2.0
        assert!((bound.gates[1].angle - 2.0).abs() < 1e-12);
    }

    #[test]
    fn parameter_shift() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("theta")).unwrap();
        circuit.add_parameter(Parameter::new("phi")).unwrap();

        circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));
        circuit.add_gate(ParametricGate::ry(1, ParameterExpression::param("phi")));

        let base: ParameterBinding =
            HashMap::from([("theta".to_string(), 1.0), ("phi".to_string(), 2.0)]);

        let grads = parameter_shift_gradients(&circuit, &base);
        assert_eq!(grads.len(), 2); // one per parameter

        // Find the gradient for phi
        let phi_grad = grads.iter().find(|g| g.parameter_name == "phi").unwrap();
        assert!((phi_grad.plus_shift["phi"] - (2.0 + PI / 2.0)).abs() < 1e-12);
        assert!((phi_grad.minus_shift["phi"] - (2.0 - PI / 2.0)).abs() < 1e-12);
        // Other parameters unchanged
        assert!((phi_grad.plus_shift["theta"] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn real_amplitudes_ansatz() {
        let num_qubits = 3;
        let reps = 2;
        let circuit = real_amplitudes(num_qubits, reps, EntanglementStrategy::Linear);

        // Parameters: num_qubits * (reps + 1) = 3 * 3 = 9
        assert_eq!(circuit.parameters.len(), 9);

        // Count rotation gates: 9 Ry
        let ry_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Ry)
            .count();
        assert_eq!(ry_count, 9);

        // Count entanglement gates: 2 layers * (num_qubits - 1) = 2 * 2 = 4 CX
        let cx_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cx)
            .count();
        assert_eq!(cx_count, 4);

        assert_eq!(circuit.num_qubits, num_qubits);
    }

    #[test]
    fn efficient_su2_ansatz() {
        let num_qubits = 2;
        let reps = 1;
        let circuit = efficient_su2(num_qubits, reps, EntanglementStrategy::Linear);

        // Parameters: 2 * num_qubits * (reps + 1) = 2 * 2 * 2 = 8
        assert_eq!(circuit.parameters.len(), 8);

        // Count Ry gates: num_qubits * (reps + 1) = 2 * 2 = 4
        let ry_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Ry)
            .count();
        assert_eq!(ry_count, 4);

        // Count Rz gates: same as Ry
        let rz_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Rz)
            .count();
        assert_eq!(rz_count, 4);

        // Entanglement: 1 layer * (num_qubits - 1) = 1 CX
        let cx_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cx)
            .count();
        assert_eq!(cx_count, 1);
    }

    #[test]
    fn two_local_ansatz() {
        let num_qubits = 3;
        let reps = 1;
        let circuit = two_local(
            num_qubits,
            reps,
            &[ParametricGateKind::Ry, ParametricGateKind::Rz],
            ParametricGateKind::Cz,
            EntanglementStrategy::Linear,
        );

        // Parameters: 2 rotation_gates * num_qubits * (reps + 1) = 2 * 3 * 2 = 12
        assert_eq!(circuit.parameters.len(), 12);

        // Ry: num_qubits * (reps + 1) = 3 * 2 = 6
        let ry_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Ry)
            .count();
        assert_eq!(ry_count, 6);

        // Rz: same
        let rz_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Rz)
            .count();
        assert_eq!(rz_count, 6);

        // CZ entanglement: 1 layer * (num_qubits - 1) = 2
        let cz_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cz)
            .count();
        assert_eq!(cz_count, 2);

        // Verify alternating structure: rotation block, then entanglement, then rotation block
        // First 6 gates should be rotations (Ry, Ry, Ry, Rz, Rz, Rz)
        assert_eq!(circuit.gates[0].kind, ParametricGateKind::Ry);
        assert_eq!(circuit.gates[3].kind, ParametricGateKind::Rz);
        // Then entanglement
        assert_eq!(circuit.gates[6].kind, ParametricGateKind::Cz);
        assert_eq!(circuit.gates[7].kind, ParametricGateKind::Cz);
        // Then more rotations
        assert_eq!(circuit.gates[8].kind, ParametricGateKind::Ry);
    }

    #[test]
    fn circuit_composition() {
        let mut c1 = ParametricCircuit::new();
        c1.add_parameter(Parameter::new("alpha")).unwrap();
        c1.add_gate(ParametricGate::ry(0, ParameterExpression::param("alpha")));

        let mut c2 = ParametricCircuit::new();
        c2.add_parameter(Parameter::new("alpha")).unwrap();
        c2.add_parameter(Parameter::new("beta")).unwrap();
        c2.add_gate(ParametricGate::rz(0, ParameterExpression::param("alpha")));
        c2.add_gate(ParametricGate::rx(1, ParameterExpression::param("beta")));

        let combined = c1.compose(&c2).unwrap();

        // 3 parameters: alpha (from c1), alpha_2 (renamed from c2), beta
        assert_eq!(combined.parameters.len(), 3);
        assert!(combined.parameters.contains_key("alpha"));
        assert!(combined.parameters.contains_key("alpha_2"));
        assert!(combined.parameters.contains_key("beta"));

        // 3 gates total
        assert_eq!(combined.gates.len(), 3);

        // The second gate (from c2) should now reference alpha_2
        let second_gate_params = combined.gates[1].parameters();
        assert!(second_gate_params.contains("alpha_2"));
        assert!(!second_gate_params.contains("alpha"));

        // Bind all and verify
        let binding: ParameterBinding = HashMap::from([
            ("alpha".to_string(), 1.0),
            ("alpha_2".to_string(), 2.0),
            ("beta".to_string(), 3.0),
        ]);
        let bound = combined.bind(&binding).unwrap();
        assert!((bound.gates[0].angle - 1.0).abs() < 1e-12);
        assert!((bound.gates[1].angle - 2.0).abs() < 1e-12);
        assert!((bound.gates[2].angle - 3.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_parameter_error() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("theta")).unwrap();

        let result = circuit.add_parameter(Parameter::new("theta"));
        assert!(matches!(
            result,
            Err(ParametricError::DuplicateParameter(ref n)) if n == "theta"
        ));
    }

    #[test]
    fn out_of_bounds_error() {
        let mut circuit = ParametricCircuit::new();
        circuit
            .add_parameter(Parameter::with_bounds("theta", 0.0, PI))
            .unwrap();
        circuit.add_gate(ParametricGate::rz(0, ParameterExpression::param("theta")));

        // Value within bounds succeeds
        let binding: ParameterBinding = HashMap::from([("theta".to_string(), 1.0)]);
        assert!(circuit.bind(&binding).is_ok());

        // Value outside bounds fails
        let bad_binding: ParameterBinding = HashMap::from([("theta".to_string(), -1.0)]);
        let err = circuit.bind(&bad_binding).unwrap_err();
        match err {
            ParametricError::OutOfBounds {
                ref name,
                value,
                min,
                max,
            } => {
                assert_eq!(name, "theta");
                assert!((value - (-1.0)).abs() < 1e-12);
                assert!((min - 0.0).abs() < 1e-12);
                assert!((max - PI).abs() < 1e-12);
            }
            _ => panic!("expected OutOfBounds error, got {:?}", err),
        }

        // Value above upper bound also fails
        let high_binding: ParameterBinding = HashMap::from([("theta".to_string(), 4.0)]);
        assert!(matches!(
            circuit.bind(&high_binding),
            Err(ParametricError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn gradient_circuits() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("a")).unwrap();
        circuit.add_parameter(Parameter::new("b")).unwrap();
        circuit.add_parameter(Parameter::new("c")).unwrap();

        circuit.add_gate(ParametricGate::rx(0, ParameterExpression::param("a")));
        circuit.add_gate(ParametricGate::ry(1, ParameterExpression::param("b")));
        circuit.add_gate(ParametricGate::rz(2, ParameterExpression::param("c")));

        let base: ParameterBinding = HashMap::from([
            ("a".to_string(), 0.5),
            ("b".to_string(), 1.0),
            ("c".to_string(), 1.5),
        ]);

        let grads = parameter_shift_gradients(&circuit, &base);

        // One gradient circuit per parameter
        assert_eq!(grads.len(), 3);

        // Each produces 2 shifted bindings (plus and minus)
        for grad in &grads {
            let base_val = base[&grad.parameter_name];
            let plus_val = grad.plus_shift[&grad.parameter_name];
            let minus_val = grad.minus_shift[&grad.parameter_name];

            assert!((plus_val - (base_val + PI / 2.0)).abs() < 1e-12);
            assert!((minus_val - (base_val - PI / 2.0)).abs() < 1e-12);

            // All other parameters unchanged
            for (name, &val) in &base {
                if name != &grad.parameter_name {
                    assert!((grad.plus_shift[name] - val).abs() < 1e-12);
                    assert!((grad.minus_shift[name] - val).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn expression_display() {
        let expr = ParameterExpression::add(
            ParameterExpression::mul(
                ParameterExpression::literal(2.0),
                ParameterExpression::param("theta"),
            ),
            ParameterExpression::literal(PI / 4.0),
        );
        let s = format!("{}", expr);
        assert!(s.contains("theta"));
        assert!(s.contains("2"));
    }

    #[test]
    fn expression_partial_evaluate() {
        let expr = ParameterExpression::add(
            ParameterExpression::param("x"),
            ParameterExpression::mul(
                ParameterExpression::literal(3.0),
                ParameterExpression::param("y"),
            ),
        );

        // Partially bind x only
        let partial: ParameterBinding = HashMap::from([("x".to_string(), 10.0)]);
        let reduced = expr.partial_evaluate(&partial);

        // y is still symbolic
        assert!(!reduced.is_fully_bound());

        // Now bind y
        let full: ParameterBinding = HashMap::from([("y".to_string(), 2.0)]);
        let val = reduced.evaluate(&full).unwrap();
        // x + 3*y = 10 + 6 = 16
        assert!((val - 16.0).abs() < 1e-12);
    }

    #[test]
    fn u3_gate() {
        let mut circuit = ParametricCircuit::new();
        circuit.add_parameter(Parameter::new("t")).unwrap();
        circuit.add_parameter(Parameter::new("p")).unwrap();
        circuit.add_parameter(Parameter::new("l")).unwrap();

        circuit.add_gate(ParametricGate::u3(
            0,
            ParameterExpression::param("t"),
            ParameterExpression::param("p"),
            ParameterExpression::param("l"),
        ));

        let binding: ParameterBinding = HashMap::from([
            ("t".to_string(), 1.0),
            ("p".to_string(), 2.0),
            ("l".to_string(), 3.0),
        ]);

        let bound = circuit.bind(&binding).unwrap();
        assert_eq!(bound.gates[0].angles.len(), 3);
        assert!((bound.gates[0].angles[0] - 1.0).abs() < 1e-12);
        assert!((bound.gates[0].angles[1] - 2.0).abs() < 1e-12);
        assert!((bound.gates[0].angles[2] - 3.0).abs() < 1e-12);
        assert!((bound.gates[0].angle - 1.0).abs() < 1e-12); // primary = first
    }

    #[test]
    fn full_entanglement_strategy() {
        let circuit = real_amplitudes(4, 1, EntanglementStrategy::Full);

        // Full entanglement: C(4,2) = 6 pairs per layer, 1 layer
        let cx_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cx)
            .count();
        assert_eq!(cx_count, 6);
    }

    #[test]
    fn circular_entanglement_strategy() {
        let circuit = real_amplitudes(4, 1, EntanglementStrategy::Circular);

        // Circular: 3 linear + 1 wrap-around = 4 per layer, 1 layer
        let cx_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cx)
            .count();
        assert_eq!(cx_count, 4);
    }

    #[test]
    fn single_qubit_ansatz() {
        // Edge case: 1 qubit should produce rotations but no entanglement
        let circuit = real_amplitudes(1, 2, EntanglementStrategy::Linear);
        assert_eq!(circuit.parameters.len(), 3); // 1 qubit * 3 layers
        let cx_count = circuit
            .gates
            .iter()
            .filter(|g| g.kind == ParametricGateKind::Cx)
            .count();
        assert_eq!(cx_count, 0);
    }

    #[test]
    fn error_display() {
        let err = ParametricError::UnboundParameter("x".to_string());
        assert_eq!(format!("{}", err), "parameter 'x' has no bound value");

        let err = ParametricError::DuplicateParameter("y".to_string());
        assert_eq!(format!("{}", err), "parameter 'y' is already registered");

        let err = ParametricError::OutOfBounds {
            name: "z".to_string(),
            value: 5.0,
            min: 0.0,
            max: 3.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("z"));
        assert!(msg.contains("5"));

        let err = ParametricError::ExpressionError("division by zero".to_string());
        assert!(format!("{}", err).contains("division by zero"));
    }
}
