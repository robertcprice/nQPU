//! Circuit Optimization with Detailed Reporting
//!
//! Transpile quantum circuits to native gate sets, optimize gate count and depth,
//! and generate comprehensive before/after reports. Implements a pass-based
//! optimization pipeline with gate cancellation, commutation analysis, template
//! matching, single-qubit fusion (ZYZ decomposition), two-qubit decomposition,
//! layout mapping, SWAP insertion, and peephole optimization.
//!
//! Standalone module -- no crate-level imports required.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::optimization_report::*;
//!
//! let mut circuit = OptCircuit::new(2);
//! circuit.push(OptGate::H(0));
//! circuit.push(OptGate::H(0));  // cancels with previous H
//! circuit.push(OptGate::CX(0, 1));
//!
//! let config = OptimizerConfig::level(OptimizationLevel::Level2);
//! let optimizer = CircuitOptimizer::new(config);
//! let report = optimizer.optimize(&circuit).unwrap();
//! assert!(report.after.total_gates < report.before.total_gates);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;
use std::fmt;
use std::time::Instant;

// ============================================================
// ERROR TYPE
// ============================================================

/// Errors produced by the optimization pipeline.
#[derive(Debug, Clone)]
pub enum OptimizerError {
    /// The input circuit is structurally invalid (e.g. qubit index out of range).
    InvalidCircuit(String),
    /// An individual optimization pass failed.
    PassFailed(String),
    /// Routing could not satisfy connectivity constraints.
    RoutingFailed(String),
    /// A gate type is not supported by the target basis.
    UnsupportedGate(String),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::InvalidCircuit(s) => write!(f, "Invalid circuit: {}", s),
            OptimizerError::PassFailed(s) => write!(f, "Pass failed: {}", s),
            OptimizerError::RoutingFailed(s) => write!(f, "Routing failed: {}", s),
            OptimizerError::UnsupportedGate(s) => write!(f, "Unsupported gate: {}", s),
        }
    }
}

impl std::error::Error for OptimizerError {}

// ============================================================
// GATE REPRESENTATION
// ============================================================

/// A quantum gate in the optimization IR.
#[derive(Debug, Clone, PartialEq)]
pub enum OptGate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    CX(usize, usize),
    CZ(usize, usize),
    Swap(usize, usize),
    Toffoli(usize, usize, usize),
    /// U3(qubit, theta, phi, lambda)
    U3(usize, f64, f64, f64),
    /// Barrier on specified qubits -- optimization boundary.
    Barrier(Vec<usize>),
    /// Measurement on a single qubit.
    Measure(usize),
}

impl OptGate {
    /// Name string used in metrics tables.
    pub fn name(&self) -> &'static str {
        match self {
            OptGate::H(_) => "h",
            OptGate::X(_) => "x",
            OptGate::Y(_) => "y",
            OptGate::Z(_) => "z",
            OptGate::S(_) => "s",
            OptGate::Sdg(_) => "sdg",
            OptGate::T(_) => "t",
            OptGate::Tdg(_) => "tdg",
            OptGate::Rx(..) => "rx",
            OptGate::Ry(..) => "ry",
            OptGate::Rz(..) => "rz",
            OptGate::CX(..) => "cx",
            OptGate::CZ(..) => "cz",
            OptGate::Swap(..) => "swap",
            OptGate::Toffoli(..) => "toffoli",
            OptGate::U3(..) => "u3",
            OptGate::Barrier(_) => "barrier",
            OptGate::Measure(_) => "measure",
        }
    }

    /// Qubits this gate touches.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            OptGate::H(q)
            | OptGate::X(q)
            | OptGate::Y(q)
            | OptGate::Z(q)
            | OptGate::S(q)
            | OptGate::Sdg(q)
            | OptGate::T(q)
            | OptGate::Tdg(q)
            | OptGate::Rx(q, _)
            | OptGate::Ry(q, _)
            | OptGate::Rz(q, _)
            | OptGate::U3(q, ..)
            | OptGate::Measure(q) => vec![*q],
            OptGate::CX(a, b) | OptGate::CZ(a, b) | OptGate::Swap(a, b) => vec![*a, *b],
            OptGate::Toffoli(a, b, c) => vec![*a, *b, *c],
            OptGate::Barrier(qs) => qs.clone(),
        }
    }

    /// True if this is a single-qubit unitary (not barrier/measure).
    pub fn is_single_qubit_unitary(&self) -> bool {
        matches!(
            self,
            OptGate::H(_)
                | OptGate::X(_)
                | OptGate::Y(_)
                | OptGate::Z(_)
                | OptGate::S(_)
                | OptGate::Sdg(_)
                | OptGate::T(_)
                | OptGate::Tdg(_)
                | OptGate::Rx(..)
                | OptGate::Ry(..)
                | OptGate::Rz(..)
                | OptGate::U3(..)
        )
    }

    /// True if this is a two-qubit gate.
    pub fn is_two_qubit(&self) -> bool {
        matches!(self, OptGate::CX(..) | OptGate::CZ(..) | OptGate::Swap(..))
    }

    /// True if this is a three-qubit gate.
    pub fn is_three_qubit(&self) -> bool {
        matches!(self, OptGate::Toffoli(..))
    }

    /// True for self-inverse gates (applying twice = identity).
    pub fn is_self_inverse(&self) -> bool {
        matches!(
            self,
            OptGate::H(_) | OptGate::X(_) | OptGate::Y(_) | OptGate::Z(_)
        )
    }

    /// Return the inverse gate if one exists in a simple closed form.
    pub fn inverse(&self) -> Option<OptGate> {
        match self {
            OptGate::H(q) => Some(OptGate::H(*q)),
            OptGate::X(q) => Some(OptGate::X(*q)),
            OptGate::Y(q) => Some(OptGate::Y(*q)),
            OptGate::Z(q) => Some(OptGate::Z(*q)),
            OptGate::S(q) => Some(OptGate::Sdg(*q)),
            OptGate::Sdg(q) => Some(OptGate::S(*q)),
            OptGate::T(q) => Some(OptGate::Tdg(*q)),
            OptGate::Tdg(q) => Some(OptGate::T(*q)),
            OptGate::Rx(q, a) => Some(OptGate::Rx(*q, -a)),
            OptGate::Ry(q, a) => Some(OptGate::Ry(*q, -a)),
            OptGate::Rz(q, a) => Some(OptGate::Rz(*q, -a)),
            OptGate::CX(a, b) => Some(OptGate::CX(*a, *b)),
            OptGate::CZ(a, b) => Some(OptGate::CZ(*a, *b)),
            OptGate::Swap(a, b) => Some(OptGate::Swap(*a, *b)),
            _ => None,
        }
    }

    /// 2x2 unitary matrix for single-qubit gates (row-major [a,b,c,d]).
    /// Returns None for multi-qubit or non-unitary operations.
    pub fn unitary_2x2(&self) -> Option<[Complex; 4]> {
        let i = Complex::new(0.0, 1.0);
        let z = Complex::ZERO;
        let o = Complex::ONE;
        let s2 = Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);

        match self {
            OptGate::H(_) => Some([
                s2,
                s2,
                s2,
                Complex::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ]),
            OptGate::X(_) => Some([z, o, o, z]),
            OptGate::Y(_) => Some([z, Complex::new(0.0, -1.0), i, z]),
            OptGate::Z(_) => Some([o, z, z, Complex::new(-1.0, 0.0)]),
            OptGate::S(_) => Some([o, z, z, i]),
            OptGate::Sdg(_) => Some([o, z, z, Complex::new(0.0, -1.0)]),
            OptGate::T(_) => {
                let t = Complex::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                );
                Some([o, z, z, t])
            }
            OptGate::Tdg(_) => {
                let t = Complex::new(
                    std::f64::consts::FRAC_1_SQRT_2,
                    -std::f64::consts::FRAC_1_SQRT_2,
                );
                Some([o, z, z, t])
            }
            OptGate::Rx(_, theta) => {
                let c = Complex::new((theta / 2.0).cos(), 0.0);
                let s = Complex::new(0.0, -(theta / 2.0).sin());
                Some([c, s, s, c])
            }
            OptGate::Ry(_, theta) => {
                let c = Complex::new((theta / 2.0).cos(), 0.0);
                let sp = Complex::new((theta / 2.0).sin(), 0.0);
                let sn = Complex::new(-(theta / 2.0).sin(), 0.0);
                Some([c, sn, sp, c])
            }
            OptGate::Rz(_, theta) => {
                let ep = Complex::from_polar(1.0, -theta / 2.0);
                let em = Complex::from_polar(1.0, theta / 2.0);
                Some([ep, z, z, em])
            }
            OptGate::U3(_, theta, phi, lambda) => {
                let ct = (theta / 2.0).cos();
                let st = (theta / 2.0).sin();
                let a = Complex::new(ct, 0.0);
                let b = Complex::new(-st, 0.0) * Complex::from_polar(1.0, *lambda);
                let c = Complex::new(st, 0.0) * Complex::from_polar(1.0, *phi);
                let d = Complex::new(ct, 0.0) * Complex::from_polar(1.0, phi + lambda);
                Some([a, b, c, d])
            }
            _ => None,
        }
    }
}

impl fmt::Display for OptGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptGate::H(q) => write!(f, "H q{}", q),
            OptGate::X(q) => write!(f, "X q{}", q),
            OptGate::Y(q) => write!(f, "Y q{}", q),
            OptGate::Z(q) => write!(f, "Z q{}", q),
            OptGate::S(q) => write!(f, "S q{}", q),
            OptGate::Sdg(q) => write!(f, "Sdg q{}", q),
            OptGate::T(q) => write!(f, "T q{}", q),
            OptGate::Tdg(q) => write!(f, "Tdg q{}", q),
            OptGate::Rx(q, a) => write!(f, "Rx({:.4}) q{}", a, q),
            OptGate::Ry(q, a) => write!(f, "Ry({:.4}) q{}", a, q),
            OptGate::Rz(q, a) => write!(f, "Rz({:.4}) q{}", a, q),
            OptGate::CX(a, b) => write!(f, "CX q{},q{}", a, b),
            OptGate::CZ(a, b) => write!(f, "CZ q{},q{}", a, b),
            OptGate::Swap(a, b) => write!(f, "SWAP q{},q{}", a, b),
            OptGate::Toffoli(a, b, c) => write!(f, "Toffoli q{},q{},q{}", a, b, c),
            OptGate::U3(q, t, p, l) => write!(f, "U3({:.4},{:.4},{:.4}) q{}", t, p, l, q),
            OptGate::Barrier(qs) => write!(f, "Barrier {:?}", qs),
            OptGate::Measure(q) => write!(f, "Measure q{}", q),
        }
    }
}

// ============================================================
// MINIMAL COMPLEX TYPE (self-contained)
// ============================================================

/// Minimal complex number for gate matrix arithmetic.
/// Avoids external dependency on num_complex for this standalone module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Complex { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Complex {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    pub fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Complex;
    fn mul(self, rhs: f64) -> Complex {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

// ============================================================
// 2x2 MATRIX UTILITIES
// ============================================================

/// Row-major 2x2 complex matrix [a, b, c, d] = [[a,b],[c,d]].
type Mat2 = [Complex; 4];

fn mat2_mul(a: &Mat2, b: &Mat2) -> Mat2 {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
    ]
}

fn mat2_identity() -> Mat2 {
    [Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ONE]
}

/// Check if a 2x2 matrix is close to identity (up to global phase).
fn mat2_is_identity(m: &Mat2, tol: f64) -> bool {
    // Remove global phase from m[0,0]
    let phase = if m[0].norm() > 1e-12 {
        Complex::from_polar(1.0, -m[0].arg())
    } else {
        Complex::ONE
    };
    let a = m[0] * phase;
    let b = m[1] * phase;
    let c = m[2] * phase;
    let d = m[3] * phase;

    (a.re - 1.0).abs() < tol
        && a.im.abs() < tol
        && b.norm() < tol
        && c.norm() < tol
        && (d.re - 1.0).abs() < tol
        && d.im.abs() < tol
}

/// ZYZ decomposition: U = e^{i*alpha} Rz(beta) Ry(gamma) Rz(delta).
/// Returns (alpha, beta, gamma, delta).
fn zyz_decompose(u: &Mat2) -> (f64, f64, f64, f64) {
    // Compute determinant to extract global phase
    let det = u[0] * u[3] - u[1] * u[2];
    let det_phase = det.arg() / 2.0;

    // Remove global phase
    let phase_inv = Complex::from_polar(1.0, -det_phase);
    let m = [
        u[0] * phase_inv,
        u[1] * phase_inv,
        u[2] * phase_inv,
        u[3] * phase_inv,
    ];

    // For SU(2) matrix [[a, -b*], [b, a*]] with |a|^2 + |b|^2 = 1:
    // a = cos(gamma/2) * exp(i*(beta+delta)/2)  -- but negated sign convention
    // b = sin(gamma/2) * exp(i*(beta-delta)/2)  -- standard ZYZ
    //
    // m[0] = cos(gamma/2) * exp(-i*(beta+delta)/2)
    // m[1] = -sin(gamma/2) * exp(-i*(beta-delta)/2)
    // m[2] = sin(gamma/2) * exp(i*(beta-delta)/2)
    // m[3] = cos(gamma/2) * exp(i*(beta+delta)/2)

    let a_abs = m[0].norm();
    let b_abs = m[2].norm();

    let gamma = 2.0 * b_abs.atan2(a_abs);

    if a_abs < 1e-12 {
        // gamma ~ pi, a ~ 0
        let beta = 0.0;
        let delta = -m[2].arg() * 2.0;
        (det_phase, beta, gamma, delta)
    } else if b_abs < 1e-12 {
        // gamma ~ 0, b ~ 0
        let beta_plus_delta = -m[0].arg() * 2.0;
        let beta = beta_plus_delta / 2.0;
        let delta = beta_plus_delta / 2.0;
        (det_phase, beta, gamma, delta)
    } else {
        let a_arg = m[0].arg(); // = -(beta+delta)/2
        let b_arg = m[2].arg(); // =  (beta-delta)/2

        let beta = -a_arg + b_arg;
        let delta = -a_arg - b_arg;
        (det_phase, beta, gamma, delta)
    }
}

/// Convert ZYZ angles to a gate sequence on qubit q.
/// Returns an optimized gate list (skips near-zero rotations).
fn zyz_to_gates(q: usize, beta: f64, gamma: f64, delta: f64) -> Vec<OptGate> {
    let tol = 1e-10;
    let mut gates = Vec::new();

    // When gamma ≈ 0 (pure Z rotation), merge the two Rz rotations into one
    if gamma.abs() <= tol {
        let total = beta + delta;
        if total.abs() > tol {
            gates.push(OptGate::Rz(q, total));
        }
        return gates;
    }

    if delta.abs() > tol {
        gates.push(OptGate::Rz(q, delta));
    }
    gates.push(OptGate::Ry(q, gamma));
    if beta.abs() > tol {
        gates.push(OptGate::Rz(q, beta));
    }
    gates
}

// ============================================================
// CIRCUIT
// ============================================================

/// Quantum circuit for optimization.
#[derive(Debug, Clone)]
pub struct OptCircuit {
    pub num_qubits: usize,
    pub gates: Vec<OptGate>,
}

impl OptCircuit {
    /// Create an empty circuit on `n` qubits.
    pub fn new(num_qubits: usize) -> Self {
        OptCircuit {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Append a gate.
    pub fn push(&mut self, gate: OptGate) {
        self.gates.push(gate);
    }

    /// Validate qubit indices.
    pub fn validate(&self) -> Result<(), OptimizerError> {
        for (i, g) in self.gates.iter().enumerate() {
            for q in g.qubits() {
                if q >= self.num_qubits {
                    return Err(OptimizerError::InvalidCircuit(format!(
                        "Gate {} references qubit {} but circuit has {} qubits",
                        i, q, self.num_qubits
                    )));
                }
            }
        }
        Ok(())
    }

    /// Compute circuit metrics.
    pub fn metrics(&self) -> CircuitMetrics {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        let mut single = 0usize;
        let mut two = 0usize;
        let mut three = 0usize;
        let mut cx_count = 0usize;
        let mut t_count = 0usize;

        for g in &self.gates {
            match g {
                OptGate::Barrier(_) | OptGate::Measure(_) => {}
                _ => {
                    *counts.entry(g.name()).or_insert(0) += 1;
                    if g.is_single_qubit_unitary() {
                        single += 1;
                    }
                    if g.is_two_qubit() {
                        two += 1;
                    }
                    if g.is_three_qubit() {
                        three += 1;
                    }
                    if matches!(g, OptGate::CX(..)) {
                        cx_count += 1;
                    }
                    if matches!(g, OptGate::T(_) | OptGate::Tdg(_)) {
                        t_count += 1;
                    }
                }
            }
        }

        let total = single + two + three;
        let depth = self.compute_depth();

        let mut gate_counts: Vec<(String, usize)> = counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        gate_counts.sort_by(|a, b| b.1.cmp(&a.1));

        CircuitMetrics {
            num_qubits: self.num_qubits,
            total_gates: total,
            single_qubit_gates: single,
            two_qubit_gates: two,
            three_qubit_gates: three,
            depth,
            cx_count,
            t_count,
            gate_counts,
        }
    }

    /// Compute the circuit depth (critical-path length through the DAG).
    pub fn compute_depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }
        // Track the depth at each qubit wire.
        let mut qubit_depth = vec![0usize; self.num_qubits];

        for g in &self.gates {
            match g {
                OptGate::Barrier(_) | OptGate::Measure(_) => continue,
                _ => {}
            }
            let qs = g.qubits();
            let max_d = qs.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
            let new_d = max_d + 1;
            for &q in &qs {
                qubit_depth[q] = new_d;
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }
}

// ============================================================
// METRICS AND REPORTS
// ============================================================

/// Quantitative metrics for a circuit snapshot.
#[derive(Debug, Clone)]
pub struct CircuitMetrics {
    pub num_qubits: usize,
    pub total_gates: usize,
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub three_qubit_gates: usize,
    pub depth: usize,
    pub cx_count: usize,
    pub t_count: usize,
    pub gate_counts: Vec<(String, usize)>,
}

/// Report for a single optimization pass.
#[derive(Debug, Clone)]
pub struct PassReport {
    pub pass_name: String,
    pub gates_before: usize,
    pub gates_after: usize,
    pub gates_removed: usize,
    pub time_us: f64,
}

/// Summary of improvement percentages.
#[derive(Debug, Clone)]
pub struct ImprovementSummary {
    pub gate_reduction_pct: f64,
    pub depth_reduction_pct: f64,
    pub cx_reduction_pct: f64,
    pub t_reduction_pct: f64,
    /// Weighted improvement score: 0.4*gate + 0.3*depth + 0.2*cx + 0.1*t.
    pub total_improvement_score: f64,
}

/// Comprehensive optimization report.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    pub before: CircuitMetrics,
    pub after: CircuitMetrics,
    pub pass_reports: Vec<PassReport>,
    pub total_time_us: f64,
    pub improvement_summary: ImprovementSummary,
    /// The optimized circuit.
    pub optimized_circuit: OptCircuit,
}

fn reduction_pct(before: usize, after: usize) -> f64 {
    if before == 0 {
        0.0
    } else {
        ((before as f64 - after as f64) / before as f64) * 100.0
    }
}

fn compute_improvement(before: &CircuitMetrics, after: &CircuitMetrics) -> ImprovementSummary {
    let gate_r = reduction_pct(before.total_gates, after.total_gates);
    let depth_r = reduction_pct(before.depth, after.depth);
    let cx_r = reduction_pct(before.cx_count, after.cx_count);
    let t_r = reduction_pct(before.t_count, after.t_count);
    let score = 0.4 * gate_r + 0.3 * depth_r + 0.2 * cx_r + 0.1 * t_r;
    ImprovementSummary {
        gate_reduction_pct: gate_r,
        depth_reduction_pct: depth_r,
        cx_reduction_pct: cx_r,
        t_reduction_pct: t_r,
        total_improvement_score: score,
    }
}

impl fmt::Display for OptimizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Optimization Report ===")?;
        writeln!(
            f,
            "Before: {} gates, depth {}, {} CX, {} T",
            self.before.total_gates, self.before.depth, self.before.cx_count, self.before.t_count
        )?;
        writeln!(
            f,
            "After:  {} gates, depth {}, {} CX, {} T",
            self.after.total_gates, self.after.depth, self.after.cx_count, self.after.t_count
        )?;
        writeln!(f, "---")?;
        for pr in &self.pass_reports {
            writeln!(
                f,
                "  {}: {} -> {} (-{} gates, {:.0}us)",
                pr.pass_name, pr.gates_before, pr.gates_after, pr.gates_removed, pr.time_us
            )?;
        }
        writeln!(f, "---")?;
        writeln!(
            f,
            "Improvement: {:.1}% gates, {:.1}% depth, {:.1}% CX, {:.1}% T (score {:.1})",
            self.improvement_summary.gate_reduction_pct,
            self.improvement_summary.depth_reduction_pct,
            self.improvement_summary.cx_reduction_pct,
            self.improvement_summary.t_reduction_pct,
            self.improvement_summary.total_improvement_score,
        )?;
        writeln!(f, "Total time: {:.0} us", self.total_time_us)?;
        Ok(())
    }
}

// ============================================================
// OPTIMIZATION PASSES
// ============================================================

/// An optimization pass the pipeline can apply.
#[derive(Debug, Clone)]
pub enum OptimizationPass {
    /// Cancel adjacent inverse gate pairs (HH, XX, SS-dagger, TT-dagger).
    GateCancellation,
    /// Fuse consecutive single-qubit gates on the same qubit into a U3.
    SingleQubitFusion,
    /// Reorder commuting gates to expose further cancellation.
    CommutationAnalysis,
    /// Optimize CX direction for target connectivity.
    CxDirectionOptimization,
    /// Replace known gate patterns with shorter equivalents.
    TemplateMatching,
    /// Decompose multi-qubit gates into CX + single-qubit gates.
    TwoQubitDecomposition,
    /// Map logical qubits to physical qubits.
    LayoutMapping,
    /// Insert SWAP gates for non-adjacent two-qubit interactions.
    SwapInsertion,
    /// Sliding-window peephole optimization.
    PeepholeOptimization { window: usize },
    /// Strip all barrier instructions.
    RemoveBarriers,
    /// Remove gates equivalent to identity.
    RemoveIdentity,
}

impl OptimizationPass {
    pub fn name(&self) -> &str {
        match self {
            OptimizationPass::GateCancellation => "GateCancellation",
            OptimizationPass::SingleQubitFusion => "SingleQubitFusion",
            OptimizationPass::CommutationAnalysis => "CommutationAnalysis",
            OptimizationPass::CxDirectionOptimization => "CxDirectionOptimization",
            OptimizationPass::TemplateMatching => "TemplateMatching",
            OptimizationPass::TwoQubitDecomposition => "TwoQubitDecomposition",
            OptimizationPass::LayoutMapping => "LayoutMapping",
            OptimizationPass::SwapInsertion => "SwapInsertion",
            OptimizationPass::PeepholeOptimization { .. } => "PeepholeOptimization",
            OptimizationPass::RemoveBarriers => "RemoveBarriers",
            OptimizationPass::RemoveIdentity => "RemoveIdentity",
        }
    }
}

// ============================================================
// OPTIMIZATION LEVEL
// ============================================================

/// Preset optimization levels (modeled after Qiskit 0-3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Level 0: No optimization, just validate.
    Level0,
    /// Level 1: Gate cancellation and single-qubit fusion.
    Level1,
    /// Level 2: Level 1 + commutation analysis and template matching.
    Level2,
    /// Level 3: Level 2 + peephole, two-qubit decomposition, remove barriers.
    Level3,
}

impl OptimizationLevel {
    /// Build the pass sequence for this level.
    pub fn passes(&self) -> Vec<OptimizationPass> {
        match self {
            OptimizationLevel::Level0 => vec![],
            OptimizationLevel::Level1 => vec![
                OptimizationPass::RemoveIdentity,
                OptimizationPass::GateCancellation,
                OptimizationPass::SingleQubitFusion,
            ],
            OptimizationLevel::Level2 => vec![
                OptimizationPass::RemoveIdentity,
                OptimizationPass::GateCancellation,
                OptimizationPass::CommutationAnalysis,
                OptimizationPass::GateCancellation,
                OptimizationPass::SingleQubitFusion,
                OptimizationPass::TemplateMatching,
            ],
            OptimizationLevel::Level3 => vec![
                OptimizationPass::RemoveBarriers,
                OptimizationPass::RemoveIdentity,
                OptimizationPass::TwoQubitDecomposition,
                OptimizationPass::GateCancellation,
                OptimizationPass::CommutationAnalysis,
                OptimizationPass::GateCancellation,
                OptimizationPass::SingleQubitFusion,
                OptimizationPass::TemplateMatching,
                OptimizationPass::PeepholeOptimization { window: 4 },
                OptimizationPass::GateCancellation,
                OptimizationPass::SingleQubitFusion,
            ],
        }
    }
}

// ============================================================
// OPTIMIZER CONFIG
// ============================================================

/// Configuration for the circuit optimizer.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub level: OptimizationLevel,
    /// Target native gate set, e.g. `["cx", "rz", "sx"]`.
    pub target_basis: Vec<String>,
    /// Physical device connectivity (pairs of connected qubits).
    pub connectivity: Option<Vec<(usize, usize)>>,
    /// Custom pass list (overrides level-derived passes when `Some`).
    pub passes: Option<Vec<OptimizationPass>>,
    /// Maximum iterations for convergent passes.
    pub max_iterations: usize,
}

impl OptimizerConfig {
    /// Create a config from a preset level with default settings.
    pub fn level(level: OptimizationLevel) -> Self {
        OptimizerConfig {
            level,
            target_basis: vec![
                "cx".into(),
                "rz".into(),
                "ry".into(),
                "u3".into(),
                "h".into(),
                "x".into(),
            ],
            connectivity: None,
            passes: None,
            max_iterations: 10,
        }
    }

    /// Get the effective pass list.
    pub fn effective_passes(&self) -> Vec<OptimizationPass> {
        self.passes.clone().unwrap_or_else(|| self.level.passes())
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::level(OptimizationLevel::Level1)
    }
}

// ============================================================
// TEMPLATE
// ============================================================

/// A circuit template for pattern-matching optimization.
#[derive(Debug, Clone)]
pub struct CircuitTemplate {
    pub pattern: Vec<OptGate>,
    pub replacement: Vec<OptGate>,
    pub name: String,
}

/// Pre-built template libraries.
pub struct TemplateLibrary;

impl TemplateLibrary {
    /// Standard identities: HH=I, XX=I, SWAP=3CX, CX-CX=I.
    pub fn standard() -> Vec<CircuitTemplate> {
        vec![
            CircuitTemplate {
                name: "HH=I".into(),
                pattern: vec![OptGate::H(0), OptGate::H(0)],
                replacement: vec![],
            },
            CircuitTemplate {
                name: "XX=I".into(),
                pattern: vec![OptGate::X(0), OptGate::X(0)],
                replacement: vec![],
            },
            CircuitTemplate {
                name: "ZZ=I".into(),
                pattern: vec![OptGate::Z(0), OptGate::Z(0)],
                replacement: vec![],
            },
            CircuitTemplate {
                name: "CX-CX=I".into(),
                pattern: vec![OptGate::CX(0, 1), OptGate::CX(0, 1)],
                replacement: vec![],
            },
            CircuitTemplate {
                name: "SWAP=3CX".into(),
                pattern: vec![OptGate::Swap(0, 1)],
                replacement: vec![OptGate::CX(0, 1), OptGate::CX(1, 0), OptGate::CX(0, 1)],
            },
        ]
    }

    /// Clifford simplification templates.
    pub fn clifford() -> Vec<CircuitTemplate> {
        let mut t = Self::standard();
        t.push(CircuitTemplate {
            name: "S-Sdg=I".into(),
            pattern: vec![OptGate::S(0), OptGate::Sdg(0)],
            replacement: vec![],
        });
        t.push(CircuitTemplate {
            name: "Sdg-S=I".into(),
            pattern: vec![OptGate::Sdg(0), OptGate::S(0)],
            replacement: vec![],
        });
        t.push(CircuitTemplate {
            name: "SS=Z".into(),
            pattern: vec![OptGate::S(0), OptGate::S(0)],
            replacement: vec![OptGate::Z(0)],
        });
        t.push(CircuitTemplate {
            name: "HZH=X".into(),
            pattern: vec![OptGate::H(0), OptGate::Z(0), OptGate::H(0)],
            replacement: vec![OptGate::X(0)],
        });
        t.push(CircuitTemplate {
            name: "HXH=Z".into(),
            pattern: vec![OptGate::H(0), OptGate::X(0), OptGate::H(0)],
            replacement: vec![OptGate::Z(0)],
        });
        t
    }

    /// T-gate count reduction templates.
    pub fn t_optimization() -> Vec<CircuitTemplate> {
        let mut t = Self::clifford();
        t.push(CircuitTemplate {
            name: "T-Tdg=I".into(),
            pattern: vec![OptGate::T(0), OptGate::Tdg(0)],
            replacement: vec![],
        });
        t.push(CircuitTemplate {
            name: "Tdg-T=I".into(),
            pattern: vec![OptGate::Tdg(0), OptGate::T(0)],
            replacement: vec![],
        });
        t.push(CircuitTemplate {
            name: "TT=S".into(),
            pattern: vec![OptGate::T(0), OptGate::T(0)],
            replacement: vec![OptGate::S(0)],
        });
        t.push(CircuitTemplate {
            name: "Tdg-Tdg=Sdg".into(),
            pattern: vec![OptGate::Tdg(0), OptGate::Tdg(0)],
            replacement: vec![OptGate::Sdg(0)],
        });
        t
    }
}

// ============================================================
// PASS IMPLEMENTATIONS
// ============================================================

/// Remove all barrier instructions.
fn pass_remove_barriers(gates: &[OptGate]) -> Vec<OptGate> {
    gates
        .iter()
        .filter(|g| !matches!(g, OptGate::Barrier(_)))
        .cloned()
        .collect()
}

/// Remove gates that are equivalent to identity (rotation by ~0).
fn pass_remove_identity(gates: &[OptGate]) -> Vec<OptGate> {
    let tol = 1e-10;
    gates
        .iter()
        .filter(|g| match g {
            OptGate::Rx(_, a) | OptGate::Ry(_, a) | OptGate::Rz(_, a) => a.abs() > tol,
            OptGate::U3(_, t, p, l) => t.abs() > tol || p.abs() > tol || l.abs() > tol,
            _ => true,
        })
        .cloned()
        .collect()
}

/// Cancel adjacent inverse gate pairs.
/// Iterates until no more cancellations are found.
fn pass_gate_cancellation(gates: &[OptGate]) -> Vec<OptGate> {
    let mut result: Vec<OptGate> = gates.to_vec();
    let mut changed = true;

    while changed {
        changed = false;
        let mut new_result: Vec<OptGate> = Vec::with_capacity(result.len());
        let mut skip_next = false;

        for i in 0..result.len() {
            if skip_next {
                skip_next = false;
                continue;
            }
            if i + 1 < result.len() && gates_cancel(&result[i], &result[i + 1]) {
                skip_next = true;
                changed = true;
                continue;
            }
            new_result.push(result[i].clone());
        }
        result = new_result;
    }
    result
}

/// Check whether two gates cancel (compose to identity).
fn gates_cancel(a: &OptGate, b: &OptGate) -> bool {
    match (a, b) {
        // Self-inverse single-qubit
        (OptGate::H(q1), OptGate::H(q2)) if q1 == q2 => true,
        (OptGate::X(q1), OptGate::X(q2)) if q1 == q2 => true,
        (OptGate::Y(q1), OptGate::Y(q2)) if q1 == q2 => true,
        (OptGate::Z(q1), OptGate::Z(q2)) if q1 == q2 => true,
        // S and S-dagger
        (OptGate::S(q1), OptGate::Sdg(q2)) if q1 == q2 => true,
        (OptGate::Sdg(q1), OptGate::S(q2)) if q1 == q2 => true,
        // T and T-dagger
        (OptGate::T(q1), OptGate::Tdg(q2)) if q1 == q2 => true,
        (OptGate::Tdg(q1), OptGate::T(q2)) if q1 == q2 => true,
        // Self-inverse two-qubit
        (OptGate::CX(a1, b1), OptGate::CX(a2, b2)) if a1 == a2 && b1 == b2 => true,
        (OptGate::CZ(a1, b1), OptGate::CZ(a2, b2)) if a1 == a2 && b1 == b2 => true,
        (OptGate::Swap(a1, b1), OptGate::Swap(a2, b2)) if a1 == a2 && b1 == b2 => true,
        // Rotation inverse: Rz(a) Rz(-a) etc.
        (OptGate::Rx(q1, a1), OptGate::Rx(q2, a2)) if q1 == q2 && (a1 + a2).abs() < 1e-10 => true,
        (OptGate::Ry(q1, a1), OptGate::Ry(q2, a2)) if q1 == q2 && (a1 + a2).abs() < 1e-10 => true,
        (OptGate::Rz(q1, a1), OptGate::Rz(q2, a2)) if q1 == q2 && (a1 + a2).abs() < 1e-10 => true,
        _ => false,
    }
}

/// Fuse consecutive single-qubit gates on the same qubit into minimal gate sequences.
fn pass_single_qubit_fusion(gates: &[OptGate]) -> Vec<OptGate> {
    if gates.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<OptGate> = Vec::new();
    let mut i = 0;

    while i < gates.len() {
        if !gates[i].is_single_qubit_unitary() {
            result.push(gates[i].clone());
            i += 1;
            continue;
        }

        let q = gates[i].qubits()[0];
        let mut run_end = i + 1;

        // Collect a run of consecutive single-qubit unitaries on the same qubit.
        while run_end < gates.len()
            && gates[run_end].is_single_qubit_unitary()
            && gates[run_end].qubits()[0] == q
        {
            run_end += 1;
        }

        let run_len = run_end - i;
        if run_len < 2 {
            result.push(gates[i].clone());
            i += 1;
            continue;
        }

        // Multiply the 2x2 matrices together.
        let mut product = mat2_identity();
        for j in i..run_end {
            if let Some(u) = gates[j].unitary_2x2() {
                product = mat2_mul(&u, &product);
            }
        }

        // If product is identity (up to global phase), skip entirely.
        if mat2_is_identity(&product, 1e-8) {
            i = run_end;
            continue;
        }

        // ZYZ decompose.
        let (_alpha, beta, gamma, delta) = zyz_decompose(&product);
        let fused = zyz_to_gates(q, beta, gamma, delta);

        if fused.is_empty() {
            // Identity -- skip.
        } else {
            result.extend(fused);
        }
        i = run_end;
    }
    result
}

/// Check if two gates commute (can swap order without changing result).
fn gates_commute(a: &OptGate, b: &OptGate) -> bool {
    let qa = a.qubits();
    let qb = b.qubits();

    // No overlap in qubits => trivially commute.
    if qa.iter().all(|q| !qb.contains(q)) {
        return true;
    }

    // Diagonal gates commute with each other.
    let is_diag = |g: &OptGate| -> bool {
        matches!(
            g,
            OptGate::Z(_)
                | OptGate::S(_)
                | OptGate::Sdg(_)
                | OptGate::T(_)
                | OptGate::Tdg(_)
                | OptGate::Rz(..)
                | OptGate::CZ(..)
        )
    };

    if is_diag(a) && is_diag(b) {
        return true;
    }

    // Rz on control qubit commutes with CX.
    match (a, b) {
        (OptGate::Rz(q, _), OptGate::CX(ctrl, _)) | (OptGate::CX(ctrl, _), OptGate::Rz(q, _))
            if q == ctrl =>
        {
            return true;
        }
        (OptGate::Z(q), OptGate::CX(ctrl, _)) | (OptGate::CX(ctrl, _), OptGate::Z(q))
            if q == ctrl =>
        {
            return true;
        }
        // X on target commutes with CX.
        (OptGate::X(q), OptGate::CX(_, tgt)) | (OptGate::CX(_, tgt), OptGate::X(q)) if q == tgt => {
            return true;
        }
        _ => {}
    }

    false
}

/// Commutation analysis: move commuting gates past each other to expose
/// cancellation or fusion opportunities.
fn pass_commutation_analysis(gates: &[OptGate]) -> Vec<OptGate> {
    let mut result = gates.to_vec();
    let mut changed = true;
    let mut iters = 0;

    while changed && iters < 20 {
        changed = false;
        iters += 1;

        for i in 0..result.len().saturating_sub(1) {
            // Only try swapping if it brings cancellable or fusable pairs together.
            if !gates_commute(&result[i], &result[i + 1]) {
                continue;
            }
            // Check if swapping brings a potential match with neighbors.
            let swap_benefit = check_swap_benefit(&result, i);
            if swap_benefit {
                result.swap(i, i + 1);
                changed = true;
            }
        }
    }
    result
}

/// Heuristic: would swapping gates at positions i and i+1 bring matching
/// pairs together (for cancellation or fusion)?
fn check_swap_benefit(gates: &[OptGate], i: usize) -> bool {
    let gi = &gates[i];
    let gi1 = &gates[i + 1];

    // Would gi1 cancel or fuse with the gate before it (at i-1)?
    if i > 0 {
        if gates_cancel(&gates[i - 1], gi1) {
            return true;
        }
        if gates[i - 1].is_single_qubit_unitary()
            && gi1.is_single_qubit_unitary()
            && gates[i - 1].qubits() == gi1.qubits()
        {
            return true;
        }
    }

    // Would gi cancel or fuse with the gate after i+1 (at i+2)?
    if i + 2 < gates.len() {
        if gates_cancel(gi, &gates[i + 2]) {
            return true;
        }
        if gi.is_single_qubit_unitary()
            && gates[i + 2].is_single_qubit_unitary()
            && gi.qubits() == gates[i + 2].qubits()
        {
            return true;
        }
    }

    false
}

/// CX direction optimization: if the circuit has CX(a,b) but the device
/// only supports CX(b,a), insert H gates to flip direction.
fn pass_cx_direction(
    gates: &[OptGate],
    connectivity: &Option<Vec<(usize, usize)>>,
) -> Vec<OptGate> {
    let conn = match connectivity {
        Some(c) => c,
        None => return gates.to_vec(),
    };

    let allowed: HashSet<(usize, usize)> = conn.iter().cloned().collect();

    gates
        .iter()
        .flat_map(|g| match g {
            OptGate::CX(ctrl, tgt) => {
                if allowed.contains(&(*ctrl, *tgt)) || allowed.is_empty() {
                    vec![g.clone()]
                } else if allowed.contains(&(*tgt, *ctrl)) {
                    // Flip CX direction: H-CX(tgt,ctrl)-H
                    vec![
                        OptGate::H(*ctrl),
                        OptGate::H(*tgt),
                        OptGate::CX(*tgt, *ctrl),
                        OptGate::H(*tgt),
                        OptGate::H(*ctrl),
                    ]
                } else {
                    vec![g.clone()] // Leave as-is; routing pass will handle
                }
            }
            _ => vec![g.clone()],
        })
        .collect()
}

/// Template matching: scan the circuit for known patterns and replace them.
fn pass_template_matching(gates: &[OptGate], templates: &[CircuitTemplate]) -> Vec<OptGate> {
    let mut result = gates.to_vec();

    for template in templates {
        let plen = template.pattern.len();
        if plen == 0 || result.len() < plen {
            continue;
        }

        let mut i = 0;
        let mut new_result = Vec::with_capacity(result.len());

        while i < result.len() {
            if i + plen <= result.len() {
                if let Some(mapping) = match_template(&result[i..i + plen], &template.pattern) {
                    // Apply the replacement with the qubit mapping.
                    for rg in &template.replacement {
                        new_result.push(remap_gate(rg, &mapping));
                    }
                    i += plen;
                    continue;
                }
            }
            new_result.push(result[i].clone());
            i += 1;
        }
        result = new_result;
    }
    result
}

/// Try to match a circuit slice against a template pattern.
/// Template uses abstract qubit indices (0, 1, 2...) that map to actual qubits.
/// Returns the qubit mapping if match succeeds.
fn match_template(slice: &[OptGate], pattern: &[OptGate]) -> Option<HashMap<usize, usize>> {
    if slice.len() != pattern.len() {
        return None;
    }

    let mut mapping: HashMap<usize, usize> = HashMap::new();

    for (sg, pg) in slice.iter().zip(pattern.iter()) {
        if std::mem::discriminant(sg) != std::mem::discriminant(pg) {
            // Allow matching gate types with different parameters -- check structural match.
            if !gates_structurally_match(sg, pg) {
                return None;
            }
        }
        let sq = sg.qubits();
        let pq = pg.qubits();
        if sq.len() != pq.len() {
            return None;
        }
        for (actual, abstract_q) in sq.iter().zip(pq.iter()) {
            match mapping.get(abstract_q) {
                Some(&mapped) => {
                    if mapped != *actual {
                        return None;
                    }
                }
                None => {
                    mapping.insert(*abstract_q, *actual);
                }
            }
        }
    }
    Some(mapping)
}

/// Check if two gates match structurally (same gate kind, ignoring qubit indices).
fn gates_structurally_match(a: &OptGate, b: &OptGate) -> bool {
    // Match by name (gate type).
    a.name() == b.name()
}

/// Remap a gate's qubit indices using a mapping.
fn remap_gate(gate: &OptGate, mapping: &HashMap<usize, usize>) -> OptGate {
    let remap = |q: usize| -> usize { *mapping.get(&q).unwrap_or(&q) };

    match gate {
        OptGate::H(q) => OptGate::H(remap(*q)),
        OptGate::X(q) => OptGate::X(remap(*q)),
        OptGate::Y(q) => OptGate::Y(remap(*q)),
        OptGate::Z(q) => OptGate::Z(remap(*q)),
        OptGate::S(q) => OptGate::S(remap(*q)),
        OptGate::Sdg(q) => OptGate::Sdg(remap(*q)),
        OptGate::T(q) => OptGate::T(remap(*q)),
        OptGate::Tdg(q) => OptGate::Tdg(remap(*q)),
        OptGate::Rx(q, a) => OptGate::Rx(remap(*q), *a),
        OptGate::Ry(q, a) => OptGate::Ry(remap(*q), *a),
        OptGate::Rz(q, a) => OptGate::Rz(remap(*q), *a),
        OptGate::CX(a, b) => OptGate::CX(remap(*a), remap(*b)),
        OptGate::CZ(a, b) => OptGate::CZ(remap(*a), remap(*b)),
        OptGate::Swap(a, b) => OptGate::Swap(remap(*a), remap(*b)),
        OptGate::Toffoli(a, b, c) => OptGate::Toffoli(remap(*a), remap(*b), remap(*c)),
        OptGate::U3(q, t, p, l) => OptGate::U3(remap(*q), *t, *p, *l),
        OptGate::Barrier(qs) => OptGate::Barrier(qs.iter().map(|q| remap(*q)).collect()),
        OptGate::Measure(q) => OptGate::Measure(remap(*q)),
    }
}

/// Two-qubit decomposition: decompose Toffoli and SWAP into CX + single-qubit gates.
fn pass_two_qubit_decomposition(gates: &[OptGate]) -> Vec<OptGate> {
    let mut result = Vec::new();
    for g in gates {
        match g {
            OptGate::Toffoli(a, b, c) => {
                // Standard Toffoli decomposition into 6 CX + single-qubit gates.
                result.push(OptGate::H(*c));
                result.push(OptGate::CX(*b, *c));
                result.push(OptGate::Tdg(*c));
                result.push(OptGate::CX(*a, *c));
                result.push(OptGate::T(*c));
                result.push(OptGate::CX(*b, *c));
                result.push(OptGate::Tdg(*c));
                result.push(OptGate::CX(*a, *c));
                result.push(OptGate::T(*b));
                result.push(OptGate::T(*c));
                result.push(OptGate::H(*c));
                result.push(OptGate::CX(*a, *b));
                result.push(OptGate::T(*a));
                result.push(OptGate::Tdg(*b));
                result.push(OptGate::CX(*a, *b));
            }
            OptGate::Swap(a, b) => {
                // SWAP = 3 CX
                result.push(OptGate::CX(*a, *b));
                result.push(OptGate::CX(*b, *a));
                result.push(OptGate::CX(*a, *b));
            }
            _ => result.push(g.clone()),
        }
    }
    result
}

/// Layout mapping: map logical qubits to physical qubits via a greedy heuristic.
/// Returns (remapped gates, logical_to_physical mapping).
fn pass_layout_mapping(
    gates: &[OptGate],
    num_qubits: usize,
    connectivity: &Option<Vec<(usize, usize)>>,
) -> (Vec<OptGate>, Vec<usize>) {
    // If no connectivity constraints, identity mapping.
    let mapping: Vec<usize> = (0..num_qubits).collect();
    let conn = match connectivity {
        Some(c) => c,
        None => return (gates.to_vec(), mapping),
    };

    if conn.is_empty() {
        return (gates.to_vec(), mapping);
    }

    // Build adjacency from connectivity.
    let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
    for &(a, b) in conn {
        adj.entry(a).or_default().insert(b);
        adj.entry(b).or_default().insert(a);
    }

    // Greedy initial placement: assign most-used logical qubits to
    // most-connected physical qubits.
    let mut qubit_usage: Vec<(usize, usize)> = (0..num_qubits).map(|q| (q, 0)).collect();
    for g in gates {
        if g.is_two_qubit() {
            let qs = g.qubits();
            qubit_usage[qs[0]].1 += 1;
            qubit_usage[qs[1]].1 += 1;
        }
    }
    qubit_usage.sort_by(|a, b| b.1.cmp(&a.1));

    let mut phys_degree: Vec<(usize, usize)> = (0..num_qubits)
        .map(|q| (q, adj.get(&q).map_or(0, |s| s.len())))
        .collect();
    phys_degree.sort_by(|a, b| b.1.cmp(&a.1));

    let mut log_to_phys = vec![0usize; num_qubits];
    let mut used_phys: HashSet<usize> = HashSet::new();

    for (log_q, _usage) in &qubit_usage {
        for (phys_q, _deg) in &phys_degree {
            if !used_phys.contains(phys_q) {
                log_to_phys[*log_q] = *phys_q;
                used_phys.insert(*phys_q);
                break;
            }
        }
    }

    // Remap all gates.
    let mut mapping_map: HashMap<usize, usize> = HashMap::new();
    for (i, &p) in log_to_phys.iter().enumerate() {
        mapping_map.insert(i, p);
    }

    let remapped: Vec<OptGate> = gates.iter().map(|g| remap_gate(g, &mapping_map)).collect();
    (remapped, log_to_phys)
}

/// SWAP insertion: insert SWAP gates where two-qubit gates act on
/// non-adjacent qubits according to the connectivity graph.
fn pass_swap_insertion(
    gates: &[OptGate],
    connectivity: &Option<Vec<(usize, usize)>>,
) -> Vec<OptGate> {
    let conn = match connectivity {
        Some(c) => c,
        None => return gates.to_vec(),
    };

    if conn.is_empty() {
        return gates.to_vec();
    }

    let max_q = conn.iter().flat_map(|&(a, b)| [a, b]).max().unwrap_or(0) + 1;

    // Build adjacency for BFS.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); max_q];
    for &(a, b) in conn {
        if a < max_q && b < max_q {
            adj[a].push(b);
            adj[b].push(a);
        }
    }

    let mut result = Vec::new();
    // Track current permutation of logical -> physical qubits.
    let mut perm: Vec<usize> = (0..max_q).collect();
    let mut inv_perm: Vec<usize> = (0..max_q).collect();

    for g in gates {
        if !g.is_two_qubit() {
            result.push(g.clone());
            continue;
        }

        let qs = g.qubits();
        let phys_a = perm[qs[0].min(max_q - 1)];
        let phys_b = perm[qs[1].min(max_q - 1)];

        if adj[phys_a].contains(&phys_b) || phys_a == phys_b {
            // Already adjacent -- emit directly.
            result.push(g.clone());
            continue;
        }

        // BFS to find shortest path from phys_a to phys_b.
        if let Some(path) = bfs_path(&adj, phys_a, phys_b) {
            // Insert SWAPs along the path to bring phys_a adjacent to phys_b.
            for w in 0..path.len() - 2 {
                let p = path[w];
                let q = path[w + 1];
                // Emit SWAP as 3 CX gates.
                result.push(OptGate::CX(p, q));
                result.push(OptGate::CX(q, p));
                result.push(OptGate::CX(p, q));
                // Update permutation.
                let log_p = inv_perm[p];
                let log_q = inv_perm[q];
                perm[log_p] = q;
                perm[log_q] = p;
                inv_perm[p] = log_q;
                inv_perm[q] = log_p;
            }
            // Now phys_a has been moved adjacent to phys_b; emit the gate.
            result.push(g.clone());
        } else {
            // No path found -- emit as-is (will be caught by validation).
            result.push(g.clone());
        }
    }
    result
}

/// BFS shortest path between two nodes in an adjacency list.
fn bfs_path(adj: &[Vec<usize>], start: usize, end: usize) -> Option<Vec<usize>> {
    if start == end {
        return Some(vec![start]);
    }
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut parent = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    visited[start] = true;
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        for &next in &adj[node] {
            if next < n && !visited[next] {
                visited[next] = true;
                parent[next] = node;
                if next == end {
                    // Reconstruct path.
                    let mut path = vec![end];
                    let mut cur = end;
                    while cur != start {
                        cur = parent[cur];
                        path.push(cur);
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(next);
            }
        }
    }
    None
}

/// Peephole optimization: sliding window of size `window` over the circuit.
/// For each window, try gate cancellation and single-qubit fusion.
fn pass_peephole(gates: &[OptGate], window: usize) -> Vec<OptGate> {
    if gates.is_empty() || window < 2 {
        return gates.to_vec();
    }

    let mut result = gates.to_vec();
    let mut changed = true;
    let mut iters = 0;

    while changed && iters < 20 {
        changed = false;
        iters += 1;
        let mut i = 0;

        while i + window <= result.len() {
            let slice = &result[i..i + window];
            // Try commutation, then cancellation, then fusion on the window.
            let opt0 = pass_commutation_analysis(slice);
            let opt1 = pass_gate_cancellation(&opt0);
            let opt2 = pass_single_qubit_fusion(&opt1);

            if opt2.len() < window {
                // Improvement found.
                let mut new_result = Vec::with_capacity(result.len());
                new_result.extend_from_slice(&result[..i]);
                new_result.extend(opt2);
                new_result.extend_from_slice(&result[i + window..]);
                result = new_result;
                changed = true;
                // Don't advance i -- recheck from same position.
            } else {
                i += 1;
            }
        }
    }
    result
}

// ============================================================
// CIRCUIT OPTIMIZER
// ============================================================

/// The main circuit optimizer engine.
#[derive(Debug, Clone)]
pub struct CircuitOptimizer {
    pub config: OptimizerConfig,
    pub templates: Vec<CircuitTemplate>,
}

impl CircuitOptimizer {
    /// Create an optimizer with the given config and default templates.
    pub fn new(config: OptimizerConfig) -> Self {
        let templates = match config.level {
            OptimizationLevel::Level0 | OptimizationLevel::Level1 => TemplateLibrary::standard(),
            OptimizationLevel::Level2 => TemplateLibrary::clifford(),
            OptimizationLevel::Level3 => TemplateLibrary::t_optimization(),
        };
        CircuitOptimizer { config, templates }
    }

    /// Create an optimizer with custom templates.
    pub fn with_templates(config: OptimizerConfig, templates: Vec<CircuitTemplate>) -> Self {
        CircuitOptimizer { config, templates }
    }

    /// Run the full optimization pipeline and produce a report.
    pub fn optimize(&self, circuit: &OptCircuit) -> Result<OptimizationReport, OptimizerError> {
        circuit.validate()?;

        let overall_start = Instant::now();
        let before = circuit.metrics();

        let passes = self.config.effective_passes();
        let mut current_gates = circuit.gates.clone();
        let mut pass_reports = Vec::new();

        for pass in &passes {
            let pass_start = Instant::now();
            let gates_before = count_unitary_gates(&current_gates);

            current_gates = self.apply_pass(pass, &current_gates, circuit.num_qubits)?;

            let gates_after = count_unitary_gates(&current_gates);
            let elapsed = pass_start.elapsed().as_nanos() as f64 / 1000.0;

            pass_reports.push(PassReport {
                pass_name: pass.name().to_string(),
                gates_before,
                gates_after,
                gates_removed: gates_before.saturating_sub(gates_after),
                time_us: elapsed,
            });
        }

        let optimized = OptCircuit {
            num_qubits: circuit.num_qubits,
            gates: current_gates,
        };
        let after = optimized.metrics();
        let total_time_us = overall_start.elapsed().as_nanos() as f64 / 1000.0;
        let improvement_summary = compute_improvement(&before, &after);

        Ok(OptimizationReport {
            before,
            after,
            pass_reports,
            total_time_us,
            improvement_summary,
            optimized_circuit: optimized,
        })
    }

    /// Apply a single optimization pass.
    fn apply_pass(
        &self,
        pass: &OptimizationPass,
        gates: &[OptGate],
        num_qubits: usize,
    ) -> Result<Vec<OptGate>, OptimizerError> {
        let result = match pass {
            OptimizationPass::GateCancellation => pass_gate_cancellation(gates),
            OptimizationPass::SingleQubitFusion => pass_single_qubit_fusion(gates),
            OptimizationPass::CommutationAnalysis => pass_commutation_analysis(gates),
            OptimizationPass::CxDirectionOptimization => {
                pass_cx_direction(gates, &self.config.connectivity)
            }
            OptimizationPass::TemplateMatching => pass_template_matching(gates, &self.templates),
            OptimizationPass::TwoQubitDecomposition => pass_two_qubit_decomposition(gates),
            OptimizationPass::LayoutMapping => {
                let (remapped, _mapping) =
                    pass_layout_mapping(gates, num_qubits, &self.config.connectivity);
                remapped
            }
            OptimizationPass::SwapInsertion => {
                pass_swap_insertion(gates, &self.config.connectivity)
            }
            OptimizationPass::PeepholeOptimization { window } => pass_peephole(gates, *window),
            OptimizationPass::RemoveBarriers => pass_remove_barriers(gates),
            OptimizationPass::RemoveIdentity => pass_remove_identity(gates),
        };
        Ok(result)
    }
}

/// Count gates that are unitaries (exclude barriers and measurements).
fn count_unitary_gates(gates: &[OptGate]) -> usize {
    gates
        .iter()
        .filter(|g| !matches!(g, OptGate::Barrier(_) | OptGate::Measure(_)))
        .count()
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // --- Test 1: Circuit creation ---
    #[test]
    fn test_circuit_creation() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        c.push(OptGate::Measure(2));
        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.gates.len(), 3);
    }

    // --- Test 2: Gate counting per type ---
    #[test]
    fn test_gate_counting() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0));
        c.push(OptGate::H(1));
        c.push(OptGate::CX(0, 1));
        c.push(OptGate::T(2));
        c.push(OptGate::CX(1, 2));
        c.push(OptGate::Toffoli(0, 1, 2));
        let m = c.metrics();
        assert_eq!(m.single_qubit_gates, 3); // 2 H + 1 T
        assert_eq!(m.two_qubit_gates, 2); // 2 CX
        assert_eq!(m.three_qubit_gates, 1); // 1 Toffoli
        assert_eq!(m.total_gates, 6);
        assert_eq!(m.cx_count, 2);
        assert_eq!(m.t_count, 1);
    }

    // --- Test 3: Depth of a linear circuit ---
    #[test]
    fn test_depth_linear() {
        let mut c = OptCircuit::new(1);
        c.push(OptGate::H(0));
        c.push(OptGate::X(0));
        c.push(OptGate::Z(0));
        assert_eq!(c.compute_depth(), 3);
    }

    // --- Test 4: Depth of a parallel circuit ---
    #[test]
    fn test_depth_parallel() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0));
        c.push(OptGate::H(1));
        c.push(OptGate::H(2));
        // All three H gates can execute in parallel => depth 1.
        assert_eq!(c.compute_depth(), 1);
    }

    // --- Test 5: Gate cancellation HH ---
    #[test]
    fn test_cancel_hh() {
        let gates = vec![OptGate::H(0), OptGate::H(0)];
        let result = pass_gate_cancellation(&gates);
        assert!(result.is_empty(), "HH should cancel to identity");
    }

    // --- Test 6: Gate cancellation XX ---
    #[test]
    fn test_cancel_xx() {
        let gates = vec![OptGate::X(0), OptGate::X(0)];
        let result = pass_gate_cancellation(&gates);
        assert!(result.is_empty(), "XX should cancel to identity");
    }

    // --- Test 7: Gate cancellation SS-dagger ---
    #[test]
    fn test_cancel_ss_dagger() {
        let gates = vec![OptGate::S(0), OptGate::Sdg(0)];
        let result = pass_gate_cancellation(&gates);
        assert!(result.is_empty(), "S.Sdg should cancel to identity");
    }

    // --- Test 8: Nested cancellation H.X.X.H ---
    #[test]
    fn test_cancel_nested() {
        let gates = vec![OptGate::H(0), OptGate::X(0), OptGate::X(0), OptGate::H(0)];
        let result = pass_gate_cancellation(&gates);
        assert!(
            result.is_empty(),
            "H.X.X.H should fully cancel (XX first, then HH)"
        );
    }

    // --- Test 9: Single-qubit fusion Rz.Rz ---
    #[test]
    fn test_fusion_rz_rz() {
        let gates = vec![OptGate::Rz(0, 0.3), OptGate::Rz(0, 0.5)];
        let result = pass_single_qubit_fusion(&gates);
        // Two Rz should fuse into a single rotation.
        assert!(
            result.len() <= 2,
            "Rz.Rz should fuse (got {} gates)",
            result.len()
        );
        // The total rotation angle should be 0.8.
        // After ZYZ decomposition of Rz(0.3)*Rz(0.5)=Rz(0.8),
        // we expect a single Rz.
        assert_eq!(result.len(), 1, "Two Rz should fuse into one Rz");
    }

    // --- Test 10: Single-qubit fusion H.S ---
    #[test]
    fn test_fusion_h_s() {
        let gates = vec![OptGate::H(0), OptGate::S(0)];
        let result = pass_single_qubit_fusion(&gates);
        // Should fuse into at most 3 gates (ZYZ) but fewer than 2 inputs is
        // not guaranteed; the point is it merges into a single U3-equivalent.
        assert!(
            result.len() <= 3,
            "H.S should fuse into at most 3 gates, got {}",
            result.len()
        );
    }

    // --- Test 11: Three gates merged ---
    #[test]
    fn test_fusion_three_gates() {
        let gates = vec![OptGate::H(0), OptGate::S(0), OptGate::H(0)];
        let result = pass_single_qubit_fusion(&gates);
        // H.S.H is a well-known identity (equals Sdg up to phase).
        // Should be at most 3 ZYZ components.
        assert!(
            result.len() <= 3,
            "H.S.H should fuse into <= 3 gates, got {}",
            result.len()
        );
    }

    // --- Test 12: Rz commutes past Rz ---
    #[test]
    fn test_commutation_rz_rz() {
        assert!(gates_commute(&OptGate::Rz(0, 0.5), &OptGate::Rz(0, 0.3)));
    }

    // --- Test 13: Commutation enables additional cancellation ---
    #[test]
    fn test_commutation_enables_cancellation() {
        // H(0), Rz(1, 0.5), H(0) -- the Rz is on a different qubit so H.H
        // should be brought together.
        let gates = vec![OptGate::H(0), OptGate::Rz(1, 0.5), OptGate::H(0)];
        let commuted = pass_commutation_analysis(&gates);
        let cancelled = pass_gate_cancellation(&commuted);
        // After commutation the two H(0) are adjacent and cancel.
        assert!(
            cancelled.len() <= 1,
            "Commutation should expose HH cancellation, got {:?}",
            cancelled
        );
    }

    // --- Test 14: Template matching detects SWAP ---
    #[test]
    fn test_template_swap_detected() {
        let templates = TemplateLibrary::standard();
        let gates = vec![OptGate::Swap(0, 1)];
        let result = pass_template_matching(&gates, &templates);
        // SWAP should be replaced by 3 CX.
        let cx_count = result
            .iter()
            .filter(|g| matches!(g, OptGate::CX(..)))
            .count();
        assert_eq!(cx_count, 3, "SWAP should decompose into 3 CX gates");
    }

    // --- Test 15: Template replacement applied ---
    #[test]
    fn test_template_replacement_applied() {
        let templates = TemplateLibrary::standard();
        let gates = vec![OptGate::H(2), OptGate::Swap(0, 1), OptGate::H(2)];
        let result = pass_template_matching(&gates, &templates);
        // Should have H, 3xCX, H (SWAP replaced, H preserved).
        assert!(
            !result.iter().any(|g| matches!(g, OptGate::Swap(..))),
            "No SWAP should remain after template matching"
        );
    }

    // --- Test 16: Two-qubit decomposition CX count <= 3 ---
    #[test]
    fn test_two_qubit_decomp_cx_count() {
        let gates = vec![OptGate::Swap(0, 1)];
        let result = pass_two_qubit_decomposition(&gates);
        let cx_count = result
            .iter()
            .filter(|g| matches!(g, OptGate::CX(..)))
            .count();
        assert!(cx_count <= 3, "SWAP decomposition should use <= 3 CX");
    }

    // --- Test 17: Product state decomposition = 0 CX ---
    #[test]
    fn test_decomp_product_state() {
        // Single-qubit gate: no CX needed.
        let gates = vec![OptGate::H(0)];
        let result = pass_two_qubit_decomposition(&gates);
        let cx_count = result
            .iter()
            .filter(|g| matches!(g, OptGate::CX(..)))
            .count();
        assert_eq!(cx_count, 0, "Single-qubit gate needs 0 CX");
    }

    // --- Test 18: Layout mapping respects connectivity ---
    #[test]
    fn test_layout_mapping_respects_connectivity() {
        let gates = vec![OptGate::CX(0, 1)];
        let connectivity = Some(vec![(0, 1), (1, 2)]);
        let (remapped, mapping) = pass_layout_mapping(&gates, 3, &connectivity);
        // Mapping should produce valid indices.
        assert_eq!(mapping.len(), 3);
        assert!(!remapped.is_empty());
    }

    // --- Test 19: SWAP insertion fixes non-adjacent CX ---
    #[test]
    fn test_swap_insertion_non_adjacent() {
        // Linear chain: 0-1-2. CX(0,2) is non-adjacent.
        let gates = vec![OptGate::CX(0, 2)];
        let connectivity = Some(vec![(0, 1), (1, 2)]);
        let result = pass_swap_insertion(&gates, &connectivity);
        // Should have inserted SWAPs (as CX triplets) plus the original gate.
        assert!(
            result.len() > 1,
            "Non-adjacent CX should trigger SWAP insertion, got {} gates",
            result.len()
        );
    }

    // --- Test 20: Peephole reduces gate count ---
    #[test]
    fn test_peephole_reduces() {
        let gates = vec![OptGate::H(0), OptGate::X(1), OptGate::H(0), OptGate::X(1)];
        let result = pass_peephole(&gates, 4);
        assert!(
            result.len() < gates.len(),
            "Peephole should reduce H._.H and X._.X patterns"
        );
    }

    // --- Test 21: Level 0 no optimization ---
    #[test]
    fn test_level0_no_optimization() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        let config = OptimizerConfig::level(OptimizationLevel::Level0);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // Level 0 applies no passes, so circuit is unchanged.
        assert_eq!(report.after.total_gates, report.before.total_gates);
    }

    // --- Test 22: Level 1 light optimization ---
    #[test]
    fn test_level1_light() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // HH cancels, leaving just CX.
        assert_eq!(report.after.total_gates, 1);
    }

    // --- Test 23: Level 2 medium optimization ---
    #[test]
    fn test_level2_medium() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(0));
        c.push(OptGate::Rz(1, 0.5));
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        let config = OptimizerConfig::level(OptimizationLevel::Level2);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // Level 2 has commutation + cancellation: H(0) commutes past Rz(1)
        // then cancels with H(0).
        assert!(
            report.after.total_gates <= 3,
            "Level 2 should reduce gates via commutation, got {}",
            report.after.total_gates
        );
    }

    // --- Test 24: Level 3 heavy optimization ---
    #[test]
    fn test_level3_heavy() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        c.push(OptGate::Swap(0, 1));
        c.push(OptGate::T(2));
        c.push(OptGate::Tdg(2));
        c.push(OptGate::Barrier(vec![0, 1, 2]));
        let config = OptimizerConfig::level(OptimizationLevel::Level3);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // HH cancels, SWAP decomposes, T.Tdg cancels, barrier removed.
        assert!(
            report.after.total_gates < report.before.total_gates,
            "Level 3 should significantly reduce: before={}, after={}",
            report.before.total_gates,
            report.after.total_gates
        );
    }

    // --- Test 25: Report before metrics correct ---
    #[test]
    fn test_report_before_metrics() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        c.push(OptGate::T(1));
        let config = OptimizerConfig::level(OptimizationLevel::Level0);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        assert_eq!(report.before.total_gates, 3);
        assert_eq!(report.before.single_qubit_gates, 2);
        assert_eq!(report.before.two_qubit_gates, 1);
        assert_eq!(report.before.cx_count, 1);
        assert_eq!(report.before.t_count, 1);
    }

    // --- Test 26: Report after metrics correct ---
    #[test]
    fn test_report_after_metrics() {
        let mut c = OptCircuit::new(1);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        assert_eq!(report.after.total_gates, 0);
        assert_eq!(report.after.depth, 0);
    }

    // --- Test 27: Pass-by-pass details ---
    #[test]
    fn test_pass_by_pass_details() {
        let mut c = OptCircuit::new(1);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // Level 1 has 3 passes: RemoveIdentity, GateCancellation, SingleQubitFusion.
        assert_eq!(report.pass_reports.len(), 3);
        // GateCancellation pass should report gates_removed > 0.
        let cancel_pass = report
            .pass_reports
            .iter()
            .find(|p| p.pass_name == "GateCancellation")
            .expect("Should have GateCancellation pass");
        assert!(
            cancel_pass.gates_removed > 0,
            "GateCancellation should remove HH"
        );
    }

    // --- Test 28: Improvement summary percentages ---
    #[test]
    fn test_improvement_summary() {
        let mut c = OptCircuit::new(2);
        // 4 gates: H, H (cancel), CX, T
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        c.push(OptGate::T(1));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // Before: 4 gates. After: 2 gates (CX + T). Reduction = 50%.
        assert!(
            (report.improvement_summary.gate_reduction_pct - 50.0).abs() < 1.0,
            "Expected ~50% gate reduction, got {:.1}%",
            report.improvement_summary.gate_reduction_pct
        );
    }

    // --- Test 29: Config builder defaults ---
    #[test]
    fn test_config_defaults() {
        let config = OptimizerConfig::default();
        assert_eq!(config.level, OptimizationLevel::Level1);
        assert!(config.connectivity.is_none());
        assert!(config.passes.is_none());
        assert_eq!(config.max_iterations, 10);
        assert!(!config.target_basis.is_empty());
    }

    // --- Test 30: Large circuit 20 qubits 200 gates ---
    #[test]
    fn test_large_circuit() {
        let mut c = OptCircuit::new(20);
        for i in 0..100 {
            let q = i % 20;
            c.push(OptGate::H(q));
            c.push(OptGate::H(q)); // These will cancel.
        }
        assert_eq!(c.gates.len(), 200);
        let config = OptimizerConfig::level(OptimizationLevel::Level2);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // All 200 gates should cancel in pairs.
        assert_eq!(report.after.total_gates, 0);
        assert_eq!(report.improvement_summary.gate_reduction_pct, 100.0);
    }

    // --- Test 31: Validate catches bad qubit index ---
    #[test]
    fn test_validate_bad_qubit() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(5));
        assert!(c.validate().is_err());
    }

    // --- Test 32: Remove barriers pass ---
    #[test]
    fn test_remove_barriers() {
        let gates = vec![
            OptGate::H(0),
            OptGate::Barrier(vec![0, 1]),
            OptGate::CX(0, 1),
        ];
        let result = pass_remove_barriers(&gates);
        assert_eq!(result.len(), 2);
        assert!(!result.iter().any(|g| matches!(g, OptGate::Barrier(_))));
    }

    // --- Test 33: Remove identity rotations ---
    #[test]
    fn test_remove_identity_rotations() {
        let gates = vec![
            OptGate::Rz(0, 0.0),
            OptGate::Ry(0, 0.0),
            OptGate::Rx(0, 0.0),
            OptGate::H(0),
        ];
        let result = pass_remove_identity(&gates);
        assert_eq!(result.len(), 1); // Only H remains.
    }

    // --- Test 34: TT-dagger cancellation ---
    #[test]
    fn test_cancel_tt_dagger() {
        let gates = vec![OptGate::T(0), OptGate::Tdg(0)];
        let result = pass_gate_cancellation(&gates);
        assert!(result.is_empty(), "T.Tdg should cancel");
    }

    // --- Test 35: CX-CX cancellation ---
    #[test]
    fn test_cancel_cx_cx() {
        let gates = vec![OptGate::CX(0, 1), OptGate::CX(0, 1)];
        let result = pass_gate_cancellation(&gates);
        assert!(result.is_empty(), "CX.CX should cancel");
    }

    // --- Test 36: Toffoli decomposition ---
    #[test]
    fn test_toffoli_decomposition() {
        let gates = vec![OptGate::Toffoli(0, 1, 2)];
        let result = pass_two_qubit_decomposition(&gates);
        // Should have CX gates and single-qubit gates, no Toffoli.
        assert!(!result.iter().any(|g| matches!(g, OptGate::Toffoli(..))));
        let cx_count = result
            .iter()
            .filter(|g| matches!(g, OptGate::CX(..)))
            .count();
        assert_eq!(cx_count, 6, "Standard Toffoli decomposition uses 6 CX");
    }

    // --- Test 37: Optimized circuit is stored in report ---
    #[test]
    fn test_optimized_circuit_in_report() {
        let mut c = OptCircuit::new(1);
        c.push(OptGate::X(0));
        c.push(OptGate::X(0));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        assert_eq!(report.optimized_circuit.gates.len(), 0);
        assert_eq!(report.optimized_circuit.num_qubits, 1);
    }

    // --- Test 38: Display formatting ---
    #[test]
    fn test_report_display() {
        let mut c = OptCircuit::new(2);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        let config = OptimizerConfig::level(OptimizationLevel::Level1);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        let display = format!("{}", report);
        assert!(display.contains("Optimization Report"));
        assert!(display.contains("Improvement"));
    }

    // --- Test 39: Custom passes override level ---
    #[test]
    fn test_custom_passes() {
        let mut config = OptimizerConfig::level(OptimizationLevel::Level0);
        config.passes = Some(vec![OptimizationPass::GateCancellation]);
        let opt = CircuitOptimizer::new(config);
        let mut c = OptCircuit::new(1);
        c.push(OptGate::H(0));
        c.push(OptGate::H(0));
        let report = opt.optimize(&c).unwrap();
        // Custom pass should cancel HH even at Level0.
        assert_eq!(report.after.total_gates, 0);
    }

    // --- Test 40: Depth with mixed parallel and serial gates ---
    #[test]
    fn test_depth_mixed() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0)); // depth 1 on q0
        c.push(OptGate::H(1)); // depth 1 on q1 (parallel)
        c.push(OptGate::CX(0, 1)); // depth 2 (depends on both q0, q1)
        c.push(OptGate::H(2)); // depth 1 on q2 (parallel)
        assert_eq!(c.compute_depth(), 2);
    }

    // --- Test 41: Gate name method ---
    #[test]
    fn test_gate_names() {
        assert_eq!(OptGate::H(0).name(), "h");
        assert_eq!(OptGate::CX(0, 1).name(), "cx");
        assert_eq!(OptGate::Toffoli(0, 1, 2).name(), "toffoli");
        assert_eq!(OptGate::U3(0, 1.0, 2.0, 3.0).name(), "u3");
    }

    // --- Test 42: Gate display ---
    #[test]
    fn test_gate_display() {
        let s = format!("{}", OptGate::CX(0, 1));
        assert_eq!(s, "CX q0,q1");
        let s = format!("{}", OptGate::Rz(3, 1.5708));
        assert!(s.contains("Rz"));
        assert!(s.contains("q3"));
    }

    // --- Test 43: Template library sizes ---
    #[test]
    fn test_template_libraries() {
        let std = TemplateLibrary::standard();
        let cliff = TemplateLibrary::clifford();
        let topt = TemplateLibrary::t_optimization();
        assert!(std.len() >= 4);
        assert!(cliff.len() > std.len());
        assert!(topt.len() > cliff.len());
    }

    // --- Test 44: Clifford template HZH=X ---
    #[test]
    fn test_template_hzh_x() {
        let templates = TemplateLibrary::clifford();
        let gates = vec![OptGate::H(0), OptGate::Z(0), OptGate::H(0)];
        let result = pass_template_matching(&gates, &templates);
        assert_eq!(result.len(), 1);
        assert!(matches!(result[0], OptGate::X(0)));
    }

    // --- Test 45: ZYZ decomposition round-trip ---
    #[test]
    fn test_zyz_roundtrip() {
        // Test that ZYZ decomposition of an H gate reconstructs H.
        let h_mat = OptGate::H(0).unitary_2x2().unwrap();
        let (alpha, beta, gamma, delta) = zyz_decompose(&h_mat);
        // Reconstruct: Rz(beta) * Ry(gamma) * Rz(delta)
        let rz_b = OptGate::Rz(0, beta).unitary_2x2().unwrap();
        let ry_g = OptGate::Ry(0, gamma).unitary_2x2().unwrap();
        let rz_d = OptGate::Rz(0, delta).unitary_2x2().unwrap();
        let product = mat2_mul(&rz_b, &mat2_mul(&ry_g, &rz_d));
        // Should match H up to global phase.
        let phase = if product[0].norm() > 1e-12 && h_mat[0].norm() > 1e-12 {
            let p = Complex::new(h_mat[0].re, h_mat[0].im)
                * Complex::new(product[0].re, product[0].im).conj();
            p * Complex::new(1.0 / p.norm(), 0.0)
        } else {
            Complex::ONE
        };
        for i in 0..4 {
            let expected = h_mat[i];
            let got_raw = product[i];
            let got = got_raw * phase;
            let diff = ((expected.re - got.re).powi(2) + (expected.im - got.im).powi(2)).sqrt();
            assert!(
                diff < 1e-6,
                "ZYZ roundtrip mismatch at index {}: expected {:?}, got {:?}",
                i,
                expected,
                got
            );
        }
    }

    // --- Test 46: Empty circuit optimization ---
    #[test]
    fn test_empty_circuit() {
        let c = OptCircuit::new(2);
        let config = OptimizerConfig::level(OptimizationLevel::Level3);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        assert_eq!(report.before.total_gates, 0);
        assert_eq!(report.after.total_gates, 0);
    }

    // --- Test 47: Improvement score weighted correctly ---
    #[test]
    fn test_improvement_score_weighting() {
        let before = CircuitMetrics {
            num_qubits: 2,
            total_gates: 100,
            single_qubit_gates: 80,
            two_qubit_gates: 20,
            three_qubit_gates: 0,
            depth: 50,
            cx_count: 20,
            t_count: 10,
            gate_counts: vec![],
        };
        let after = CircuitMetrics {
            num_qubits: 2,
            total_gates: 50,
            single_qubit_gates: 40,
            two_qubit_gates: 10,
            three_qubit_gates: 0,
            depth: 25,
            cx_count: 10,
            t_count: 5,
            gate_counts: vec![],
        };
        let summary = compute_improvement(&before, &after);
        // 50% reduction across the board.
        assert!((summary.gate_reduction_pct - 50.0).abs() < 0.1);
        assert!((summary.depth_reduction_pct - 50.0).abs() < 0.1);
        assert!((summary.cx_reduction_pct - 50.0).abs() < 0.1);
        assert!((summary.t_reduction_pct - 50.0).abs() < 0.1);
        // Score = 0.4*50 + 0.3*50 + 0.2*50 + 0.1*50 = 50.
        assert!((summary.total_improvement_score - 50.0).abs() < 0.1);
    }

    // --- Test 48: BFS path finding ---
    #[test]
    fn test_bfs_path() {
        let adj = vec![vec![1], vec![0, 2], vec![1]]; // Linear 0-1-2
        let path = bfs_path(&adj, 0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    // --- Test 49: Gates commute when no qubit overlap ---
    #[test]
    fn test_no_overlap_commutes() {
        assert!(gates_commute(&OptGate::H(0), &OptGate::X(1)));
        assert!(gates_commute(&OptGate::CX(0, 1), &OptGate::H(2)));
    }

    // --- Test 50: Mixed optimization preserves correctness ---
    #[test]
    fn test_mixed_circuit_preserves_structure() {
        let mut c = OptCircuit::new(3);
        c.push(OptGate::H(0));
        c.push(OptGate::CX(0, 1));
        c.push(OptGate::T(2));
        c.push(OptGate::CX(1, 2));
        c.push(OptGate::Measure(0));
        let config = OptimizerConfig::level(OptimizationLevel::Level2);
        let opt = CircuitOptimizer::new(config);
        let report = opt.optimize(&c).unwrap();
        // No cancellations possible; circuit should be mostly preserved.
        assert!(report.after.total_gates >= 3);
        // Measure should still be present.
        assert!(report
            .optimized_circuit
            .gates
            .iter()
            .any(|g| matches!(g, OptGate::Measure(0))));
    }
}
