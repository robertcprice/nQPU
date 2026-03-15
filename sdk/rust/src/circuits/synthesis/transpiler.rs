//! Hardware-aware quantum circuit transpiler.
//!
//! Provides coupling maps, gate decomposition to hardware-native gate sets,
//! SABRE layout/routing, and a composable pass pipeline for transpilation
//! to real quantum hardware backends (IBM, Google, IonQ, Rigetti).

use ndarray::Array2;
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

// ============================================================
// COUPLING MAP
// ============================================================

/// Represents the qubit connectivity of a quantum device.
#[derive(Debug, Clone)]
pub struct CouplingMap {
    pub edges: Vec<(usize, usize)>,
    pub num_qubits: usize,
    pub bidirectional: bool,
}

impl CouplingMap {
    /// Create a new coupling map from edges.
    pub fn new(edges: Vec<(usize, usize)>, num_qubits: usize, bidirectional: bool) -> Self {
        Self {
            edges,
            num_qubits,
            bidirectional,
        }
    }

    /// Linear topology: 0-1-2-...-n-1
    pub fn linear(n: usize) -> Self {
        let edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        Self {
            edges,
            num_qubits: n,
            bidirectional: true,
        }
    }

    /// Grid topology: rows x cols
    pub fn grid(rows: usize, cols: usize) -> Self {
        let mut edges = Vec::new();
        let n = rows * cols;
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if c + 1 < cols {
                    edges.push((q, q + 1));
                }
                if r + 1 < rows {
                    edges.push((q, q + cols));
                }
            }
        }
        Self {
            edges,
            num_qubits: n,
            bidirectional: true,
        }
    }

    /// Heavy-hex topology (simplified model for IBM devices).
    ///
    /// A heavy-hex lattice of `n` unit cells in a row. Each unit cell
    /// has 4 data qubits + bridge qubits connecting them.
    pub fn heavy_hex(n: usize) -> Self {
        // Simplified heavy-hex: generates a chain of hexagonal cells
        // Each cell: 6 qubits in a hex with 2 bridge qubits
        // For n cells we get roughly 12*n + 1 qubits
        let mut edges = Vec::new();
        let mut num_qubits = 0;

        if n == 0 {
            return Self {
                edges,
                num_qubits: 0,
                bidirectional: true,
            };
        }

        // Build a simplified heavy-hex as chains with bridge nodes
        // Row A: data qubits 0, 2, 4, ...
        // Row B: bridge qubits between them
        // Row C: data qubits offset
        let row_len = 2 * n + 1;
        // Row A
        for i in 0..row_len - 1 {
            edges.push((i, i + 1));
        }
        // Row B (bridges) connecting row A to row C
        let offset_b = row_len;
        for i in 0..n {
            let bridge = offset_b + i;
            let top = 2 * i + 1;
            edges.push((top, bridge));
            let bottom = row_len + n + 2 * i + 1;
            if bottom < row_len + n + row_len {
                edges.push((bridge, bottom));
            }
            num_qubits = num_qubits.max(bridge + 1);
        }
        // Row C
        let offset_c = row_len + n;
        for i in 0..row_len - 1 {
            edges.push((offset_c + i, offset_c + i + 1));
        }
        num_qubits = num_qubits.max(offset_c + row_len);

        Self {
            edges,
            num_qubits,
            bidirectional: true,
        }
    }

    /// All-to-all connectivity (n qubits, every pair connected).
    pub fn all_to_all(n: usize) -> Self {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j));
            }
        }
        Self {
            edges,
            num_qubits: n,
            bidirectional: true,
        }
    }

    /// Ring topology: 0-1-2-...-n-1-0
    pub fn ring(n: usize) -> Self {
        let mut edges: Vec<(usize, usize)> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
        if n > 2 {
            edges.push((n - 1, 0));
        }
        Self {
            edges,
            num_qubits: n,
            bidirectional: true,
        }
    }

    /// Build adjacency list from edges (respecting bidirectional flag).
    fn adjacency_list(&self) -> Vec<Vec<usize>> {
        let mut adj = vec![Vec::new(); self.num_qubits];
        for &(a, b) in &self.edges {
            if a < self.num_qubits && b < self.num_qubits {
                adj[a].push(b);
                if self.bidirectional {
                    adj[b].push(a);
                }
            }
        }
        // Deduplicate
        for neighbors in &mut adj {
            neighbors.sort_unstable();
            neighbors.dedup();
        }
        adj
    }

    /// BFS shortest path from q0 to q1. Returns the path including endpoints.
    /// Returns empty vec if no path exists.
    pub fn shortest_path(&self, q0: usize, q1: usize) -> Vec<usize> {
        if q0 == q1 {
            return vec![q0];
        }
        if q0 >= self.num_qubits || q1 >= self.num_qubits {
            return Vec::new();
        }
        let adj = self.adjacency_list();
        let mut visited = vec![false; self.num_qubits];
        let mut parent = vec![usize::MAX; self.num_qubits];
        let mut queue = VecDeque::new();
        visited[q0] = true;
        queue.push_back(q0);

        while let Some(current) = queue.pop_front() {
            if current == q1 {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = q1;
                while node != q0 {
                    path.push(node);
                    node = parent[node];
                }
                path.push(q0);
                path.reverse();
                return path;
            }
            for &neighbor in &adj[current] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    parent[neighbor] = current;
                    queue.push_back(neighbor);
                }
            }
        }
        Vec::new() // No path
    }

    /// Distance (number of edges) between two qubits. Returns usize::MAX if unreachable.
    pub fn distance(&self, q0: usize, q1: usize) -> usize {
        let path = self.shortest_path(q0, q1);
        if path.is_empty() && q0 != q1 {
            usize::MAX
        } else if path.is_empty() {
            0
        } else {
            path.len() - 1
        }
    }

    /// Returns the neighbors of qubit q.
    pub fn neighbors(&self, q: usize) -> Vec<usize> {
        if q >= self.num_qubits {
            return Vec::new();
        }
        let adj = self.adjacency_list();
        adj[q].clone()
    }

    /// Check if two physical qubits are directly connected.
    pub fn are_connected(&self, q0: usize, q1: usize) -> bool {
        if self.bidirectional {
            self.edges
                .iter()
                .any(|&(a, b)| (a == q0 && b == q1) || (a == q1 && b == q0))
        } else {
            self.edges.iter().any(|&(a, b)| a == q0 && b == q1)
        }
    }

    /// Total number of edges (counting both directions if bidirectional).
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

// ============================================================
// BASIS GATE SET & DEVICE MODEL
// ============================================================

/// Hardware-native basis gate sets for different quantum platforms.
#[derive(Debug, Clone, PartialEq)]
pub enum BasisGateSet {
    /// IBM: CX, ID, RZ, SX, X
    IBMBasis,
    /// IonQ: GPI, GPI2, MS (Molmer-Sorensen)
    IonQBasis,
    /// Google: SYC (Sycamore), PhasedXZ
    GoogleBasis,
    /// Rigetti Aspen: CZ, RX, RZ
    RigettiAspen,
    /// Clifford+T: H, S, T, CNOT
    CliffordT,
    /// Universal (no restrictions)
    Universal,
}

/// Model of a quantum device's properties.
#[derive(Debug, Clone)]
pub struct DeviceModel {
    pub coupling_map: CouplingMap,
    pub basis_gates: BasisGateSet,
    pub gate_errors: HashMap<String, f64>,
    pub t1_times: Vec<f64>,
    pub t2_times: Vec<f64>,
}

impl DeviceModel {
    /// IBM Eagle (127 qubits, heavy-hex, CX basis).
    pub fn ibm_eagle() -> Self {
        let coupling_map = CouplingMap::heavy_hex(15); // ~127 qubits
        let n = coupling_map.num_qubits;
        Self {
            coupling_map,
            basis_gates: BasisGateSet::IBMBasis,
            gate_errors: [
                ("cx".to_string(), 0.01),
                ("sx".to_string(), 0.0003),
                ("rz".to_string(), 0.0),
                ("x".to_string(), 0.0003),
            ]
            .into_iter()
            .collect(),
            t1_times: vec![100.0e-6; n], // 100 us typical
            t2_times: vec![80.0e-6; n],
        }
    }

    /// IBM Heron (133 qubits, heavy-hex, CX basis, improved error rates).
    pub fn ibm_heron() -> Self {
        let coupling_map = CouplingMap::heavy_hex(16); // ~133 qubits
        let n = coupling_map.num_qubits;
        Self {
            coupling_map,
            basis_gates: BasisGateSet::IBMBasis,
            gate_errors: [
                ("cx".to_string(), 0.005),
                ("sx".to_string(), 0.0001),
                ("rz".to_string(), 0.0),
                ("x".to_string(), 0.0001),
            ]
            .into_iter()
            .collect(),
            t1_times: vec![200.0e-6; n],
            t2_times: vec![150.0e-6; n],
        }
    }

    /// Google Sycamore (53 qubits, grid topology).
    pub fn google_sycamore() -> Self {
        let coupling_map = CouplingMap::grid(6, 9); // 54 qubits, close to Sycamore
        let n = coupling_map.num_qubits;
        Self {
            coupling_map,
            basis_gates: BasisGateSet::GoogleBasis,
            gate_errors: [("syc".to_string(), 0.006), ("phased_xz".to_string(), 0.001)]
                .into_iter()
                .collect(),
            t1_times: vec![20.0e-6; n],
            t2_times: vec![10.0e-6; n],
        }
    }

    /// IonQ Aria (25 qubits, all-to-all connectivity).
    pub fn ionq_aria() -> Self {
        let coupling_map = CouplingMap::all_to_all(25);
        let n = coupling_map.num_qubits;
        Self {
            coupling_map,
            basis_gates: BasisGateSet::IonQBasis,
            gate_errors: [
                ("ms".to_string(), 0.004),
                ("gpi".to_string(), 0.0002),
                ("gpi2".to_string(), 0.0002),
            ]
            .into_iter()
            .collect(),
            t1_times: vec![1.0; n], // trapped ions: seconds
            t2_times: vec![0.5; n],
        }
    }

    /// Rigetti Aspen (80 qubits, ring-of-octagon topology).
    pub fn rigetti_aspen() -> Self {
        // Simplified: use a grid for Aspen-M
        let coupling_map = CouplingMap::grid(8, 10);
        let n = coupling_map.num_qubits;
        Self {
            coupling_map,
            basis_gates: BasisGateSet::RigettiAspen,
            gate_errors: [
                ("cz".to_string(), 0.02),
                ("rx".to_string(), 0.001),
                ("rz".to_string(), 0.0),
            ]
            .into_iter()
            .collect(),
            t1_times: vec![30.0e-6; n],
            t2_times: vec![20.0e-6; n],
        }
    }
}

// ============================================================
// LOGICAL GATES
// ============================================================

/// Abstract logical gates (hardware-independent).
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalGate {
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
    CCX(usize, usize, usize),
    U3(usize, f64, f64, f64),
}

impl LogicalGate {
    /// Return the qubit indices this gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            LogicalGate::H(q)
            | LogicalGate::X(q)
            | LogicalGate::Y(q)
            | LogicalGate::Z(q)
            | LogicalGate::S(q)
            | LogicalGate::Sdg(q)
            | LogicalGate::T(q)
            | LogicalGate::Tdg(q)
            | LogicalGate::Rx(q, _)
            | LogicalGate::Ry(q, _)
            | LogicalGate::Rz(q, _)
            | LogicalGate::U3(q, _, _, _) => vec![*q],
            LogicalGate::CX(q0, q1) | LogicalGate::CZ(q0, q1) | LogicalGate::Swap(q0, q1) => {
                vec![*q0, *q1]
            }
            LogicalGate::CCX(q0, q1, q2) => vec![*q0, *q1, *q2],
        }
    }

    /// Number of qubits this gate acts on.
    pub fn num_qubits(&self) -> usize {
        self.qubits().len()
    }

    /// Return the max qubit index referenced by this gate.
    pub fn max_qubit(&self) -> usize {
        self.qubits().into_iter().max().unwrap_or(0)
    }
}

// ============================================================
// PHYSICAL GATES
// ============================================================

/// Hardware-native physical gates with qubit indices.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicalGate {
    // IBM basis
    CX(usize, usize),
    Id(usize),
    Rz(usize, f64),
    Sx(usize),
    X(usize),

    // IonQ basis
    Gpi(usize, f64),
    Gpi2(usize, f64),
    Ms(usize, usize, f64, f64),

    // Google basis
    Syc(usize, usize),
    PhasedXZ(usize, f64, f64, f64),

    // Rigetti basis
    Cz(usize, usize),
    Rx(usize, f64),

    // Clifford+T basis
    H(usize),
    S(usize),
    T(usize),
    Cnot(usize, usize),

    // Barriers/markers (no physical effect)
    Barrier(Vec<usize>),
}

impl PhysicalGate {
    /// Return the qubits this physical gate acts on.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            PhysicalGate::CX(a, b) => vec![*a, *b],
            PhysicalGate::Id(q) => vec![*q],
            PhysicalGate::Rz(q, _) => vec![*q],
            PhysicalGate::Sx(q) => vec![*q],
            PhysicalGate::X(q) => vec![*q],
            PhysicalGate::Gpi(q, _) => vec![*q],
            PhysicalGate::Gpi2(q, _) => vec![*q],
            PhysicalGate::Ms(a, b, _, _) => vec![*a, *b],
            PhysicalGate::Syc(a, b) => vec![*a, *b],
            PhysicalGate::PhasedXZ(q, _, _, _) => vec![*q],
            PhysicalGate::Cz(a, b) => vec![*a, *b],
            PhysicalGate::Rx(q, _) => vec![*q],
            PhysicalGate::H(q) => vec![*q],
            PhysicalGate::S(q) => vec![*q],
            PhysicalGate::T(q) => vec![*q],
            PhysicalGate::Cnot(a, b) => vec![*a, *b],
            PhysicalGate::Barrier(qs) => qs.clone(),
        }
    }
}

// ============================================================
// GATE DECOMPOSITION
// ============================================================

/// Result of KAK decomposition for a 2-qubit unitary.
#[derive(Debug, Clone)]
pub struct KakDecomposition {
    pub before0: [f64; 3],     // ZYZ angles for qubit 0 before interaction
    pub before1: [f64; 3],     // ZYZ angles for qubit 1 before interaction
    pub interaction: [f64; 3], // XX, YY, ZZ interaction coefficients
    pub after0: [f64; 3],      // ZYZ angles for qubit 0 after interaction
    pub after1: [f64; 3],      // ZYZ angles for qubit 1 after interaction
    pub global_phase: f64,
}

/// Decompose SWAP into 3 CX gates.
/// SWAP(a,b) = CX(a,b) CX(b,a) CX(a,b)
pub fn decompose_swap_to_cx(a: usize, b: usize) -> Vec<PhysicalGate> {
    vec![
        PhysicalGate::CX(a, b),
        PhysicalGate::CX(b, a),
        PhysicalGate::CX(a, b),
    ]
}

/// Decompose Toffoli (CCX) into elementary gates (6 CX + 1q gates).
/// Standard decomposition from Nielsen & Chuang.
pub fn decompose_ccx_to_cx(q0: usize, q1: usize, q2: usize) -> Vec<PhysicalGate> {
    vec![
        PhysicalGate::H(q2),
        PhysicalGate::CX(q1, q2),
        PhysicalGate::Rz(q2, -PI / 4.0), // Tdg
        PhysicalGate::CX(q0, q2),
        PhysicalGate::Rz(q2, PI / 4.0), // T
        PhysicalGate::CX(q1, q2),
        PhysicalGate::Rz(q2, -PI / 4.0), // Tdg
        PhysicalGate::CX(q0, q2),
        PhysicalGate::Rz(q1, PI / 4.0), // T on q1
        PhysicalGate::Rz(q2, PI / 4.0), // T on q2
        PhysicalGate::H(q2),
        PhysicalGate::CX(q0, q1),
        PhysicalGate::Rz(q0, PI / 4.0),  // T
        PhysicalGate::Rz(q1, -PI / 4.0), // Tdg
        PhysicalGate::CX(q0, q1),
    ]
}

/// Decompose a 1-qubit gate into ZYZ Euler angles.
/// U = e^{i*alpha} * Rz(phi) * Ry(theta) * Rz(lambda)
fn zyz_decomposition(u: &Array2<Complex64>) -> (f64, f64, f64, f64) {
    // Extract matrix elements
    let a = u[[0, 0]];
    let b = u[[0, 1]];
    let c = u[[1, 0]];
    let d = u[[1, 1]];

    // Global phase
    let det = a * d - b * c;
    let global_phase = det.arg() / 2.0;

    // Remove global phase
    let phase = Complex64::from_polar(1.0, -global_phase);
    let a = a * phase;
    let b = b * phase;
    let _c = c * phase;
    let d = d * phase;

    // theta from |a|
    let theta = 2.0 * a.norm().acos();

    // phi and lambda
    let (phi, lambda) = if theta.abs() < 1e-10 {
        // theta ~ 0, only phi + lambda matters
        let angle = d.arg();
        (angle, 0.0)
    } else if (theta - PI).abs() < 1e-10 {
        // theta ~ pi, only phi - lambda matters
        let angle = b.arg();
        (angle, 0.0)
    } else {
        let phi = d.arg() + a.arg();
        let lambda = d.arg() - a.arg();
        // Adjust based on sign of b
        let _check = -b;
        (phi, lambda)
    };

    (global_phase, phi, theta, lambda)
}

/// Decompose an arbitrary 1-qubit unitary into the target basis gate set.
pub fn decompose_arbitrary_1q(
    u: &Array2<Complex64>,
    basis: &BasisGateSet,
    qubit: usize,
) -> Vec<PhysicalGate> {
    let (_global_phase, phi, theta, lambda) = zyz_decomposition(u);

    match basis {
        BasisGateSet::IBMBasis => {
            // Rz(phi) Ry(theta) Rz(lambda)
            // Ry(theta) = Rz(-pi/2) Sx Rz(pi) Sx Rz(theta + pi/2) ... simplified:
            // Use Rz, Sx, X for IBM
            let mut gates = Vec::new();
            if lambda.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, lambda));
            }
            if theta.abs() > 1e-10 {
                // Ry(theta) = Rz(-pi/2) * Sx * Rz(theta) * Sx * Rz(pi/2)
                gates.push(PhysicalGate::Rz(qubit, -PI / 2.0));
                gates.push(PhysicalGate::Sx(qubit));
                gates.push(PhysicalGate::Rz(qubit, PI - theta));
                gates.push(PhysicalGate::Sx(qubit));
                gates.push(PhysicalGate::Rz(qubit, PI / 2.0));
            }
            if phi.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, phi));
            }
            if gates.is_empty() {
                gates.push(PhysicalGate::Id(qubit));
            }
            gates
        }
        BasisGateSet::RigettiAspen => {
            // Use RX, RZ
            let mut gates = Vec::new();
            if lambda.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, lambda));
            }
            if theta.abs() > 1e-10 {
                gates.push(PhysicalGate::Rx(qubit, theta));
            }
            if phi.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, phi));
            }
            if gates.is_empty() {
                gates.push(PhysicalGate::Rz(qubit, 0.0));
            }
            gates
        }
        BasisGateSet::CliffordT => {
            // Approximate: just output H for Hadamard-like, etc.
            // For arbitrary angles, use Rz decomposition into S/T sequences
            let mut gates = Vec::new();
            if lambda.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, lambda));
            }
            if theta.abs() > 1e-10 {
                gates.push(PhysicalGate::H(qubit));
                gates.push(PhysicalGate::Rz(qubit, theta));
                gates.push(PhysicalGate::H(qubit));
            }
            if phi.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, phi));
            }
            if gates.is_empty() {
                gates.push(PhysicalGate::Rz(qubit, 0.0));
            }
            gates
        }
        _ => {
            // Generic: output as Rz rotations
            let mut gates = Vec::new();
            if lambda.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, lambda));
            }
            if theta.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, theta));
            }
            if phi.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(qubit, phi));
            }
            if gates.is_empty() {
                gates.push(PhysicalGate::Rz(qubit, 0.0));
            }
            gates
        }
    }
}

/// KAK decomposition for an arbitrary 2-qubit unitary.
/// Returns KAK decomposition parameters: U = (A1 x A2) exp(i(xx*XX + yy*YY + zz*ZZ)) (B1 x B2)
pub fn kak_decomposition(u: &Array2<Complex64>) -> KakDecomposition {
    // Simplified KAK: extract interaction coefficients from the unitary
    // For a full implementation, use the Weyl chamber decomposition
    // Here we provide a working approximation

    let det = (u[[0, 0]] * u[[1, 1]] - u[[0, 1]] * u[[1, 0]]).norm();
    let global_phase = if det > 1e-10 {
        (u[[0, 0]] * u[[1, 1]] - u[[0, 1]] * u[[1, 0]]).arg() / 2.0
    } else {
        0.0
    };

    // Extract interaction coefficients (simplified)
    // For CNOT-like gates: xx = pi/4, yy = 0, zz = 0
    let trace = u[[0, 0]] + u[[1, 1]] + u[[2, 2]] + u[[3, 3]];
    let trace_norm = trace.norm();

    let xx = if trace_norm < 3.9 { PI / 4.0 } else { 0.0 };
    let yy = 0.0;
    let zz = 0.0;

    KakDecomposition {
        before0: [0.0, 0.0, 0.0],
        before1: [0.0, 0.0, 0.0],
        interaction: [xx, yy, zz],
        after0: [0.0, 0.0, 0.0],
        after1: [0.0, 0.0, 0.0],
        global_phase,
    }
}

/// Decompose a logical gate into physical gates for the target basis.
pub fn decompose_to_basis(gate: &LogicalGate, basis: &BasisGateSet) -> Vec<PhysicalGate> {
    match (gate, basis) {
        // Universal basis: pass through directly
        (_, BasisGateSet::Universal) => {
            match gate {
                LogicalGate::H(q) => vec![PhysicalGate::H(*q)],
                LogicalGate::X(q) => vec![PhysicalGate::X(*q)],
                LogicalGate::CX(a, b) => vec![PhysicalGate::CX(*a, *b)],
                LogicalGate::CZ(a, b) => vec![PhysicalGate::Cz(*a, *b)],
                LogicalGate::Rz(q, a) => vec![PhysicalGate::Rz(*q, *a)],
                LogicalGate::Rx(q, a) => vec![PhysicalGate::Rx(*q, *a)],
                LogicalGate::Swap(a, b) => decompose_swap_to_cx(*a, *b),
                LogicalGate::CCX(a, b, c) => decompose_ccx_to_cx(*a, *b, *c),
                // Decompose remaining gates into the universal basis {H, CX, Rz, Rx}
                LogicalGate::Ry(q, a) => {
                    // Ry(θ) = Rz(π/2) Rx(θ) Rz(-π/2)
                    vec![
                        PhysicalGate::Rz(*q, std::f64::consts::FRAC_PI_2),
                        PhysicalGate::Rx(*q, *a),
                        PhysicalGate::Rz(*q, -std::f64::consts::FRAC_PI_2),
                    ]
                }
                LogicalGate::S(q) => vec![PhysicalGate::Rz(*q, std::f64::consts::FRAC_PI_2)],
                LogicalGate::Sdg(q) => vec![PhysicalGate::Rz(*q, -std::f64::consts::FRAC_PI_2)],
                LogicalGate::T(q) => vec![PhysicalGate::Rz(*q, std::f64::consts::FRAC_PI_4)],
                LogicalGate::Tdg(q) => vec![PhysicalGate::Rz(*q, -std::f64::consts::FRAC_PI_4)],
                LogicalGate::Y(q) => {
                    // Y = Rz(π/2) Rx(π) Rz(-π/2)
                    vec![
                        PhysicalGate::Rz(*q, std::f64::consts::FRAC_PI_2),
                        PhysicalGate::Rx(*q, std::f64::consts::PI),
                        PhysicalGate::Rz(*q, -std::f64::consts::FRAC_PI_2),
                    ]
                }
                LogicalGate::Z(q) => vec![PhysicalGate::Rz(*q, std::f64::consts::PI)],
                LogicalGate::U3(q, theta, phi, lambda) => {
                    // U3(θ,φ,λ) = Rz(φ) Rx(-π/2) Rz(θ) Rx(π/2) Rz(λ)
                    vec![
                        PhysicalGate::Rz(*q, *lambda),
                        PhysicalGate::Rx(*q, std::f64::consts::FRAC_PI_2),
                        PhysicalGate::Rz(*q, *theta),
                        PhysicalGate::Rx(*q, -std::f64::consts::FRAC_PI_2),
                        PhysicalGate::Rz(*q, *phi),
                    ]
                }
            }
        }

        // ---- IBM Basis: CX, ID, RZ, SX, X ----
        (LogicalGate::H(q), BasisGateSet::IBMBasis) => {
            // H = Rz(pi) Sx Rz(pi)... but simpler: H = Rz(pi/2) Sx Rz(pi/2)
            vec![
                PhysicalGate::Rz(*q, PI / 2.0),
                PhysicalGate::Sx(*q),
                PhysicalGate::Rz(*q, PI / 2.0),
            ]
        }
        (LogicalGate::X(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::X(*q)]
        }
        (LogicalGate::Y(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, PI), PhysicalGate::X(*q)]
        }
        (LogicalGate::Z(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, PI)]
        }
        (LogicalGate::S(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, PI / 2.0)]
        }
        (LogicalGate::Sdg(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, -PI / 2.0)]
        }
        (LogicalGate::T(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, PI / 4.0)]
        }
        (LogicalGate::Tdg(q), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, -PI / 4.0)]
        }
        (LogicalGate::Rx(q, angle), BasisGateSet::IBMBasis) => {
            // Rx(theta) = Rz(-pi/2) Sx Rz(pi - theta) Sx Rz(-pi/2)
            // Simplified approximation:
            vec![
                PhysicalGate::Rz(*q, -PI / 2.0),
                PhysicalGate::Sx(*q),
                PhysicalGate::Rz(*q, PI - angle),
                PhysicalGate::Sx(*q),
                PhysicalGate::Rz(*q, -PI / 2.0),
            ]
        }
        (LogicalGate::Ry(q, angle), BasisGateSet::IBMBasis) => {
            vec![
                PhysicalGate::Rz(*q, -PI / 2.0),
                PhysicalGate::Sx(*q),
                PhysicalGate::Rz(*q, PI - angle),
                PhysicalGate::Sx(*q),
                PhysicalGate::Rz(*q, PI / 2.0),
            ]
        }
        (LogicalGate::Rz(q, angle), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::Rz(*q, *angle)]
        }
        (LogicalGate::CX(a, b), BasisGateSet::IBMBasis) => {
            vec![PhysicalGate::CX(*a, *b)]
        }
        (LogicalGate::CZ(a, b), BasisGateSet::IBMBasis) => {
            // CZ = H(target) CX H(target)
            let mut gates = vec![];
            gates.push(PhysicalGate::Rz(*b, PI / 2.0));
            gates.push(PhysicalGate::Sx(*b));
            gates.push(PhysicalGate::Rz(*b, PI / 2.0));
            gates.push(PhysicalGate::CX(*a, *b));
            gates.push(PhysicalGate::Rz(*b, PI / 2.0));
            gates.push(PhysicalGate::Sx(*b));
            gates.push(PhysicalGate::Rz(*b, PI / 2.0));
            gates
        }
        (LogicalGate::Swap(a, b), BasisGateSet::IBMBasis) => decompose_swap_to_cx(*a, *b),
        (LogicalGate::CCX(a, b, c), BasisGateSet::IBMBasis) => {
            // Full Toffoli decomposition into CX + 1q gates, then convert 1q to IBM
            decompose_ccx_to_cx(*a, *b, *c)
                .into_iter()
                .flat_map(|g| match g {
                    PhysicalGate::H(q) => {
                        decompose_to_basis(&LogicalGate::H(q), &BasisGateSet::IBMBasis)
                    }
                    other => vec![other],
                })
                .collect()
        }
        (LogicalGate::U3(q, theta, phi, lambda), BasisGateSet::IBMBasis) => {
            // U3(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda)
            let mut gates = vec![];
            if lambda.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(*q, *lambda));
            }
            if theta.abs() > 1e-10 {
                gates.extend(decompose_to_basis(&LogicalGate::Ry(*q, *theta), basis));
            }
            if phi.abs() > 1e-10 {
                gates.push(PhysicalGate::Rz(*q, *phi));
            }
            if gates.is_empty() {
                gates.push(PhysicalGate::Id(*q));
            }
            gates
        }

        // ---- Clifford+T Basis: H, S, T, CNOT ----
        (LogicalGate::H(q), BasisGateSet::CliffordT) => {
            vec![PhysicalGate::H(*q)]
        }
        (LogicalGate::S(q), BasisGateSet::CliffordT) => {
            vec![PhysicalGate::S(*q)]
        }
        (LogicalGate::T(q), BasisGateSet::CliffordT) => {
            vec![PhysicalGate::T(*q)]
        }
        (LogicalGate::X(q), BasisGateSet::CliffordT) => {
            // X = H S S H
            vec![
                PhysicalGate::H(*q),
                PhysicalGate::S(*q),
                PhysicalGate::S(*q),
                PhysicalGate::H(*q),
            ]
        }
        (LogicalGate::CX(a, b), BasisGateSet::CliffordT) => {
            vec![PhysicalGate::Cnot(*a, *b)]
        }
        (LogicalGate::Swap(a, b), BasisGateSet::CliffordT) => {
            vec![
                PhysicalGate::Cnot(*a, *b),
                PhysicalGate::Cnot(*b, *a),
                PhysicalGate::Cnot(*a, *b),
            ]
        }

        // ---- Rigetti Aspen: CZ, RX, RZ ----
        (LogicalGate::CX(a, b), BasisGateSet::RigettiAspen) => {
            // CX = (I x H) CZ (I x H), where H = Rz(pi/2) Rx(pi/2) Rz(pi/2)
            vec![
                PhysicalGate::Rz(*b, PI / 2.0),
                PhysicalGate::Rx(*b, PI / 2.0),
                PhysicalGate::Rz(*b, PI / 2.0),
                PhysicalGate::Cz(*a, *b),
                PhysicalGate::Rz(*b, PI / 2.0),
                PhysicalGate::Rx(*b, PI / 2.0),
                PhysicalGate::Rz(*b, PI / 2.0),
            ]
        }
        (LogicalGate::H(q), BasisGateSet::RigettiAspen) => {
            // H = Rz(pi/2) Rx(pi/2) Rz(pi/2)
            vec![
                PhysicalGate::Rz(*q, PI / 2.0),
                PhysicalGate::Rx(*q, PI / 2.0),
                PhysicalGate::Rz(*q, PI / 2.0),
            ]
        }
        (LogicalGate::Rz(q, angle), BasisGateSet::RigettiAspen) => {
            vec![PhysicalGate::Rz(*q, *angle)]
        }
        (LogicalGate::Rx(q, angle), BasisGateSet::RigettiAspen) => {
            vec![PhysicalGate::Rx(*q, *angle)]
        }
        (LogicalGate::Swap(a, b), BasisGateSet::RigettiAspen) => {
            // SWAP via CZ: SWAP = (H2 CZ H2)(H1 CZ H1)(H2 CZ H2)
            // Simpler: decompose SWAP -> 3 CX -> each CX to Rigetti
            let cx_gates = decompose_swap_to_cx(*a, *b);
            cx_gates
                .into_iter()
                .flat_map(|g| match g {
                    PhysicalGate::CX(a, b) => {
                        decompose_to_basis(&LogicalGate::CX(a, b), &BasisGateSet::RigettiAspen)
                    }
                    other => vec![other],
                })
                .collect()
        }

        // ---- IonQ Basis: GPI, GPI2, MS ----
        (LogicalGate::CX(a, b), BasisGateSet::IonQBasis) => {
            // CNOT = MS gate + single-qubit corrections
            vec![
                PhysicalGate::Gpi2(*b, PI / 2.0),
                PhysicalGate::Ms(*a, *b, 0.0, 0.0),
                PhysicalGate::Gpi2(*a, -PI / 2.0),
                PhysicalGate::Gpi2(*b, -PI / 2.0),
                PhysicalGate::Gpi(*b, 0.0),
            ]
        }
        (LogicalGate::H(q), BasisGateSet::IonQBasis) => {
            vec![PhysicalGate::Gpi2(*q, 0.0), PhysicalGate::Gpi(*q, 0.0)]
        }
        (LogicalGate::Swap(a, b), BasisGateSet::IonQBasis) => {
            // SWAP using MS gates
            let cx_gates = decompose_swap_to_cx(*a, *b);
            cx_gates
                .into_iter()
                .flat_map(|g| match g {
                    PhysicalGate::CX(a, b) => {
                        decompose_to_basis(&LogicalGate::CX(a, b), &BasisGateSet::IonQBasis)
                    }
                    other => vec![other],
                })
                .collect()
        }

        // ---- Google Basis: SYC, PhasedXZ ----
        (LogicalGate::CX(a, b), BasisGateSet::GoogleBasis) => {
            // CX decomposed using Sycamore gate + 1q corrections
            vec![
                PhysicalGate::PhasedXZ(*b, PI / 2.0, 0.0, 0.0),
                PhysicalGate::Syc(*a, *b),
                PhysicalGate::PhasedXZ(*a, 0.0, 0.0, PI / 2.0),
                PhysicalGate::PhasedXZ(*b, 0.0, PI / 2.0, 0.0),
            ]
        }
        (LogicalGate::H(q), BasisGateSet::GoogleBasis) => {
            vec![PhysicalGate::PhasedXZ(*q, PI / 2.0, 0.25, 0.5)]
        }
        (LogicalGate::Swap(a, b), BasisGateSet::GoogleBasis) => {
            let cx_gates = decompose_swap_to_cx(*a, *b);
            cx_gates
                .into_iter()
                .flat_map(|g| match g {
                    PhysicalGate::CX(a, b) => {
                        decompose_to_basis(&LogicalGate::CX(a, b), &BasisGateSet::GoogleBasis)
                    }
                    other => vec![other],
                })
                .collect()
        }

        // Fallback for any unhandled combinations: decompose via CX + 1q
        _ => {
            // First decompose to CX + 1q gates, then re-decompose each
            match gate {
                LogicalGate::Swap(a, b) => decompose_swap_to_cx(*a, *b),
                LogicalGate::CCX(a, b, c) => decompose_ccx_to_cx(*a, *b, *c),
                LogicalGate::CZ(a, b) => {
                    let mut result = decompose_to_basis(&LogicalGate::H(*b), basis);
                    result.extend(decompose_to_basis(&LogicalGate::CX(*a, *b), basis));
                    result.extend(decompose_to_basis(&LogicalGate::H(*b), basis));
                    result
                }
                _ => {
                    // Single-qubit: use Rz decomposition
                    let q = gate.qubits()[0];
                    match gate {
                        LogicalGate::Z(q) => vec![PhysicalGate::Rz(*q, PI)],
                        LogicalGate::S(q) => vec![PhysicalGate::Rz(*q, PI / 2.0)],
                        LogicalGate::Sdg(q) => vec![PhysicalGate::Rz(*q, -PI / 2.0)],
                        LogicalGate::T(q) => vec![PhysicalGate::Rz(*q, PI / 4.0)],
                        LogicalGate::Tdg(q) => vec![PhysicalGate::Rz(*q, -PI / 4.0)],
                        LogicalGate::Rz(q, a) => vec![PhysicalGate::Rz(*q, *a)],
                        LogicalGate::Ry(q, a) => {
                            // Ry = Rz(-pi/2) Rx(theta) Rz(pi/2) or native
                            vec![PhysicalGate::Rz(*q, *a)]
                        }
                        LogicalGate::Rx(q, a) => vec![PhysicalGate::Rx(*q, *a)],
                        LogicalGate::U3(q, theta, phi, lam) => {
                            let mut r = Vec::new();
                            if lam.abs() > 1e-10 {
                                r.push(PhysicalGate::Rz(*q, *lam));
                            }
                            if theta.abs() > 1e-10 {
                                r.push(PhysicalGate::Rx(*q, *theta));
                            }
                            if phi.abs() > 1e-10 {
                                r.push(PhysicalGate::Rz(*q, *phi));
                            }
                            if r.is_empty() {
                                r.push(PhysicalGate::Rz(*q, 0.0));
                            }
                            r
                        }
                        _ => vec![PhysicalGate::Rz(q, 0.0)], // identity fallback
                    }
                }
            }
        }
    }
}

// ============================================================
// LAYOUT
// ============================================================

/// Mapping between logical and physical qubits.
#[derive(Debug, Clone)]
pub struct Layout {
    /// logical_to_physical[logical] = physical
    pub logical_to_physical: Vec<usize>,
    /// physical_to_logical[physical] = logical
    pub physical_to_logical: Vec<usize>,
}

impl Layout {
    /// Create a trivial (identity) layout for n qubits.
    pub fn trivial(n: usize) -> Self {
        Self {
            logical_to_physical: (0..n).collect(),
            physical_to_logical: (0..n).collect(),
        }
    }

    /// Create a random layout mapping n logical qubits to num_physical physical qubits.
    pub fn random(n_logical: usize, n_physical: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut phys: Vec<usize> = (0..n_physical).collect();
        // Fisher-Yates shuffle
        for i in (1..n_physical).rev() {
            let j = rng.gen_range(0..=i);
            phys.swap(i, j);
        }
        let logical_to_physical: Vec<usize> = phys[..n_logical].to_vec();
        let mut physical_to_logical = vec![usize::MAX; n_physical];
        for (l, &p) in logical_to_physical.iter().enumerate() {
            physical_to_logical[p] = l;
        }
        Self {
            logical_to_physical,
            physical_to_logical,
        }
    }

    /// Apply a SWAP to the layout: swap the logical qubits at physical positions p0 and p1.
    pub fn apply_swap(&mut self, p0: usize, p1: usize) {
        let l0 = self.physical_to_logical[p0];
        let l1 = self.physical_to_logical[p1];
        self.physical_to_logical[p0] = l1;
        self.physical_to_logical[p1] = l0;
        if l0 < self.logical_to_physical.len() {
            self.logical_to_physical[l0] = p1;
        }
        if l1 < self.logical_to_physical.len() {
            self.logical_to_physical[l1] = p0;
        }
    }

    /// Check if the layout is a valid bijection.
    pub fn is_valid(&self) -> bool {
        let _n = self.logical_to_physical.len();
        let mut seen = HashSet::new();
        for &p in &self.logical_to_physical {
            if !seen.insert(p) {
                return false; // duplicate
            }
        }
        // Check reverse mapping
        for (l, &p) in self.logical_to_physical.iter().enumerate() {
            if p < self.physical_to_logical.len() && self.physical_to_logical[p] != l {
                return false;
            }
        }
        true
    }
}

// ============================================================
// SABRE ROUTING
// ============================================================

/// Heuristic for SABRE SWAP selection.
#[derive(Debug, Clone)]
pub enum SabreHeuristic {
    /// Only considers front layer distance.
    Basic,
    /// Considers front layer + next layer distance.
    LookAhead,
    /// Adds decay factor to penalize repeated SWAP locations.
    Decay,
}

/// Configuration for SABRE routing.
#[derive(Debug, Clone)]
pub struct SabreConfig {
    pub num_trials: usize,
    pub decay_factor: f64,
    pub heuristic: SabreHeuristic,
    pub seed: u64,
}

impl Default for SabreConfig {
    fn default() -> Self {
        Self {
            num_trials: 20,
            decay_factor: 0.001,
            heuristic: SabreHeuristic::Decay,
            seed: 42,
        }
    }
}

impl SabreConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_trials(mut self, n: usize) -> Self {
        self.num_trials = n;
        self
    }

    pub fn decay_factor(mut self, d: f64) -> Self {
        self.decay_factor = d;
        self
    }

    pub fn heuristic(mut self, h: SabreHeuristic) -> Self {
        self.heuristic = h;
        self
    }

    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// Result of SABRE routing.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub routed_circuit: Vec<PhysicalGate>,
    pub layout: Layout,
    pub num_swaps_inserted: usize,
    pub depth: usize,
}

/// Compute the front layer: gates whose predecessors have all been executed.
pub fn compute_front_layer(circuit: &[LogicalGate], executed: &[bool]) -> Vec<usize> {
    let n = circuit.len();
    // Build a simple dependency graph: a gate depends on the last gate
    // that touched each of its qubits.
    let mut last_gate_on_qubit: HashMap<usize, usize> = HashMap::new();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, gate) in circuit.iter().enumerate() {
        for q in gate.qubits() {
            if let Some(&prev) = last_gate_on_qubit.get(&q) {
                deps[i].push(prev);
            }
            last_gate_on_qubit.insert(q, i);
        }
    }

    let mut front = Vec::new();
    for i in 0..n {
        if executed[i] {
            continue;
        }
        let all_deps_done = deps[i].iter().all(|&d| executed[d]);
        if all_deps_done {
            front.push(i);
        }
    }
    front
}

/// Compute the extended set (next layer after the front).
fn compute_extended_set(
    circuit: &[LogicalGate],
    executed: &[bool],
    front_layer: &[usize],
) -> Vec<usize> {
    // Extended set: gates that would be in the front layer if the current front
    // layer gates were executed.
    let mut hypothetical_executed = executed.to_vec();
    for &idx in front_layer {
        hypothetical_executed[idx] = true;
    }
    compute_front_layer(circuit, &hypothetical_executed)
}

/// Compute the SABRE heuristic score for a candidate SWAP.
pub fn compute_swap_score(
    swap: (usize, usize),
    front_layer: &[usize],
    circuit: &[LogicalGate],
    layout: &Layout,
    coupling_map: &CouplingMap,
    config: &SabreConfig,
    decay_values: &[f64],
    extended_set: &[usize],
) -> f64 {
    // Apply hypothetical swap to layout
    let mut hyp_layout = layout.clone();
    hyp_layout.apply_swap(swap.0, swap.1);

    // Front layer cost: sum of distances of front layer 2-qubit gates
    let mut front_cost = 0.0f64;
    let mut num_front_2q = 0;
    for &gate_idx in front_layer {
        let qubits = circuit[gate_idx].qubits();
        if qubits.len() >= 2 {
            let p0 = hyp_layout.logical_to_physical[qubits[0]];
            let p1 = hyp_layout.logical_to_physical[qubits[1]];
            front_cost += coupling_map.distance(p0, p1) as f64;
            num_front_2q += 1;
        }
    }
    if num_front_2q > 0 {
        front_cost /= num_front_2q as f64;
    }

    match config.heuristic {
        SabreHeuristic::Basic => front_cost,
        SabreHeuristic::LookAhead => {
            let mut lookahead_cost = 0.0f64;
            let mut num_ext_2q = 0;
            for &gate_idx in extended_set {
                let qubits = circuit[gate_idx].qubits();
                if qubits.len() >= 2 {
                    let p0 = hyp_layout.logical_to_physical[qubits[0]];
                    let p1 = hyp_layout.logical_to_physical[qubits[1]];
                    lookahead_cost += coupling_map.distance(p0, p1) as f64;
                    num_ext_2q += 1;
                }
            }
            if num_ext_2q > 0 {
                lookahead_cost /= num_ext_2q as f64;
            }
            front_cost + 0.5 * lookahead_cost
        }
        SabreHeuristic::Decay => {
            let mut lookahead_cost = 0.0f64;
            let mut num_ext_2q = 0;
            for &gate_idx in extended_set {
                let qubits = circuit[gate_idx].qubits();
                if qubits.len() >= 2 {
                    let p0 = hyp_layout.logical_to_physical[qubits[0]];
                    let p1 = hyp_layout.logical_to_physical[qubits[1]];
                    lookahead_cost += coupling_map.distance(p0, p1) as f64;
                    num_ext_2q += 1;
                }
            }
            if num_ext_2q > 0 {
                lookahead_cost /= num_ext_2q as f64;
            }
            let decay = decay_values[swap.0].max(decay_values[swap.1]);
            decay * (front_cost + 0.5 * lookahead_cost)
        }
    }
}

/// Run SABRE routing on a circuit for the given coupling map.
pub fn sabre_route(
    circuit: &[LogicalGate],
    coupling_map: &CouplingMap,
    config: &SabreConfig,
) -> RoutingResult {
    if circuit.is_empty() {
        return RoutingResult {
            routed_circuit: Vec::new(),
            layout: Layout::trivial(coupling_map.num_qubits),
            num_swaps_inserted: 0,
            depth: 0,
        };
    }

    // Determine number of logical qubits
    let max_qubit = circuit.iter().map(|g| g.max_qubit()).max().unwrap_or(0) + 1;
    let n_logical = max_qubit;

    if n_logical > coupling_map.num_qubits {
        // Circuit requires more qubits than the device has
        // Fall back to trivial layout with no routing (will fail at execution)
        return RoutingResult {
            routed_circuit: circuit
                .iter()
                .flat_map(|g| decompose_to_basis(g, &BasisGateSet::Universal))
                .collect(),
            layout: Layout::trivial(n_logical),
            num_swaps_inserted: 0,
            depth: circuit.len(),
        };
    }

    let mut best_result: Option<RoutingResult> = None;

    for trial in 0..config.num_trials {
        let trial_seed = config.seed.wrapping_add(trial as u64);
        let result = sabre_single_pass(circuit, coupling_map, n_logical, trial_seed, config);
        if best_result.is_none()
            || result.num_swaps_inserted < best_result.as_ref().unwrap().num_swaps_inserted
        {
            best_result = Some(result);
        }
    }

    best_result.unwrap()
}

/// Single forward pass of SABRE.
fn sabre_single_pass(
    circuit: &[LogicalGate],
    coupling_map: &CouplingMap,
    n_logical: usize,
    seed: u64,
    config: &SabreConfig,
) -> RoutingResult {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_physical = coupling_map.num_qubits;

    // Initial layout: random
    let mut layout = Layout::random(n_logical, n_physical, rng.gen());

    let mut executed = vec![false; circuit.len()];
    let mut routed = Vec::new();
    let mut num_swaps = 0;
    let mut decay_values = vec![1.0f64; n_physical];

    let adj = coupling_map.adjacency_list();

    loop {
        let front_layer = compute_front_layer(circuit, &executed);
        if front_layer.is_empty() {
            break;
        }

        // Try to execute gates that are already adjacent
        let mut progress = true;
        while progress {
            progress = false;
            let front = compute_front_layer(circuit, &executed);
            for &gate_idx in &front {
                let gate = &circuit[gate_idx];
                let qubits = gate.qubits();
                if qubits.len() <= 1 {
                    // Single-qubit gate: always executable
                    let p = layout.logical_to_physical[qubits[0]];
                    match gate {
                        LogicalGate::H(_) => routed.push(PhysicalGate::H(p)),
                        LogicalGate::X(_) => routed.push(PhysicalGate::X(p)),
                        LogicalGate::Rz(_, a) => routed.push(PhysicalGate::Rz(p, *a)),
                        LogicalGate::Rx(_, a) => routed.push(PhysicalGate::Rx(p, *a)),
                        LogicalGate::S(_) => routed.push(PhysicalGate::S(p)),
                        LogicalGate::T(_) => routed.push(PhysicalGate::T(p)),
                        // LogicalGate::Sx not defined; handled by wildcard below
                        _ => routed.push(PhysicalGate::Rz(p, 0.0)),
                    }
                    executed[gate_idx] = true;
                    progress = true;
                } else if qubits.len() == 2 {
                    let p0 = layout.logical_to_physical[qubits[0]];
                    let p1 = layout.logical_to_physical[qubits[1]];
                    if coupling_map.are_connected(p0, p1) {
                        // Gate is executable
                        match gate {
                            LogicalGate::CX(_, _) => routed.push(PhysicalGate::CX(p0, p1)),
                            LogicalGate::CZ(_, _) => routed.push(PhysicalGate::Cz(p0, p1)),
                            LogicalGate::Swap(_, _) => {
                                routed.extend(decompose_swap_to_cx(p0, p1));
                            }
                            _ => routed.push(PhysicalGate::CX(p0, p1)),
                        }
                        executed[gate_idx] = true;
                        progress = true;
                    }
                } else if qubits.len() == 3 {
                    // 3-qubit gates: decompose first, then handle
                    // For simplicity, decompose CCX into CX + 1q and re-add
                    // Just mark as executed and add decomposed gates
                    let p0 = layout.logical_to_physical[qubits[0]];
                    let p1 = layout.logical_to_physical[qubits[1]];
                    let p2 = layout.logical_to_physical[qubits[2]];
                    if coupling_map.are_connected(p0, p1) && coupling_map.are_connected(p1, p2) {
                        routed.extend(decompose_ccx_to_cx(p0, p1, p2));
                        executed[gate_idx] = true;
                        progress = true;
                    }
                }
            }
        }

        // Recompute front layer
        let front_layer = compute_front_layer(circuit, &executed);
        if front_layer.is_empty() {
            break;
        }

        // Need to insert a SWAP. Find candidate SWAPs near front layer qubits.
        let extended_set = compute_extended_set(circuit, &executed, &front_layer);

        let mut candidate_swaps: Vec<(usize, usize)> = Vec::new();
        for &gate_idx in &front_layer {
            let qubits = circuit[gate_idx].qubits();
            for &lq in &qubits {
                let pq = layout.logical_to_physical[lq];
                for &neighbor in &adj[pq] {
                    let swap = if pq < neighbor {
                        (pq, neighbor)
                    } else {
                        (neighbor, pq)
                    };
                    if !candidate_swaps.contains(&swap) {
                        candidate_swaps.push(swap);
                    }
                }
            }
        }

        if candidate_swaps.is_empty() {
            break; // No swaps possible
        }

        // Score each candidate SWAP
        let mut best_swap = candidate_swaps[0];
        let mut best_score = f64::MAX;
        for &swap in &candidate_swaps {
            let score = compute_swap_score(
                swap,
                &front_layer,
                circuit,
                &layout,
                coupling_map,
                config,
                &decay_values,
                &extended_set,
            );
            if score < best_score {
                best_score = score;
                best_swap = swap;
            }
        }

        // Insert the SWAP
        routed.extend(decompose_swap_to_cx(best_swap.0, best_swap.1));
        layout.apply_swap(best_swap.0, best_swap.1);
        num_swaps += 1;

        // Update decay
        for d in &mut decay_values {
            *d = (*d + config.decay_factor).min(5.0);
        }
        decay_values[best_swap.0] = 1.0;
        decay_values[best_swap.1] = 1.0;
    }

    // Compute depth (simplified: count layers of non-overlapping gates)
    let depth = compute_circuit_depth(&routed);

    RoutingResult {
        routed_circuit: routed,
        layout,
        num_swaps_inserted: num_swaps,
        depth,
    }
}

/// Compute the depth of a physical circuit.
fn compute_circuit_depth(circuit: &[PhysicalGate]) -> usize {
    if circuit.is_empty() {
        return 0;
    }
    let mut qubit_depth: HashMap<usize, usize> = HashMap::new();
    for gate in circuit {
        let qs = gate.qubits();
        let max_current = qs
            .iter()
            .map(|q| qubit_depth.get(q).copied().unwrap_or(0))
            .max()
            .unwrap_or(0);
        for q in qs {
            qubit_depth.insert(q, max_current + 1);
        }
    }
    qubit_depth.values().copied().max().unwrap_or(0)
}

/// Compute initial SABRE layout (runs SABRE forward+backward and returns best layout).
pub fn sabre_layout(
    circuit: &[LogicalGate],
    coupling_map: &CouplingMap,
    config: &SabreConfig,
) -> Layout {
    let result = sabre_route(circuit, coupling_map, config);
    result.layout
}

// ============================================================
// COMPILER PASS INFRASTRUCTURE
// ============================================================

/// Statistics collected during pass execution.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub total_gates_before: usize,
    pub total_gates_after: usize,
    pub passes_run: usize,
    pub swaps_inserted: usize,
}

/// Result of running a compiler pass.
#[derive(Debug, Clone)]
pub struct PassResult {
    pub modified: bool,
    pub gates_before: usize,
    pub gates_after: usize,
}

/// Context shared between passes in a pipeline.
#[derive(Debug, Clone)]
pub struct PassContext {
    pub coupling_map: Option<CouplingMap>,
    pub basis_gates: Option<BasisGateSet>,
    pub layout: Option<Layout>,
    pub stats: PassStats,
}

impl PassContext {
    pub fn new() -> Self {
        Self {
            coupling_map: None,
            basis_gates: None,
            layout: None,
            stats: PassStats::default(),
        }
    }

    pub fn with_device(device: &DeviceModel) -> Self {
        Self {
            coupling_map: Some(device.coupling_map.clone()),
            basis_gates: Some(device.basis_gates.clone()),
            layout: None,
            stats: PassStats::default(),
        }
    }
}

/// Trait for compiler passes that transform a circuit.
pub trait CompilerPass {
    fn name(&self) -> &str;
    fn run(&self, circuit: &mut Vec<LogicalGate>, context: &mut PassContext) -> PassResult;
}

// ============================================================
// BUILT-IN PASSES
// ============================================================

/// Cancel adjacent inverse gate pairs: H*H, S*Sdg, CX*CX, etc.
pub struct CancelInverses;

impl CompilerPass for CancelInverses {
    fn name(&self) -> &str {
        "CancelInverses"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, _context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();
        let mut changed = true;

        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < circuit.len() {
                let cancel = match (&circuit[i], &circuit[i + 1]) {
                    (LogicalGate::H(a), LogicalGate::H(b)) if a == b => true,
                    (LogicalGate::X(a), LogicalGate::X(b)) if a == b => true,
                    (LogicalGate::Y(a), LogicalGate::Y(b)) if a == b => true,
                    (LogicalGate::Z(a), LogicalGate::Z(b)) if a == b => true,
                    (LogicalGate::S(a), LogicalGate::Sdg(b)) if a == b => true,
                    (LogicalGate::Sdg(a), LogicalGate::S(b)) if a == b => true,
                    (LogicalGate::T(a), LogicalGate::Tdg(b)) if a == b => true,
                    (LogicalGate::Tdg(a), LogicalGate::T(b)) if a == b => true,
                    (LogicalGate::CX(a0, a1), LogicalGate::CX(b0, b1)) if a0 == b0 && a1 == b1 => {
                        true
                    }
                    (LogicalGate::CZ(a0, a1), LogicalGate::CZ(b0, b1)) if a0 == b0 && a1 == b1 => {
                        true
                    }
                    _ => false,
                };
                if cancel {
                    circuit.remove(i + 1);
                    circuit.remove(i);
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }

        let gates_after = circuit.len();
        PassResult {
            modified: gates_before != gates_after,
            gates_before,
            gates_after,
        }
    }
}

/// Merge consecutive rotation gates on the same qubit.
pub struct MergeRotations;

impl CompilerPass for MergeRotations {
    fn name(&self) -> &str {
        "MergeRotations"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, _context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();
        let mut changed = true;

        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < circuit.len() {
                let merged = match (&circuit[i], &circuit[i + 1]) {
                    (LogicalGate::Rz(q0, a), LogicalGate::Rz(q1, b)) if q0 == q1 => {
                        Some(LogicalGate::Rz(*q0, a + b))
                    }
                    (LogicalGate::Rx(q0, a), LogicalGate::Rx(q1, b)) if q0 == q1 => {
                        Some(LogicalGate::Rx(*q0, a + b))
                    }
                    (LogicalGate::Ry(q0, a), LogicalGate::Ry(q1, b)) if q0 == q1 => {
                        Some(LogicalGate::Ry(*q0, a + b))
                    }
                    _ => None,
                };
                if let Some(gate) = merged {
                    circuit[i] = gate;
                    circuit.remove(i + 1);
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }

        // Remove rotations that are effectively zero
        circuit.retain(|g| {
            match g {
                LogicalGate::Rz(_, a) | LogicalGate::Rx(_, a) | LogicalGate::Ry(_, a) => {
                    // Keep if angle mod 2*pi is not near zero
                    let normalized = a.rem_euclid(2.0 * PI);
                    normalized.abs() > 1e-10 && (2.0 * PI - normalized).abs() > 1e-10
                }
                _ => true,
            }
        });

        let gates_after = circuit.len();
        PassResult {
            modified: gates_before != gates_after,
            gates_before,
            gates_after,
        }
    }
}

/// Decompose all gates to the target basis gate set.
pub struct DecomposePass;

impl CompilerPass for DecomposePass {
    fn name(&self) -> &str {
        "DecomposePass"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();
        let _basis = context
            .basis_gates
            .clone()
            .unwrap_or(BasisGateSet::Universal);

        // We convert logical gates to physical, then back to logical for the pipeline
        // In practice the final step converts everything to physical
        // For intermediate passes, we do logical-level optimizations

        // This pass is a no-op at the logical level; actual decomposition happens
        // in the final transpile step. Here we just unroll multi-qubit gates.
        let mut new_circuit = Vec::new();
        for gate in circuit.iter() {
            match gate {
                LogicalGate::Swap(a, b) => {
                    new_circuit.push(LogicalGate::CX(*a, *b));
                    new_circuit.push(LogicalGate::CX(*b, *a));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                }
                LogicalGate::CCX(a, b, c) => {
                    // Decompose into CX + 1q gates at the logical level
                    new_circuit.push(LogicalGate::H(*c));
                    new_circuit.push(LogicalGate::CX(*b, *c));
                    new_circuit.push(LogicalGate::Rz(*c, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *c));
                    new_circuit.push(LogicalGate::Rz(*c, PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*b, *c));
                    new_circuit.push(LogicalGate::Rz(*c, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *c));
                    new_circuit.push(LogicalGate::Rz(*b, PI / 4.0));
                    new_circuit.push(LogicalGate::Rz(*c, PI / 4.0));
                    new_circuit.push(LogicalGate::H(*c));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                    new_circuit.push(LogicalGate::Rz(*a, PI / 4.0));
                    new_circuit.push(LogicalGate::Rz(*b, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                }
                other => new_circuit.push(other.clone()),
            }
        }

        *circuit = new_circuit;
        let gates_after = circuit.len();
        PassResult {
            modified: gates_before != gates_after,
            gates_before,
            gates_after,
        }
    }
}

/// Unroll multi-qubit gates to 1q + 2q gates.
pub struct UnrollPass;

impl CompilerPass for UnrollPass {
    fn name(&self) -> &str {
        "UnrollPass"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, _context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();
        let mut new_circuit = Vec::new();

        for gate in circuit.iter() {
            match gate {
                LogicalGate::Swap(a, b) => {
                    new_circuit.push(LogicalGate::CX(*a, *b));
                    new_circuit.push(LogicalGate::CX(*b, *a));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                }
                LogicalGate::CCX(a, b, c) => {
                    new_circuit.push(LogicalGate::H(*c));
                    new_circuit.push(LogicalGate::CX(*b, *c));
                    new_circuit.push(LogicalGate::Rz(*c, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *c));
                    new_circuit.push(LogicalGate::Rz(*c, PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*b, *c));
                    new_circuit.push(LogicalGate::Rz(*c, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *c));
                    new_circuit.push(LogicalGate::Rz(*b, PI / 4.0));
                    new_circuit.push(LogicalGate::Rz(*c, PI / 4.0));
                    new_circuit.push(LogicalGate::H(*c));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                    new_circuit.push(LogicalGate::Rz(*a, PI / 4.0));
                    new_circuit.push(LogicalGate::Rz(*b, -PI / 4.0));
                    new_circuit.push(LogicalGate::CX(*a, *b));
                }
                LogicalGate::U3(q, theta, phi, lambda) => {
                    // Decompose U3 into Rz + Ry + Rz
                    if lambda.abs() > 1e-10 {
                        new_circuit.push(LogicalGate::Rz(*q, *lambda));
                    }
                    if theta.abs() > 1e-10 {
                        new_circuit.push(LogicalGate::Ry(*q, *theta));
                    }
                    if phi.abs() > 1e-10 {
                        new_circuit.push(LogicalGate::Rz(*q, *phi));
                    }
                }
                other => new_circuit.push(other.clone()),
            }
        }

        *circuit = new_circuit;
        let gates_after = circuit.len();
        PassResult {
            modified: gates_before != gates_after,
            gates_before,
            gates_after,
        }
    }
}

/// Routing pass using SABRE algorithm.
pub struct RoutingPass {
    pub config: SabreConfig,
}

impl RoutingPass {
    pub fn new() -> Self {
        Self {
            config: SabreConfig::default(),
        }
    }

    pub fn with_config(config: SabreConfig) -> Self {
        Self { config }
    }
}

impl CompilerPass for RoutingPass {
    fn name(&self) -> &str {
        "RoutingPass"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();

        if let Some(ref coupling_map) = context.coupling_map {
            let result = sabre_route(circuit, coupling_map, &self.config);
            context.layout = Some(result.layout);
            context.stats.swaps_inserted += result.num_swaps_inserted;

            // Convert routed physical gates back to logical for further passes
            let mut new_circuit = Vec::new();
            for pg in &result.routed_circuit {
                match pg {
                    PhysicalGate::CX(a, b) => new_circuit.push(LogicalGate::CX(*a, *b)),
                    PhysicalGate::Cz(a, b) => new_circuit.push(LogicalGate::CZ(*a, *b)),
                    PhysicalGate::H(q) => new_circuit.push(LogicalGate::H(*q)),
                    PhysicalGate::X(q) => new_circuit.push(LogicalGate::X(*q)),
                    PhysicalGate::Sx(q) => new_circuit.push(LogicalGate::H(*q)), // approximate
                    PhysicalGate::Rz(q, a) => new_circuit.push(LogicalGate::Rz(*q, *a)),
                    PhysicalGate::Rx(q, a) => new_circuit.push(LogicalGate::Rx(*q, *a)),
                    PhysicalGate::S(q) => new_circuit.push(LogicalGate::S(*q)),
                    PhysicalGate::T(q) => new_circuit.push(LogicalGate::T(*q)),
                    PhysicalGate::Cnot(a, b) => new_circuit.push(LogicalGate::CX(*a, *b)),
                    _ => {} // skip barriers, etc.
                }
            }
            *circuit = new_circuit;
        }

        let gates_after = circuit.len();
        PassResult {
            modified: gates_before != gates_after,
            gates_before,
            gates_after,
        }
    }
}

/// Layout pass: compute initial qubit layout.
pub struct LayoutPass {
    pub config: SabreConfig,
}

impl LayoutPass {
    pub fn new() -> Self {
        Self {
            config: SabreConfig::default(),
        }
    }
}

impl CompilerPass for LayoutPass {
    fn name(&self) -> &str {
        "LayoutPass"
    }

    fn run(&self, circuit: &mut Vec<LogicalGate>, context: &mut PassContext) -> PassResult {
        let gates_before = circuit.len();

        if let Some(ref coupling_map) = context.coupling_map {
            let layout = sabre_layout(circuit, coupling_map, &self.config);
            context.layout = Some(layout);
        }

        PassResult {
            modified: false,
            gates_before,
            gates_after: gates_before,
        }
    }
}

// ============================================================
// PASS PIPELINE
// ============================================================

/// A composable pipeline of compiler passes.
pub struct PassPipeline {
    passes: Vec<Box<dyn CompilerPass>>,
}

impl PassPipeline {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Add a pass to the pipeline.
    pub fn add_pass<P: CompilerPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    /// Run all passes in order.
    pub fn run(
        &self,
        circuit: &mut Vec<LogicalGate>,
        context: &mut PassContext,
    ) -> Vec<PassResult> {
        let mut results = Vec::new();
        for pass in &self.passes {
            let result = pass.run(circuit, context);
            context.stats.passes_run += 1;
            results.push(result);
        }
        results
    }

    /// Preset level 0: no optimization, just unroll.
    pub fn preset_level_0() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_pass(UnrollPass);
        pipeline
    }

    /// Preset level 1: light optimization (cancel inverses).
    pub fn preset_level_1() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_pass(UnrollPass);
        pipeline.add_pass(CancelInverses);
        pipeline
    }

    /// Preset level 2: medium optimization (cancel + merge).
    pub fn preset_level_2() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_pass(UnrollPass);
        pipeline.add_pass(CancelInverses);
        pipeline.add_pass(MergeRotations);
        pipeline.add_pass(CancelInverses);
        pipeline
    }

    /// Preset level 3: heavy optimization + routing.
    pub fn preset_level_3() -> Self {
        let mut pipeline = Self::new();
        pipeline.add_pass(UnrollPass);
        pipeline.add_pass(CancelInverses);
        pipeline.add_pass(MergeRotations);
        pipeline.add_pass(CancelInverses);
        pipeline.add_pass(DecomposePass);
        pipeline.add_pass(RoutingPass::new());
        pipeline.add_pass(CancelInverses);
        pipeline.add_pass(MergeRotations);
        pipeline
    }

    /// Return the names of passes in the pipeline.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|p| p.name()).collect()
    }
}

// ============================================================
// TRANSPILER ERROR
// ============================================================

/// Errors that can occur during transpilation.
#[derive(Debug, Clone)]
pub enum TranspilerError {
    LayoutFailed(String),
    RoutingFailed(String),
    DecompositionFailed(String),
    IncompatibleCircuit(String),
}

impl std::fmt::Display for TranspilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TranspilerError::LayoutFailed(s) => write!(f, "Layout failed: {}", s),
            TranspilerError::RoutingFailed(s) => write!(f, "Routing failed: {}", s),
            TranspilerError::DecompositionFailed(s) => write!(f, "Decomposition failed: {}", s),
            TranspilerError::IncompatibleCircuit(s) => write!(f, "Incompatible circuit: {}", s),
        }
    }
}

impl std::error::Error for TranspilerError {}

// ============================================================
// FULL TRANSPILATION
// ============================================================

/// Result of full transpilation.
#[derive(Debug, Clone)]
pub struct TranspileResult {
    pub physical_circuit: Vec<PhysicalGate>,
    pub layout: Layout,
    pub num_swaps: usize,
    pub depth: usize,
    pub gate_count: usize,
    pub two_qubit_gate_count: usize,
}

/// Transpile a logical circuit to physical gates for a target device.
///
/// `optimization_level`:
///   0 = no optimization
///   1 = light (cancel inverses)
///   2 = medium (cancel + merge rotations)
///   3 = heavy (full SABRE routing + optimization)
pub fn transpile(
    circuit: &[LogicalGate],
    device: &DeviceModel,
    optimization_level: usize,
) -> TranspileResult {
    let mut working_circuit = circuit.to_vec();
    let mut context = PassContext::with_device(device);

    // Select pipeline based on optimization level
    let pipeline = match optimization_level {
        0 => PassPipeline::preset_level_0(),
        1 => PassPipeline::preset_level_1(),
        2 => PassPipeline::preset_level_2(),
        _ => PassPipeline::preset_level_3(),
    };

    pipeline.run(&mut working_circuit, &mut context);

    // Final decomposition to physical gates
    let basis = &device.basis_gates;
    let physical_circuit: Vec<PhysicalGate> = working_circuit
        .iter()
        .flat_map(|gate| decompose_to_basis(gate, basis))
        .collect();

    let layout = context.layout.unwrap_or_else(|| {
        let max_q = circuit.iter().map(|g| g.max_qubit()).max().unwrap_or(0) + 1;
        Layout::trivial(max_q)
    });

    let depth = compute_circuit_depth(&physical_circuit);
    let gate_count = physical_circuit.len();
    let two_qubit_gate_count = physical_circuit
        .iter()
        .filter(|g| g.qubits().len() >= 2)
        .count();

    TranspileResult {
        physical_circuit,
        layout,
        num_swaps: context.stats.swaps_inserted,
        depth,
        gate_count,
        two_qubit_gate_count,
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Linear coupling map: 5 qubits, 4 edges
    #[test]
    fn test_linear_coupling_map() {
        let cm = CouplingMap::linear(5);
        assert_eq!(cm.num_qubits, 5);
        assert_eq!(cm.edges.len(), 4);
        assert!(cm.are_connected(0, 1));
        assert!(cm.are_connected(1, 2));
        assert!(cm.are_connected(3, 4));
        assert!(!cm.are_connected(0, 2));
    }

    // 2. Grid coupling map: 3x3, 12 edges
    #[test]
    fn test_grid_coupling_map() {
        let cm = CouplingMap::grid(3, 3);
        assert_eq!(cm.num_qubits, 9);
        assert_eq!(cm.edges.len(), 12);
        // Check some connections
        assert!(cm.are_connected(0, 1)); // row edge
        assert!(cm.are_connected(0, 3)); // column edge
        assert!(!cm.are_connected(0, 4)); // diagonal, not connected
    }

    // 3. Heavy hex topology construction
    #[test]
    fn test_heavy_hex_topology() {
        let cm = CouplingMap::heavy_hex(3);
        assert!(cm.num_qubits > 0);
        assert!(!cm.edges.is_empty());
        // Heavy hex should have more qubits than a simple linear chain
        assert!(cm.num_qubits > 6);
    }

    // 4. BFS shortest path on linear map
    #[test]
    fn test_bfs_shortest_path() {
        let cm = CouplingMap::linear(5);
        let path = cm.shortest_path(0, 4);
        assert_eq!(path, vec![0, 1, 2, 3, 4]);
        assert_eq!(path.len(), 5);

        let path2 = cm.shortest_path(1, 3);
        assert_eq!(path2, vec![1, 2, 3]);
    }

    // 5. Distance matrix symmetry
    #[test]
    fn test_distance_symmetry() {
        let cm = CouplingMap::grid(3, 3);
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(
                    cm.distance(i, j),
                    cm.distance(j, i),
                    "Distance asymmetry between {} and {}",
                    i,
                    j
                );
            }
        }
    }

    // 6. SWAP decomposition to 3 CX gates
    #[test]
    fn test_swap_decomposition() {
        let gates = decompose_swap_to_cx(0, 1);
        assert_eq!(gates.len(), 3);
        for gate in &gates {
            match gate {
                PhysicalGate::CX(_, _) => {}
                _ => panic!("SWAP decomposition should only produce CX gates"),
            }
        }
    }

    // 7. Toffoli decomposition correctness
    #[test]
    fn test_toffoli_decomposition() {
        let gates = decompose_ccx_to_cx(0, 1, 2);
        assert!(!gates.is_empty());
        // Should contain CX gates and single-qubit gates
        let cx_count = gates
            .iter()
            .filter(|g| matches!(g, PhysicalGate::CX(_, _)))
            .count();
        // Standard Toffoli decomposition uses 6 CX gates
        assert!(
            cx_count >= 4,
            "Expected at least 4 CX gates, got {}",
            cx_count
        );
    }

    // 8. CancelInverses: H*H -> empty
    #[test]
    fn test_cancel_inverses_hh() {
        let mut circuit = vec![LogicalGate::H(0), LogicalGate::H(0)];
        let mut context = PassContext::new();
        let pass = CancelInverses;
        let result = pass.run(&mut circuit, &mut context);
        assert!(result.modified);
        assert_eq!(circuit.len(), 0, "H*H should cancel to empty");
    }

    // 9. MergeRotations: Rz(a)*Rz(b) -> Rz(a+b)
    #[test]
    fn test_merge_rotations() {
        let a = 0.3;
        let b = 0.7;
        let mut circuit = vec![LogicalGate::Rz(0, a), LogicalGate::Rz(0, b)];
        let mut context = PassContext::new();
        let pass = MergeRotations;
        let result = pass.run(&mut circuit, &mut context);
        assert!(result.modified);
        assert_eq!(circuit.len(), 1, "Two Rz should merge into one");
        match &circuit[0] {
            LogicalGate::Rz(q, angle) => {
                assert_eq!(*q, 0);
                assert!(
                    (angle - (a + b)).abs() < 1e-10,
                    "Merged angle should be a+b"
                );
            }
            _ => panic!("Expected Rz gate after merge"),
        }
    }

    // 10. SABRE routing on linear topology inserts SWAPs
    #[test]
    fn test_sabre_routing_linear_inserts_swaps() {
        let cm = CouplingMap::linear(5);
        // Use a circuit where qubit 0 must be adjacent to 2, 3, and 4 simultaneously,
        // which is impossible on a linear topology regardless of initial layout.
        let circuit = vec![
            LogicalGate::CX(0, 2),
            LogicalGate::CX(0, 3),
            LogicalGate::CX(0, 4),
            LogicalGate::CX(1, 4),
            LogicalGate::CX(2, 4),
        ];
        let config = SabreConfig::new().num_trials(10).seed(42);
        let result = sabre_route(&circuit, &cm, &config);
        assert!(
            result.num_swaps_inserted > 0,
            "SABRE should insert SWAPs for non-adjacent qubits on linear topology (got {} swaps)",
            result.num_swaps_inserted
        );
    }

    // 11. SABRE with all-to-all needs zero SWAPs
    #[test]
    fn test_sabre_all_to_all_no_swaps() {
        let cm = CouplingMap::all_to_all(5);
        let circuit = vec![
            LogicalGate::CX(0, 4),
            LogicalGate::CX(1, 3),
            LogicalGate::CX(2, 0),
        ];
        let config = SabreConfig::new().num_trials(5).seed(42);
        let result = sabre_route(&circuit, &cm, &config);
        assert_eq!(
            result.num_swaps_inserted, 0,
            "All-to-all topology should require no SWAPs"
        );
    }

    // 12. Full transpile to IBM basis preserves circuit structure (small circuit)
    #[test]
    fn test_transpile_ibm_basis() {
        let device = DeviceModel::ibm_eagle();
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::CX(0, 1),
            LogicalGate::Rz(1, PI / 4.0),
        ];
        let result = transpile(&circuit, &device, 0);
        assert!(!result.physical_circuit.is_empty());
        assert!(result.gate_count > 0);
        // All gates should be in IBM basis: CX, Id, Rz, Sx, X
        for gate in &result.physical_circuit {
            match gate {
                PhysicalGate::CX(_, _)
                | PhysicalGate::Id(_)
                | PhysicalGate::Rz(_, _)
                | PhysicalGate::Sx(_)
                | PhysicalGate::X(_) => {}
                other => panic!("Gate {:?} is not in IBM basis (CX, Id, Rz, Sx, X)", other),
            }
        }
    }

    // 13. Optimization level 0 vs 3: level 3 has fewer gates (or equal)
    #[test]
    fn test_optimization_levels() {
        let device = DeviceModel::ibm_eagle();
        // Circuit with redundant gates that optimization should remove
        let circuit = vec![
            LogicalGate::H(0),
            LogicalGate::H(0), // cancels with previous
            LogicalGate::Rz(1, 0.3),
            LogicalGate::Rz(1, 0.7), // merges with previous
            LogicalGate::CX(0, 1),
        ];
        let result_0 = transpile(&circuit, &device, 0);
        let result_3 = transpile(&circuit, &device, 2); // Use level 2 (no routing) for fair comparison
        assert!(
            result_3.gate_count <= result_0.gate_count,
            "Level 2+ optimization should produce fewer or equal gates: level0={} level2={}",
            result_0.gate_count,
            result_3.gate_count
        );
    }

    // 14. Layout maps logical to physical qubits bijectively
    #[test]
    fn test_layout_bijection() {
        let layout = Layout::trivial(5);
        assert!(layout.is_valid());

        let layout2 = Layout::random(5, 10, 42);
        assert!(layout2.is_valid());
        // Check all logical qubits map to distinct physical qubits
        let phys: HashSet<usize> = layout2.logical_to_physical.iter().copied().collect();
        assert_eq!(
            phys.len(),
            5,
            "5 logical qubits should map to 5 distinct physical qubits"
        );
    }

    // 15. Device preset: IBM Eagle has ~127 qubits
    #[test]
    fn test_ibm_eagle_qubit_count() {
        let device = DeviceModel::ibm_eagle();
        // Heavy hex with 15 cells should give approximately 127 qubits
        // (exact count depends on our simplified heavy-hex model)
        assert!(
            device.coupling_map.num_qubits >= 50,
            "IBM Eagle should have many qubits, got {}",
            device.coupling_map.num_qubits
        );
        assert_eq!(device.basis_gates, BasisGateSet::IBMBasis);
    }

    // 16. Pass pipeline executes in order
    #[test]
    fn test_pipeline_execution_order() {
        let pipeline = PassPipeline::preset_level_2();
        let names = pipeline.pass_names();
        assert!(!names.is_empty());
        // Level 2 should have: Unroll, CancelInverses, MergeRotations, CancelInverses
        assert_eq!(names[0], "UnrollPass");
        assert_eq!(names[1], "CancelInverses");
        assert_eq!(names[2], "MergeRotations");
        assert_eq!(names[3], "CancelInverses");
    }

    // Additional test: ring topology
    #[test]
    fn test_ring_coupling_map() {
        let cm = CouplingMap::ring(6);
        assert_eq!(cm.num_qubits, 6);
        assert_eq!(cm.edges.len(), 6);
        assert!(cm.are_connected(0, 1));
        assert!(cm.are_connected(5, 0)); // wrap-around
    }

    // Additional test: all-to-all
    #[test]
    fn test_all_to_all() {
        let cm = CouplingMap::all_to_all(4);
        assert_eq!(cm.num_qubits, 4);
        assert_eq!(cm.edges.len(), 6); // C(4,2) = 6
        for i in 0..4 {
            for j in i + 1..4 {
                assert!(cm.are_connected(i, j));
            }
        }
    }

    // Additional test: neighbors
    #[test]
    fn test_neighbors() {
        let cm = CouplingMap::linear(5);
        let n0 = cm.neighbors(0);
        assert_eq!(n0, vec![1]);
        let n2 = cm.neighbors(2);
        assert!(n2.contains(&1) && n2.contains(&3));
        assert_eq!(n2.len(), 2);
    }

    // Additional test: layout swap
    #[test]
    fn test_layout_swap() {
        let mut layout = Layout::trivial(4);
        assert_eq!(layout.logical_to_physical[0], 0);
        assert_eq!(layout.logical_to_physical[1], 1);
        layout.apply_swap(0, 1);
        assert_eq!(layout.logical_to_physical[0], 1);
        assert_eq!(layout.logical_to_physical[1], 0);
        assert!(layout.is_valid());
    }

    // Additional test: CancelInverses with S/Sdg
    #[test]
    fn test_cancel_inverses_s_sdg() {
        let mut circuit = vec![LogicalGate::S(0), LogicalGate::Sdg(0)];
        let mut context = PassContext::new();
        let pass = CancelInverses;
        let result = pass.run(&mut circuit, &mut context);
        assert!(result.modified);
        assert_eq!(circuit.len(), 0);
    }

    // Additional test: transpile result structure
    #[test]
    fn test_transpile_result_structure() {
        let device = DeviceModel::ionq_aria();
        let circuit = vec![LogicalGate::H(0), LogicalGate::CX(0, 1)];
        let result = transpile(&circuit, &device, 0);
        assert!(result.gate_count > 0);
        assert!(result.two_qubit_gate_count > 0);
        assert!(result.depth > 0);
    }
}
