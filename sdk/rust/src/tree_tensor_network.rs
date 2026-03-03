//! Tree Tensor Network (TTN) Backend
//!
//! Hierarchical tensor network with correct complex-valued tensor contraction,
//! full-contraction two-qubit gate application, real entanglement entropy
//! computation, and proper statevector contraction.
//!
//! # Architecture
//!
//! TTN is a hierarchical tensor network where:
//! - Physical qubits are at leaves
//! - Tensors form a binary tree structure
//! - Bond dimension chi controls entanglement capacity
//! - Two-qubit gates use full contraction + tree re-decomposition
//!
//! Tensors store complex amplitudes as `[f64; 2]` pairs `[re, im]`.
//!
//! # Performance
//!
//! - Memory: O(n * chi^2) for n qubits
//! - Single-qubit gate: O(chi) at leaf
//! - Two-qubit gate: O(2^n) via full contraction (exact for <= ~20 qubits)
//! - Statevector extraction: O(2^n) (full contraction)
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::tree_tensor_network::{TTN, TTNConfig};
//!
//! let mut ttn = TTN::new(TTNConfig::new(4, 32));
//! ttn.apply_h(0);
//! ttn.apply_cnot(0, 1);
//! let sv = ttn.to_statevector();
//! ```

use rand::Rng;
use std::collections::HashMap;

// ===========================================================================
// COMPLEX ARITHMETIC HELPERS
// ===========================================================================

const ZERO: [f64; 2] = [0.0, 0.0];
const ONE: [f64; 2] = [1.0, 0.0];

#[inline]
fn cmul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

#[inline]
fn cadd(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
fn cnorm_sq(a: [f64; 2]) -> f64 {
    a[0] * a[0] + a[1] * a[1]
}

#[inline]
fn cscale(s: f64, a: [f64; 2]) -> [f64; 2] {
    [s * a[0], s * a[1]]
}

#[inline]
fn cconj(a: [f64; 2]) -> [f64; 2] {
    [a[0], -a[1]]
}

// ===========================================================================
// COMPACT SVD FOR COMPLEX MATRICES
// ===========================================================================

/// Compute compact SVD of an m x n complex matrix (row-major).
/// Returns (U: m x k, sigma: k, Vt: k x n) where k = min(m, n).
/// Uses eigendecomposition of the smaller Gram matrix.
fn csvd(mat: &[[f64; 2]], m: usize, n: usize) -> (Vec<[f64; 2]>, Vec<f64>, Vec<[f64; 2]>) {
    let k = m.min(n);
    if k == 0 {
        return (vec![], vec![], vec![]);
    }

    if m >= n {
        // Form B = A^H A (n x n hermitian positive semidefinite)
        let mut b = vec![ZERO; n * n];
        for i in 0..n {
            for j in i..n {
                let mut s = ZERO;
                for r in 0..m {
                    s = cadd(s, cmul(cconj(mat[r * n + i]), mat[r * n + j]));
                }
                b[i * n + j] = s;
                if i != j {
                    b[j * n + i] = cconj(s);
                }
            }
        }

        let (evals, evecs) = hermitian_eigen(&b, n);

        // Sort eigenvalues descending
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = vec![0.0f64; k];
        let mut vt = vec![ZERO; k * n];
        for rank in 0..k {
            let ii = idx[rank];
            sigma[rank] = if evals[ii] > 1e-30 {
                evals[ii].sqrt()
            } else {
                0.0
            };
            for j in 0..n {
                // V^T row = conjugate of eigenvector column
                // evecs is column-major: evecs[row * n + col]
                vt[rank * n + j] = evecs[j * n + ii];
            }
        }

        // U = A V Sigma^{-1}
        let mut u = vec![ZERO; m * k];
        for rank in 0..k {
            if sigma[rank] > 1e-15 {
                let inv_s = 1.0 / sigma[rank];
                for r in 0..m {
                    let mut s = ZERO;
                    for c in 0..n {
                        // V column = conj(V^T row)
                        s = cadd(s, cmul(mat[r * n + c], cconj(vt[rank * n + c])));
                    }
                    u[r * k + rank] = cscale(inv_s, s);
                }
            }
        }

        (u, sigma, vt)
    } else {
        // m < n: work with A A^H (m x m) instead
        let mut b = vec![ZERO; m * m];
        for i in 0..m {
            for j in i..m {
                let mut s = ZERO;
                for c in 0..n {
                    s = cadd(s, cmul(mat[i * n + c], cconj(mat[j * n + c])));
                }
                b[i * m + j] = s;
                if i != j {
                    b[j * m + i] = cconj(s);
                }
            }
        }

        let (evals, evecs) = hermitian_eigen(&b, m);

        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sigma = vec![0.0f64; k];
        let mut u = vec![ZERO; m * k];
        for rank in 0..k {
            let ii = idx[rank];
            sigma[rank] = if evals[ii] > 1e-30 {
                evals[ii].sqrt()
            } else {
                0.0
            };
            for r in 0..m {
                u[r * k + rank] = evecs[r * m + ii];
            }
        }

        // V^H = Sigma^{-1} U^H A
        let mut vt = vec![ZERO; k * n];
        for rank in 0..k {
            if sigma[rank] > 1e-15 {
                let inv_s = 1.0 / sigma[rank];
                for c in 0..n {
                    let mut s = ZERO;
                    for r in 0..m {
                        s = cadd(s, cmul(cconj(u[r * k + rank]), mat[r * n + c]));
                    }
                    vt[rank * n + c] = cscale(inv_s, s);
                }
            }
        }

        (u, sigma, vt)
    }
}

/// Jacobi eigendecomposition of an n x n Hermitian matrix.
/// Returns (eigenvalues, eigenvectors) where eigenvectors are stored
/// column-major: evecs[row * n + col] is row-th element of col-th eigenvector.
fn hermitian_eigen(mat: &[[f64; 2]], n: usize) -> (Vec<f64>, Vec<[f64; 2]>) {
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![mat[0][0]], vec![ONE]);
    }

    let mut a = mat.to_vec();
    let mut v = vec![ZERO; n * n];
    for i in 0..n {
        v[i * n + i] = ONE;
    }

    for _sweep in 0..200 {
        // Compute off-diagonal norm
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += cnorm_sq(a[i * n + j]);
            }
        }
        if off < 1e-28 {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                let apq_norm = cnorm_sq(apq).sqrt();
                if apq_norm < 1e-30 {
                    continue;
                }

                // Phase factor to make apq real and positive
                let phase = [apq[0] / apq_norm, apq[1] / apq_norm];
                let phase_c = cconj(phase);

                let app = a[p * n + p][0];
                let aqq = a[q * n + q][0];

                // 2x2 real symmetric eigenvalue problem for [app, apq_norm; apq_norm, aqq]
                let tau = (aqq - app) / (2.0 * apq_norm);
                let t = if tau.abs() > 1e15 {
                    0.5 / tau
                } else {
                    let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
                    sign / (tau.abs() + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update matrix: apply rotation to rows/cols p and q
                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i * n + p];
                        let aiq = a[i * n + q];
                        let new_ip = cadd(cscale(c, aip), cscale(-s, cmul(phase_c, aiq)));
                        let new_iq = cadd(cscale(s, cmul(phase, aip)), cscale(c, aiq));
                        a[i * n + p] = new_ip;
                        a[p * n + i] = cconj(new_ip);
                        a[i * n + q] = new_iq;
                        a[q * n + i] = cconj(new_iq);
                    }
                }

                a[p * n + p] = [app - t * apq_norm, 0.0];
                a[q * n + q] = [aqq + t * apq_norm, 0.0];
                a[p * n + q] = ZERO;
                a[q * n + p] = ZERO;

                // Update eigenvector matrix
                for i in 0..n {
                    let vip = v[i * n + p];
                    let viq = v[i * n + q];
                    v[i * n + p] = cadd(cscale(c, vip), cscale(-s, cmul(phase_c, viq)));
                    v[i * n + q] = cadd(cscale(s, cmul(phase, vip)), cscale(c, viq));
                }
            }
        }
    }

    let evals: Vec<f64> = (0..n).map(|i| a[i * n + i][0]).collect();
    (evals, v)
}

// ===========================================================================
// CONFIGURATION
// ===========================================================================

/// Configuration for Tree Tensor Network.
#[derive(Clone, Debug)]
pub struct TTNConfig {
    /// Number of physical qubits.
    pub n_qubits: usize,
    /// Maximum bond dimension.
    pub max_bond_dim: usize,
    /// Truncation threshold.
    pub truncation_threshold: f64,
    /// Enable SV truncation (vs fixed bond dim).
    pub adaptive_truncation: bool,
    /// Random seed for initialization.
    pub seed: u64,
}

impl TTNConfig {
    /// Create a new TTN config.
    pub fn new(n_qubits: usize, max_bond_dim: usize) -> Self {
        Self {
            n_qubits,
            max_bond_dim,
            truncation_threshold: 1e-10,
            adaptive_truncation: true,
            seed: 42,
        }
    }

    /// Set truncation threshold.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.truncation_threshold = threshold;
        self
    }
}

// ===========================================================================
// TREE NODE
// ===========================================================================

/// A node in the TTN tree.
///
/// Tensor layout for complex amplitudes using `[f64; 2]` pairs `[re, im]`.
///
/// **Leaf node**: tensor has shape `[phys_dim, bond_up]`, stored row-major.
///
/// **Internal node**: tensor has shape `[bond_left, bond_right, bond_up]`.
///
/// **Root**: `bond_up = 1` (no parent edge).
#[derive(Clone, Debug)]
pub struct TTNNode {
    /// Node ID.
    pub id: usize,
    /// Complex tensor data (flattened, row-major).
    pub tensor: Vec<[f64; 2]>,
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Children node IDs (empty for leaves).
    pub children: Vec<usize>,
    /// Parent node ID (None for root).
    pub parent: Option<usize>,
    /// Physical qubit index (Some for leaves).
    pub physical_qubit: Option<usize>,
    /// Bond dimension to parent.
    pub bond_dim: usize,
}

impl TTNNode {
    /// Create a leaf node for a physical qubit in the |0> state.
    /// Tensor shape: `[2, 1]`.
    pub fn leaf(id: usize, qubit: usize) -> Self {
        Self {
            id,
            tensor: vec![ONE, ZERO],
            shape: vec![2, 1],
            children: vec![],
            parent: None,
            physical_qubit: Some(qubit),
            bond_dim: 1,
        }
    }

    /// Create an internal node connecting two children.
    /// Tensor shape: `[bond_left, bond_right, bond_up]`.
    /// Initialized for product state: tensor[0,0,0] = 1.
    pub fn internal(id: usize, left: usize, right: usize, bond_up: usize) -> Self {
        let bu = bond_up.max(1);
        let mut tensor = vec![ZERO; bu]; // [1, 1, bu] flattened; only [0,0,0]=1
        tensor[0] = ONE;
        Self {
            id,
            tensor,
            shape: vec![1, 1, bu],
            children: vec![left, right],
            parent: None,
            physical_qubit: None,
            bond_dim: bu,
        }
    }

    /// Get the number of legs (indices).
    pub fn num_legs(&self) -> usize {
        self.shape.len()
    }

    /// Get total tensor size.
    pub fn size(&self) -> usize {
        self.tensor.len()
    }
}

// ===========================================================================
// TREE TENSOR NETWORK
// ===========================================================================

/// Tree Tensor Network quantum state.
pub struct TTN {
    config: TTNConfig,
    /// All nodes in the tree, indexed by node ID.
    nodes: Vec<TTNNode>,
    /// Root node index.
    root: usize,
    /// Map from physical qubit to leaf node ID.
    qubit_to_leaf: HashMap<usize, usize>,
    /// Normalization flag.
    normalized: bool,
}

impl TTN {
    /// Create a new TTN in the |0...0> state.
    pub fn new(config: TTNConfig) -> Self {
        let n = config.n_qubits;
        let mut nodes = Vec::new();
        let mut qubit_to_leaf = HashMap::new();

        for q in 0..n {
            nodes.push(TTNNode::leaf(q, q));
            qubit_to_leaf.insert(q, q);
        }

        // Build binary tree bottom-up by pairing nodes
        let mut current_level: Vec<usize> = (0..n).collect();
        let mut next_id = n;

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                if chunk.len() == 2 {
                    let (left, right) = (chunk[0], chunk[1]);
                    let node = TTNNode::internal(next_id, left, right, 1);
                    nodes[left].parent = Some(next_id);
                    nodes[right].parent = Some(next_id);
                    next_level.push(next_id);
                    nodes.push(node);
                    next_id += 1;
                } else {
                    next_level.push(chunk[0]);
                }
            }
            current_level = next_level;
        }

        let root = current_level[0];
        Self {
            config,
            nodes,
            root,
            qubit_to_leaf,
            normalized: true,
        }
    }

    /// Number of qubits.
    pub fn n_qubits(&self) -> usize {
        self.config.n_qubits
    }

    /// Collect qubit indices in the subtree rooted at `node_id`, in tree
    /// traversal order (left-first DFS). This defines a canonical ordering
    /// for the statevector.
    fn subtree_qubits(&self, node_id: usize) -> Vec<usize> {
        let node = &self.nodes[node_id];
        if node.children.is_empty() {
            vec![node.physical_qubit.unwrap()]
        } else {
            let mut qs = self.subtree_qubits(node.children[0]);
            qs.extend(self.subtree_qubits(node.children[1]));
            qs
        }
    }

    // -----------------------------------------------------------------------
    // Single-qubit gates
    // -----------------------------------------------------------------------

    /// Apply a single-qubit gate to a leaf node.
    ///
    /// The gate is a 2x2 complex matrix in row-major: `[g00, g01, g10, g11]`.
    /// Leaf shape: `[2, bond_up]`. Contraction:
    ///   new[i, b] = sum_j gate[i*2+j] * old[j, b]
    pub fn apply_gate_1q(&mut self, gate: &[[f64; 2]; 4], qubit: usize) {
        let leaf_id = self.qubit_to_leaf[&qubit];
        let bond_up = self.nodes[leaf_id].shape[1];
        let old = self.nodes[leaf_id].tensor.clone();
        let mut new_t = vec![ZERO; 2 * bond_up];
        for i in 0..2 {
            for b in 0..bond_up {
                let mut s = ZERO;
                for j in 0..2 {
                    s = cadd(s, cmul(gate[i * 2 + j], old[j * bond_up + b]));
                }
                new_t[i * bond_up + b] = s;
            }
        }
        self.nodes[leaf_id].tensor = new_t;
        self.normalized = false;
    }

    /// Apply Hadamard gate: H = (1/sqrt2) [[1, 1], [1, -1]].
    pub fn apply_h(&mut self, qubit: usize) {
        let s = 1.0 / 2.0f64.sqrt();
        self.apply_gate_1q(&[[s, 0.0], [s, 0.0], [s, 0.0], [-s, 0.0]], qubit);
    }

    /// Apply Pauli-X: [[0,1],[1,0]].
    pub fn apply_x(&mut self, qubit: usize) {
        self.apply_gate_1q(&[ZERO, ONE, ONE, ZERO], qubit);
    }

    /// Apply Pauli-Y: [[0,-i],[i,0]].
    pub fn apply_y(&mut self, qubit: usize) {
        self.apply_gate_1q(&[ZERO, [0.0, -1.0], [0.0, 1.0], ZERO], qubit);
    }

    /// Apply Pauli-Z: [[1,0],[0,-1]].
    pub fn apply_z(&mut self, qubit: usize) {
        self.apply_gate_1q(&[ONE, ZERO, ZERO, [-1.0, 0.0]], qubit);
    }

    /// Apply S gate: diag(1, i).
    pub fn apply_s(&mut self, qubit: usize) {
        self.apply_gate_1q(&[ONE, ZERO, ZERO, [0.0, 1.0]], qubit);
    }

    /// Apply T gate: diag(1, e^{i pi/4}).
    pub fn apply_t(&mut self, qubit: usize) {
        let c = std::f64::consts::FRAC_PI_4.cos();
        let s = std::f64::consts::FRAC_PI_4.sin();
        self.apply_gate_1q(&[ONE, ZERO, ZERO, [c, s]], qubit);
    }

    /// Apply Rx(theta) = [[cos t/2, -i sin t/2],[-i sin t/2, cos t/2]].
    pub fn apply_rx(&mut self, qubit: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_gate_1q(&[[c, 0.0], [0.0, -s], [0.0, -s], [c, 0.0]], qubit);
    }

    /// Apply Ry(theta) = [[cos t/2, -sin t/2],[sin t/2, cos t/2]].
    pub fn apply_ry(&mut self, qubit: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_gate_1q(&[[c, 0.0], [-s, 0.0], [s, 0.0], [c, 0.0]], qubit);
    }

    /// Apply Rz(theta) = diag(e^{-i t/2}, e^{i t/2}).
    pub fn apply_rz(&mut self, qubit: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        self.apply_gate_1q(&[[c, -s], ZERO, ZERO, [c, s]], qubit);
    }

    // -----------------------------------------------------------------------
    // Two-qubit gates
    // -----------------------------------------------------------------------

    /// Apply a two-qubit gate (4x4 complex matrix, row-major, basis |00>,|01>,|10>,|11>).
    pub fn apply_gate_2q(&mut self, gate: &[[f64; 2]; 16], q1: usize, q2: usize) {
        self.apply_two_qubit_gate_via_sv(gate, q1, q2);
    }

    /// Apply CNOT(control, target).
    pub fn apply_cnot(&mut self, control: usize, target: usize) {
        let mut g = [ZERO; 16];
        g[0 * 4 + 0] = ONE; // |00> -> |00>
        g[1 * 4 + 1] = ONE; // |01> -> |01>
        g[2 * 4 + 3] = ONE; // |10> -> |11>
        g[3 * 4 + 2] = ONE; // |11> -> |10>
        self.apply_two_qubit_gate_via_sv(&g, control, target);
    }

    /// Apply a two-qubit gate by contracting to statevector, applying the
    /// gate, and re-decomposing into the tree.
    fn apply_two_qubit_gate_via_sv(
        &mut self,
        gate: &[[f64; 2]; 16],
        q1: usize,
        q2: usize,
    ) {
        let n = self.config.n_qubits;
        let dim = 1usize << n;
        let sv = self.contract_to_statevector();

        // Apply 4x4 gate on qubits q1, q2.
        // Our statevector index bits: bit q corresponds to qubit q.
        let mut new_sv = vec![ZERO; dim];
        for idx in 0..dim {
            if cnorm_sq(sv[idx]) < 1e-30 {
                continue;
            }
            let b1 = (idx >> q1) & 1;
            let b2 = (idx >> q2) & 1;
            let in_pair = b1 * 2 + b2;

            for out_pair in 0..4usize {
                let g = gate[out_pair * 4 + in_pair];
                if cnorm_sq(g) < 1e-30 {
                    continue;
                }
                let ob1 = (out_pair >> 1) & 1;
                let ob2 = out_pair & 1;
                let out_idx = (idx & !(1usize << q1) & !(1usize << q2))
                    | (ob1 << q1)
                    | (ob2 << q2);
                new_sv[out_idx] = cadd(new_sv[out_idx], cmul(g, sv[idx]));
            }
        }

        self.decompose_statevector_into_tree(&new_sv);
        self.normalized = false;
    }

    // -----------------------------------------------------------------------
    // Full tree contraction -> statevector
    // -----------------------------------------------------------------------

    /// Contract the entire tree into a 2^n complex statevector.
    ///
    /// Each entry `sv[idx]` is the amplitude of basis state `idx`, where
    /// bit `q` of `idx` corresponds to qubit `q`.
    fn contract_to_statevector(&self) -> Vec<[f64; 2]> {
        let n = self.config.n_qubits;
        let dim = 1usize << n;

        // Recursive contraction.
        // For a subtree rooted at node_id, returns a tensor of shape
        // [2^(num_qubits_in_subtree), bond_up] stored row-major.
        // The qubit ordering in the physical dimension follows
        // subtree_qubits (left-first DFS).
        let result = self.contract_subtree(self.root);
        // result: [phys_dim, bond_up] where bond_up of root = 1
        let root_bond = self.nodes[self.root].shape.last().copied().unwrap_or(1);
        let sub_qubits = self.subtree_qubits(self.root);
        let phys_dim = result.len() / root_bond;

        // Permute from tree qubit order to standard bit order
        let mut sv = vec![ZERO; dim];
        for t_idx in 0..phys_dim {
            // Sum over root bond (typically 1)
            let mut val = ZERO;
            for b in 0..root_bond {
                val = cadd(val, result[t_idx * root_bond + b]);
            }
            if cnorm_sq(val) < 1e-30 {
                continue;
            }
            // Map tree-order index to standard-order index
            let mut sv_idx = 0usize;
            for (pos, &q) in sub_qubits.iter().enumerate() {
                let bit = (t_idx >> (sub_qubits.len() - 1 - pos)) & 1;
                sv_idx |= bit << q;
            }
            sv[sv_idx] = cadd(sv[sv_idx], val);
        }
        sv
    }

    /// Contract a subtree rooted at `node_id`.
    /// Returns tensor of shape [phys_dim_subtree, bond_up] row-major.
    fn contract_subtree(&self, node_id: usize) -> Vec<[f64; 2]> {
        let node = &self.nodes[node_id];

        if node.children.is_empty() {
            // Leaf: tensor is [2, bond_up], return as-is
            return node.tensor.clone();
        }

        let left_id = node.children[0];
        let right_id = node.children[1];

        let left_data = self.contract_subtree(left_id);
        let right_data = self.contract_subtree(right_id);

        let left_nq = self.subtree_qubits(left_id).len();
        let right_nq = self.subtree_qubits(right_id).len();
        let nl = 1usize << left_nq;
        let nr = 1usize << right_nq;

        let bl = self.nodes[left_id].shape.last().copied().unwrap_or(1);
        let br = self.nodes[right_id].shape.last().copied().unwrap_or(1);
        let bu = node.shape[2]; // bond_up of this internal node

        // Internal tensor shape: [bl, br, bu]
        // left_data shape: [nl, bl]
        // right_data shape: [nr, br]
        //
        // result[pl * nr + pr, bu] = sum_{bl, br}
        //     left[pl, bl] * right[pr, br] * internal[bl, br, bu]

        let out_phys = nl * nr;
        let mut result = vec![ZERO; out_phys * bu];

        for pl in 0..nl {
            for pr in 0..nr {
                for b in 0..bu {
                    let mut s = ZERO;
                    for lb in 0..bl {
                        let lval = left_data[pl * bl + lb];
                        if cnorm_sq(lval) < 1e-30 {
                            continue;
                        }
                        for rb in 0..br {
                            let rval = right_data[pr * br + rb];
                            let tval = node.tensor[lb * br * bu + rb * bu + b];
                            s = cadd(s, cmul(cmul(lval, rval), tval));
                        }
                    }
                    result[(pl * nr + pr) * bu + b] = s;
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Statevector -> tree decomposition via recursive SVD
    // -----------------------------------------------------------------------

    /// Decompose a full statevector back into the tree structure.
    ///
    /// For each subtree, reshape the relevant amplitudes into
    /// (left_phys_space, right_phys_space * bond_up), SVD, and assign
    /// tensors. Works recursively top-down from the root.
    fn decompose_statevector_into_tree(&mut self, sv: &[[f64; 2]]) {
        let n = self.config.n_qubits;
        let max_bd = self.config.max_bond_dim;
        let threshold = self.config.truncation_threshold;

        // Build the subtree qubit lists (needed for index mapping)
        let sub_qubits_all: Vec<Vec<usize>> = (0..self.nodes.len())
            .map(|nid| self.subtree_qubits(nid))
            .collect();

        // Convert sv (indexed by standard qubit order) into a tensor indexed
        // by the root's subtree qubit order. Since root subtree = all qubits,
        // we just need to permute.
        let root_qubits = &sub_qubits_all[self.root];
        let dim = 1usize << n;
        let mut psi = vec![ZERO; dim]; // indexed in tree order

        for sv_idx in 0..dim {
            if cnorm_sq(sv[sv_idx]) < 1e-30 {
                continue;
            }
            let mut t_idx = 0usize;
            for (pos, &q) in root_qubits.iter().enumerate() {
                let bit = (sv_idx >> q) & 1;
                t_idx |= bit << (root_qubits.len() - 1 - pos);
            }
            psi[t_idx] = sv[sv_idx];
        }

        // Recursive decomposition. `tensor` has shape [phys_dim_subtree, bond_up].
        // We set the node's tensor and return the bond_up actually used.
        fn decompose(
            nodes: &mut Vec<TTNNode>,
            sub_qubits: &[Vec<usize>],
            node_id: usize,
            tensor: &[[f64; 2]], // [phys_dim, bond_up]
            phys_dim: usize,
            bond_up: usize,
            max_bd: usize,
            threshold: f64,
        ) {
            let children = nodes[node_id].children.clone();

            if children.is_empty() {
                // Leaf: tensor is [2, bond_up], just store it
                nodes[node_id].tensor = tensor.to_vec();
                nodes[node_id].shape = vec![2, bond_up];
                nodes[node_id].bond_dim = bond_up;
                return;
            }

            let left_id = children[0];
            let right_id = children[1];
            let nlq = sub_qubits[left_id].len();
            let nrq = sub_qubits[right_id].len();
            let nl = 1usize << nlq;
            let nr = 1usize << nrq;

            // tensor has shape [nl * nr, bond_up].
            // Reshape as [nl, nr * bond_up] and SVD to get the left-right split.
            let ncols = nr * bond_up;
            let mut mat = vec![ZERO; nl * ncols];
            for pl in 0..nl {
                for pr in 0..nr {
                    for b in 0..bond_up {
                        mat[pl * ncols + pr * bond_up + b] =
                            tensor[(pl * nr + pr) * bond_up + b];
                    }
                }
            }

            let (u, sigma, vt) = csvd(&mat, nl, ncols);
            let k = nl.min(ncols);

            // Determine how many singular values to keep
            let mut keep = 0;
            for i in 0..k {
                if sigma[i] > threshold && keep < max_bd {
                    keep += 1;
                }
            }
            keep = keep.max(1);

            // Left child tensor: shape [nl, keep] = U[:, :keep] * diag(sigma[:keep])
            let mut left_tensor = vec![ZERO; nl * keep];
            for pl in 0..nl {
                for r in 0..keep {
                    left_tensor[pl * keep + r] = cscale(sigma[r], u[pl * k + r]);
                }
            }

            // Right subtree + parent bond tensor: shape [keep, nr * bond_up]
            // = Vt[:keep, :] -- but we need to re-interpret this as
            // [nr, keep_right * bond_up_right] for the right child, where the
            // internal node absorbs the structure.
            //
            // Actually, let's think about this differently.
            // The SVD gives us: mat[pl, (pr,b)] = sum_r U[pl,r] sigma[r] Vt[r, (pr,b)]
            // So the tree decomposition is:
            //   left_child[pl, bl=r] = U[pl,r] * sigma[r]    (absorbed into left)
            //   internal[bl=r, br, bu=b] ...
            //   right_child[pr, br] ...
            //
            // The right side is Vt[r, pr * bond_up + b], shape [keep, nr, bond_up].
            // We need to further split this into right_child[pr, br] and
            // internal[bl, br, bu].
            //
            // SVD the right side: reshape Vt as [keep * bond_up, nr], transpose to
            // [nr, keep * bond_up], SVD to get right_child and connection.
            //
            // Simpler: the internal node tensor absorbs Vt directly.
            // internal[bl, br, bu] where bl=keep, and the right child
            // gets its own tensor from a further SVD of Vt.

            // Reshape Vt[r, pr*bond_up+b] as a matrix of shape [nr, keep*bond_up]
            // where rows = pr, cols = (r, b).
            let rt_ncols = keep * bond_up;
            let mut rt_mat = vec![ZERO; nr * rt_ncols];
            for pr in 0..nr {
                for r in 0..keep {
                    for b in 0..bond_up {
                        rt_mat[pr * rt_ncols + r * bond_up + b] =
                            vt[r * (nr * bond_up) + pr * bond_up + b];
                    }
                }
            }

            let (ru, rsigma, rvt) = csvd(&rt_mat, nr, rt_ncols);
            let rk = nr.min(rt_ncols);
            let mut rkeep = 0;
            for i in 0..rk {
                if rsigma[i] > threshold && rkeep < max_bd {
                    rkeep += 1;
                }
            }
            rkeep = rkeep.max(1);

            // Right child tensor: [nr, rkeep] = ru * rsigma
            let mut right_tensor = vec![ZERO; nr * rkeep];
            for pr in 0..nr {
                for r in 0..rkeep {
                    right_tensor[pr * rkeep + r] = cscale(rsigma[r], ru[pr * rk + r]);
                }
            }

            // Internal node tensor: rvt[:rkeep, :] reshaped.
            // rvt has shape [rkeep, keep * bond_up] (using rkeep rows).
            // This should become internal[bl=keep, br=rkeep, bu=bond_up].
            // rvt[br, bl * bond_up + bu] -> internal[bl, br, bu]
            let mut internal_tensor = vec![ZERO; keep * rkeep * bond_up];
            for br in 0..rkeep {
                for bl in 0..keep {
                    for bu in 0..bond_up {
                        internal_tensor[bl * rkeep * bond_up + br * bond_up + bu] =
                            rvt[br * rt_ncols + bl * bond_up + bu];
                    }
                }
            }

            nodes[node_id].tensor = internal_tensor;
            nodes[node_id].shape = vec![keep, rkeep, bond_up];
            nodes[node_id].bond_dim = bond_up;

            // Recurse into children
            decompose(
                nodes, sub_qubits, left_id, &left_tensor, nl, keep, max_bd, threshold,
            );
            decompose(
                nodes, sub_qubits, right_id, &right_tensor, nr, rkeep, max_bd, threshold,
            );
        }

        // Root: tensor is [dim, 1] (bond_up = 1 for root)
        let root_tensor: Vec<[f64; 2]> = psi.iter().map(|&c| c).collect();
        let root_id = self.root;
        decompose(
            &mut self.nodes,
            &sub_qubits_all,
            root_id,
            &root_tensor,
            dim,
            1,
            max_bd,
            threshold,
        );
    }

    // -----------------------------------------------------------------------
    // Measurement and observation
    // -----------------------------------------------------------------------

    /// Probability of qubit being |1>.
    pub fn measure_probability(&self, qubit: usize) -> f64 {
        let sv = self.contract_to_statevector();
        let n = self.config.n_qubits;
        let dim = 1usize << n;
        let mut p1 = 0.0;
        let mut total = 0.0;
        for idx in 0..dim {
            let p = cnorm_sq(sv[idx]);
            total += p;
            if (idx >> qubit) & 1 == 1 {
                p1 += p;
            }
        }
        if total > 1e-30 { p1 / total } else { 0.5 }
    }

    /// Full statevector as real parts (backward-compatible).
    pub fn to_statevector(&self) -> Vec<f64> {
        self.contract_to_statevector().iter().map(|c| c[0]).collect()
    }

    /// Full complex statevector.
    pub fn to_statevector_complex(&self) -> Vec<[f64; 2]> {
        self.contract_to_statevector()
    }

    /// Normalize the state by scaling the root tensor.
    pub fn normalize(&mut self) {
        if self.normalized {
            return;
        }
        let sv = self.contract_to_statevector();
        let norm = sv.iter().map(|c| cnorm_sq(*c)).sum::<f64>().sqrt();
        if norm > 1e-30 && (norm - 1.0).abs() > 1e-14 {
            let inv = 1.0 / norm;
            for c in &mut self.nodes[self.root].tensor {
                *c = cscale(inv, *c);
            }
        }
        self.normalized = true;
    }

    /// Entanglement entropy across a bipartition.
    ///
    /// `qubits_a` defines one side of the cut. Returns the von Neumann
    /// entropy S = -sum_i s_i^2 ln(s_i^2) of the Schmidt decomposition.
    pub fn entanglement_entropy(&self, qubits_a: &[usize]) -> f64 {
        let n = self.config.n_qubits;
        let sv = self.contract_to_statevector();

        let na = qubits_a.len();
        let nb = n - na;
        let da = 1usize << na;
        let db = 1usize << nb;

        let qubits_b: Vec<usize> = (0..n).filter(|q| !qubits_a.contains(q)).collect();

        // Reshape sv as (da, db) matrix
        let dim = 1usize << n;
        let mut mat = vec![ZERO; da * db];
        for idx in 0..dim {
            let mut ai = 0usize;
            for (pos, &q) in qubits_a.iter().enumerate() {
                ai |= ((idx >> q) & 1) << (na - 1 - pos);
            }
            let mut bi = 0usize;
            for (pos, &q) in qubits_b.iter().enumerate() {
                bi |= ((idx >> q) & 1) << (nb - 1 - pos);
            }
            mat[ai * db + bi] = sv[idx];
        }

        let (_u, sigma, _vt) = csvd(&mat, da, db);

        let mut entropy = 0.0f64;
        for &s in &sigma {
            let p = s * s;
            if p > 1e-30 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Bond dimension at a node.
    pub fn bond_dimension(&self, node_id: usize) -> usize {
        self.nodes[node_id].bond_dim
    }

    /// Total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        self.nodes
            .iter()
            .map(|n| n.tensor.len() * std::mem::size_of::<[f64; 2]>())
            .sum()
    }

    /// Truncate bond dimensions by re-decomposing from the statevector
    /// with the current max_bond_dim setting.
    pub fn truncate(&mut self) {
        let sv = self.contract_to_statevector();
        self.decompose_statevector_into_tree(&sv);
    }
}

// ===========================================================================
// TTN CIRCUIT SIMULATOR
// ===========================================================================

/// TTN-based circuit simulator with gate counting and sampling.
pub struct TTNSimulator {
    ttn: TTN,
    gate_count: usize,
}

impl TTNSimulator {
    /// Create a new TTN simulator.
    pub fn new(config: TTNConfig) -> Self {
        Self {
            ttn: TTN::new(config),
            gate_count: 0,
        }
    }

    /// Apply a named gate.
    pub fn apply(&mut self, gate_type: &str, targets: &[usize]) {
        match gate_type {
            "H" => self.ttn.apply_h(targets[0]),
            "X" => self.ttn.apply_x(targets[0]),
            "Y" => self.ttn.apply_y(targets[0]),
            "Z" => self.ttn.apply_z(targets[0]),
            "S" => self.ttn.apply_s(targets[0]),
            "T" => self.ttn.apply_t(targets[0]),
            "CNOT" | "CX" => self.ttn.apply_cnot(targets[0], targets[1]),
            _ => {}
        }
        self.gate_count += 1;
    }

    /// Single-qubit measurement probabilities (prob of |1>).
    pub fn probabilities(&self) -> Vec<f64> {
        (0..self.ttn.n_qubits())
            .map(|q| self.ttn.measure_probability(q))
            .collect()
    }

    /// Sample measurement outcomes using real randomness.
    pub fn sample(&self, shots: usize) -> Vec<Vec<bool>> {
        let sv = self.ttn.contract_to_statevector();
        let n = self.ttn.n_qubits();
        let probs: Vec<f64> = sv.iter().map(|c| cnorm_sq(*c)).collect();
        let total: f64 = probs.iter().sum();

        let mut rng = rand::thread_rng();
        let mut results = Vec::with_capacity(shots);

        for _ in 0..shots {
            let r: f64 = rng.gen::<f64>() * total;
            let mut cum = 0.0;
            let mut chosen = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum > r {
                    chosen = i;
                    break;
                }
            }
            let outcome: Vec<bool> = (0..n).map(|q| (chosen >> q) & 1 == 1).collect();
            results.push(outcome);
        }
        results
    }

    /// Gate count.
    pub fn gate_count(&self) -> usize {
        self.gate_count
    }

    /// Memory usage.
    pub fn memory(&self) -> usize {
        self.ttn.memory_usage()
    }
}

// ===========================================================================
// TESTS
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-6;

    #[test]
    fn test_ttn_config() {
        let config = TTNConfig::new(8, 32);
        assert_eq!(config.n_qubits, 8);
        assert_eq!(config.max_bond_dim, 32);
    }

    #[test]
    fn test_ttn_creation() {
        let config = TTNConfig::new(4, 16);
        let ttn = TTN::new(config);
        assert_eq!(ttn.n_qubits(), 4);
        assert!(ttn.normalized);
    }

    #[test]
    fn test_ttn_initial_statevector() {
        let config = TTNConfig::new(2, 16);
        let ttn = TTN::new(config);
        let sv = ttn.to_statevector();
        assert_eq!(sv.len(), 4);
        assert!((sv[0] - 1.0).abs() < TOL);
        assert!(sv[1].abs() < TOL);
        assert!(sv[2].abs() < TOL);
        assert!(sv[3].abs() < TOL);
    }

    #[test]
    fn test_ttn_hadamard() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        let prob = ttn.measure_probability(0);
        assert!((prob - 0.5).abs() < 0.01, "H|0> prob should be 0.5, got {}", prob);
    }

    #[test]
    fn test_ttn_x_gate() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        assert!(ttn.measure_probability(0) < 0.01);
        ttn.apply_x(0);
        assert!((ttn.measure_probability(0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ttn_ry_rotation() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_ry(0, PI / 2.0);
        let prob = ttn.measure_probability(0);
        assert!((prob - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ttn_statevector() {
        let config = TTNConfig::new(2, 16);
        let ttn = TTN::new(config);
        let sv = ttn.to_statevector();
        assert_eq!(sv.len(), 4);
        assert!((sv[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ttn_normalize() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        assert!(ttn.normalized);
        ttn.apply_h(0);
        assert!(!ttn.normalized);
        ttn.normalize();
        assert!(ttn.normalized);
        let prob = ttn.measure_probability(0);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_ttn_simulator() {
        let config = TTNConfig::new(4, 16);
        let mut sim = TTNSimulator::new(config);
        sim.apply("H", &[0]);
        sim.apply("X", &[1]);
        assert_eq!(sim.gate_count(), 2);
        let probs = sim.probabilities();
        assert_eq!(probs.len(), 4);
    }

    #[test]
    fn test_ttn_sample() {
        let config = TTNConfig::new(2, 16);
        let mut sim = TTNSimulator::new(config);
        sim.apply("H", &[0]);
        sim.apply("H", &[1]);
        let samples = sim.sample(100);
        assert_eq!(samples.len(), 100);
        for sample in &samples {
            assert_eq!(sample.len(), 2);
        }
    }

    #[test]
    fn test_ttn_memory() {
        let config = TTNConfig::new(4, 16);
        let sim = TTNSimulator::new(config);
        assert!(sim.memory() > 0);
    }

    // -------------------------------------------------------------------
    // Bell state and entanglement tests
    // -------------------------------------------------------------------

    #[test]
    fn test_cnot_bell_state() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_cnot(0, 1);

        let sv = ttn.to_statevector_complex();
        assert_eq!(sv.len(), 4);

        let s = 1.0 / 2.0f64.sqrt();
        // Bell state: (|00> + |11>) / sqrt(2)
        assert!((cnorm_sq(sv[0]).sqrt() - s).abs() < 0.01,
            "|00> amp should be 1/sqrt(2), got {}", cnorm_sq(sv[0]).sqrt());
        assert!(cnorm_sq(sv[1]) < 0.01, "|01> should be ~0");
        assert!(cnorm_sq(sv[2]) < 0.01, "|10> should be ~0");
        assert!((cnorm_sq(sv[3]).sqrt() - s).abs() < 0.01,
            "|11> amp should be 1/sqrt(2), got {}", cnorm_sq(sv[3]).sqrt());
    }

    #[test]
    fn test_entanglement_entropy_bell() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_cnot(0, 1);

        let entropy = ttn.entanglement_entropy(&[0]);
        let expected = 2.0f64.ln();
        assert!((entropy - expected).abs() < 0.05,
            "Bell entropy should be ln(2)={:.4}, got {:.4}", expected, entropy);
    }

    #[test]
    fn test_entanglement_entropy_product() {
        let config = TTNConfig::new(2, 16);
        let ttn = TTN::new(config);
        let entropy = ttn.entanglement_entropy(&[0]);
        assert!(entropy.abs() < TOL, "Product state entropy should be 0, got {:.6}", entropy);
    }

    #[test]
    fn test_statevector_ghz_3qubit() {
        // GHZ on qubits 0,1,2 of a 4-qubit system: (|0000>+|0111>)/sqrt(2)
        let config = TTNConfig::new(4, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_cnot(0, 1);
        ttn.apply_cnot(0, 2);

        let sv = ttn.to_statevector_complex();
        let s = 1.0 / 2.0f64.sqrt();

        // |0000> = index 0
        assert!((cnorm_sq(sv[0]).sqrt() - s).abs() < 0.02,
            "|0000> should be 1/sqrt(2), got {}", cnorm_sq(sv[0]).sqrt());
        // |0111> = qubits 0,1,2 set = bits 0,1,2 = 0b0111 = 7
        assert!((cnorm_sq(sv[7]).sqrt() - s).abs() < 0.02,
            "|0111> should be 1/sqrt(2), got {}", cnorm_sq(sv[7]).sqrt());
        for i in 0..sv.len() {
            if i != 0 && i != 7 {
                assert!(cnorm_sq(sv[i]) < 0.01, "sv[{}] should be ~0, got {}", i, cnorm_sq(sv[i]).sqrt());
            }
        }
    }

    #[test]
    fn test_cnot_preserves_norm() {
        let config = TTNConfig::new(4, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_h(2);
        ttn.apply_cnot(0, 1);
        ttn.apply_cnot(2, 3);

        let sv = ttn.to_statevector_complex();
        let norm: f64 = sv.iter().map(|c| cnorm_sq(*c)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.02, "Norm should be 1.0, got {}", norm);
    }

    #[test]
    fn test_multiple_cnots() {
        // X(0), CNOT chain -> |1111>
        let config = TTNConfig::new(4, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_x(0);
        ttn.apply_cnot(0, 1);
        ttn.apply_cnot(1, 2);
        ttn.apply_cnot(2, 3);

        let sv = ttn.to_statevector_complex();
        // |1111> = index 15
        assert!((cnorm_sq(sv[15]).sqrt() - 1.0).abs() < 0.02,
            "|1111> should be 1.0, got {}", cnorm_sq(sv[15]).sqrt());
        for i in 0..sv.len() {
            if i != 15 {
                assert!(cnorm_sq(sv[i]) < 0.01, "sv[{}] should be ~0", i);
            }
        }
    }

    #[test]
    fn test_rx_rz_complex_phases() {
        // Rx(pi)|0> = -i|1>
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_rx(0, PI);

        let sv = ttn.to_statevector_complex();
        // qubit 0 flipped: state = -i|01> = index 1
        assert!(cnorm_sq(sv[0]) < TOL, "|00> should be 0 after Rx(pi)");
        assert!((cnorm_sq(sv[1]) - 1.0).abs() < TOL, "|01| should be 1");
        assert!(sv[1][0].abs() < TOL, "Real part should be ~0");
        assert!((sv[1][1] + 1.0).abs() < TOL, "Imag part should be -1");
    }

    #[test]
    fn test_rz_phase() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_rz(0, PI / 2.0);

        let sv = ttn.to_statevector_complex();
        assert!((cnorm_sq(sv[0]) - 0.5).abs() < TOL);
        assert!((cnorm_sq(sv[1]) - 0.5).abs() < TOL);
        let phase00 = sv[0][1].atan2(sv[0][0]);
        let phase01 = sv[1][1].atan2(sv[1][0]);
        let diff = phase01 - phase00;
        assert!((diff - PI / 2.0).abs() < TOL, "Phase diff should be pi/2, got {}", diff);
    }

    #[test]
    fn test_y_gate() {
        // Y|0> = i|1>
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_y(0);
        let sv = ttn.to_statevector_complex();
        assert!(cnorm_sq(sv[0]) < TOL);
        assert!((cnorm_sq(sv[1]) - 1.0).abs() < TOL);
        assert!(sv[1][0].abs() < TOL);
        assert!((sv[1][1] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_s_gate_phase() {
        let config = TTNConfig::new(2, 16);
        let mut ttn = TTN::new(config);
        ttn.apply_x(0);
        ttn.apply_s(0);
        let sv = ttn.to_statevector_complex();
        assert!((sv[1][1] - 1.0).abs() < TOL, "S|1> should give i|1>");
    }

    #[test]
    fn test_sampling_statistics() {
        let config = TTNConfig::new(2, 16);
        let mut sim = TTNSimulator::new(config);
        sim.apply("H", &[0]);
        let samples = sim.sample(1000);
        let count_1: usize = samples.iter().filter(|s| s[0]).count();
        let frac = count_1 as f64 / 1000.0;
        assert!((frac - 0.5).abs() < 0.1, "H|0> should sample ~50%, got {:.1}%", frac * 100.0);
    }

    #[test]
    fn test_bell_sampling_correlations() {
        let config = TTNConfig::new(2, 16);
        let mut sim = TTNSimulator::new(config);
        sim.apply("H", &[0]);
        sim.apply("CNOT", &[0, 1]);
        let samples = sim.sample(200);
        for (i, s) in samples.iter().enumerate() {
            assert_eq!(s[0], s[1], "Shot {}: Bell outcomes must correlate, got {:?}", i, s);
        }
    }

    #[test]
    fn test_4qubit_norm_after_circuit() {
        let config = TTNConfig::new(4, 32);
        let mut ttn = TTN::new(config);
        ttn.apply_h(0);
        ttn.apply_h(1);
        ttn.apply_cnot(0, 2);
        ttn.apply_cnot(1, 3);
        ttn.apply_rz(0, 1.23);
        ttn.apply_ry(3, 0.77);

        let sv = ttn.to_statevector_complex();
        let norm: f64 = sv.iter().map(|c| cnorm_sq(*c)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.02, "Norm should be 1.0, got {}", norm);
    }
}
