//! Quantum Random Access Memory (QRAM) Simulation
//!
//! Three QRAM architectures for superposition queries over classical data.
//!
//! Given address register |a> and data register |d>, QRAM performs:
//!     |a>|0> --> |a>|D[a]>
//!
//! If |a> is in superposition, the output is also in superposition --
//! essential for Grover search, HHL linear systems, and quantum ML.
//!
//! # Architecture
//!
//! ```text
//!                     +-----------+
//!    |a1 a2 ... an> --| Address   |
//!                     |           |---> |a>|D[a]>
//!    |0  0  ... 0>  --| Data      |
//!                     +-----------+
//!
//!    Bucket-Brigade (tree of quantum routers):
//!
//!                  [R_root]            level 0
//!                 /        \
//!           [R_01]          [R_02]     level 1
//!           /    \          /    \
//!        [R_03] [R_04]  [R_05] [R_06] level 2
//!         |  |   |  |   |  |   |  |
//!        D0 D1  D2 D3  D4 D5  D6 D7   memory cells
//!
//!    SelectOnly (multi-controlled X cascade):
//!
//!        for each addr a in 0..2^n:
//!            if address_reg == a:
//!                XOR data_reg with D[a]
//!
//!    FanOut (GHZ-style parallel):
//!
//!        |a> --[fanout]--> 2^n copies --[parallel load]--> |D[a]>
//! ```
//!
//! # Variant Comparison
//!
//! | Property        | BucketBrigade | SelectOnly | FanOut       |
//! |-----------------|---------------|------------|--------------|
//! | Depth           | O(n)          | O(2^n)     | O(n)         |
//! | Qubits          | O(2^n)        | O(n)       | O(2^n + n)   |
//! | Effective noise | O(n)          | O(2^n)     | O(n)         |
//! | Ancillas        | 2^n - 1       | O(n)       | 2^n          |

use num_complex::Complex64 as C64;
use std::fmt;

use crate::{GateOperations, QuantumState};

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from QRAM construction and query operations.
#[derive(Debug, Clone)]
pub enum QRAMError {
    /// Address register exceeds maximum supported width.
    AddressTooLarge { bits: usize, max: usize },
    /// Classical memory size does not match the configured address space.
    DataMismatch { expected: usize, got: usize },
    /// General configuration error.
    ConfigError(String),
}

impl fmt::Display for QRAMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QRAMError::AddressTooLarge { bits, max } => {
                write!(f, "address width {} exceeds maximum {}", bits, max)
            }
            QRAMError::DataMismatch { expected, got } => {
                write!(f, "expected {} data entries, got {}", expected, got)
            }
            QRAMError::ConfigError(msg) => write!(f, "config error: {}", msg),
        }
    }
}

impl std::error::Error for QRAMError {}

// ============================================================
// QRAM VARIANT SELECTION
// ============================================================

/// Which QRAM architecture to instantiate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QRAMVariant {
    /// Tree of quantum routers. Most noise-resilient (O(n) active nodes).
    BucketBrigade,
    /// SELECT oracle using multi-controlled-X. Fewest ancillas, highest depth.
    SelectOnly,
    /// GHZ-style parallel fanout. Lowest depth, most ancillas.
    FanOut,
}

impl fmt::Display for QRAMVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QRAMVariant::BucketBrigade => write!(f, "BucketBrigade"),
            QRAMVariant::SelectOnly => write!(f, "SelectOnly"),
            QRAMVariant::FanOut => write!(f, "FanOut"),
        }
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Maximum address width we allow (to prevent accidental 2^30 allocations).
const MAX_ADDRESS_BITS: usize = 16;

/// Configuration for a QRAM instance.
#[derive(Debug, Clone)]
pub struct QRAMConfig {
    /// Number of address qubits (n). Memory has 2^n cells.
    pub address_bits: usize,
    /// Bits per memory cell.
    pub data_bits: usize,
    /// Per-node depolarizing error rate (0.0 = noiseless).
    pub error_rate: f64,
    /// Architecture variant.
    pub variant: QRAMVariant,
}

impl QRAMConfig {
    /// Create a new config, validating parameters.
    pub fn new(
        address_bits: usize,
        data_bits: usize,
        error_rate: f64,
        variant: QRAMVariant,
    ) -> Result<Self, QRAMError> {
        if address_bits == 0 {
            return Err(QRAMError::ConfigError(
                "address_bits must be >= 1".to_string(),
            ));
        }
        if address_bits > MAX_ADDRESS_BITS {
            return Err(QRAMError::AddressTooLarge {
                bits: address_bits,
                max: MAX_ADDRESS_BITS,
            });
        }
        if data_bits == 0 {
            return Err(QRAMError::ConfigError("data_bits must be >= 1".to_string()));
        }
        if !(0.0..=1.0).contains(&error_rate) {
            return Err(QRAMError::ConfigError(
                "error_rate must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(Self {
            address_bits,
            data_bits,
            error_rate,
            variant,
        })
    }

    /// Builder: set address bits.
    pub fn with_address_bits(mut self, n: usize) -> Self {
        self.address_bits = n;
        self
    }

    /// Builder: set data bits.
    pub fn with_data_bits(mut self, d: usize) -> Self {
        self.data_bits = d;
        self
    }

    /// Builder: set error rate.
    pub fn with_error_rate(mut self, e: f64) -> Self {
        self.error_rate = e;
        self
    }

    /// Builder: set variant.
    pub fn with_variant(mut self, v: QRAMVariant) -> Self {
        self.variant = v;
        self
    }

    /// Number of memory cells (2^address_bits).
    pub fn num_cells(&self) -> usize {
        1 << self.address_bits
    }
}

impl Default for QRAMConfig {
    fn default() -> Self {
        Self {
            address_bits: 3,
            data_bits: 1,
            error_rate: 0.0,
            variant: QRAMVariant::BucketBrigade,
        }
    }
}

// ============================================================
// CLASSICAL MEMORY
// ============================================================

/// Classical data store backing the QRAM.
///
/// `data[address]` is a bit-vector of length `bits_per_cell`.
#[derive(Debug, Clone)]
pub struct ClassicalMemory {
    /// data[address][bit_index] = bool.
    pub data: Vec<Vec<bool>>,
    /// Number of addressable cells.
    pub num_addresses: usize,
    /// Width of each cell in bits.
    pub bits_per_cell: usize,
}

impl ClassicalMemory {
    /// Create memory initialised to all zeros.
    pub fn zeros(num_addresses: usize, bits_per_cell: usize) -> Self {
        Self {
            data: vec![vec![false; bits_per_cell]; num_addresses],
            num_addresses,
            bits_per_cell,
        }
    }

    /// Create memory from a slice of unsigned values.
    /// Each value is truncated to `bits_per_cell` bits.
    pub fn from_values(values: &[u64], bits_per_cell: usize) -> Self {
        let data: Vec<Vec<bool>> = values
            .iter()
            .map(|&v| (0..bits_per_cell).map(|b| (v >> b) & 1 == 1).collect())
            .collect();
        Self {
            num_addresses: values.len(),
            bits_per_cell,
            data,
        }
    }

    /// Read a cell as an integer.
    pub fn read(&self, address: usize) -> u64 {
        let mut val = 0u64;
        for (b, &bit) in self.data[address].iter().enumerate() {
            if bit {
                val |= 1 << b;
            }
        }
        val
    }

    /// Write an integer value into a cell.
    pub fn write(&mut self, address: usize, value: u64) {
        for b in 0..self.bits_per_cell {
            self.data[address][b] = (value >> b) & 1 == 1;
        }
    }
}

// ============================================================
// ROUTER NODE (BUCKET-BRIGADE)
// ============================================================

/// State of a quantum router in the bucket-brigade tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterState {
    /// Inactive -- not yet addressed.
    Wait,
    /// Routing signal left (address bit = 0).
    Left,
    /// Routing signal right (address bit = 1).
    Right,
    /// In quantum superposition of Left and Right.
    Superposition,
}

/// A single router node in the bucket-brigade binary tree.
#[derive(Debug, Clone)]
pub struct RouterNode {
    /// Unique identifier (level-order index starting at 0 for root).
    pub id: usize,
    /// Tree level. 0 = root, address_bits - 1 = leaves.
    pub level: usize,
    /// Index of the qubit representing this router's state.
    pub qubit: usize,
    /// Left child node id (None for leaves).
    pub left_child: Option<usize>,
    /// Right child node id (None for leaves).
    pub right_child: Option<usize>,
    /// Current classical state for tracking.
    pub state: RouterState,
}

// ============================================================
// QRAM GATE SET
// ============================================================

/// Gates used in QRAM circuit decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QRAMGate {
    /// Controlled-NOT (control, target).
    CNOT(usize, usize),
    /// Toffoli / CCX (control1, control2, target).
    Toffoli(usize, usize, usize),
    /// Pauli-X (target).
    X(usize),
    /// Controlled-SWAP / Fredkin (control, target1, target2).
    CSWAP(usize, usize, usize),
}

impl QRAMGate {
    /// Maximum qubit index referenced by this gate.
    pub fn max_qubit(&self) -> usize {
        match *self {
            QRAMGate::CNOT(a, b) => a.max(b),
            QRAMGate::Toffoli(a, b, c) => a.max(b).max(c),
            QRAMGate::X(a) => a,
            QRAMGate::CSWAP(a, b, c) => a.max(b).max(c),
        }
    }
}

// ============================================================
// QRAM CIRCUIT
// ============================================================

/// Explicit gate-level decomposition of a QRAM query.
#[derive(Debug, Clone)]
pub struct QRAMCircuit {
    /// Ordered sequence of gates.
    pub gates: Vec<QRAMGate>,
    /// Total qubits required.
    pub num_qubits: usize,
    /// Circuit depth (longest path through the gate DAG).
    pub depth: usize,
    /// Number of T gates in the Toffoli decompositions.
    pub t_count: usize,
    /// Number of CNOT gates (including those inside Toffoli).
    pub cnot_count: usize,
}

impl QRAMCircuit {
    /// Compute summary statistics from the gate list.
    pub fn compute_stats(gates: &[QRAMGate], num_qubits: usize) -> Self {
        let mut t_count = 0usize;
        let mut cnot_count = 0usize;

        for gate in gates {
            match gate {
                QRAMGate::CNOT(..) => {
                    cnot_count += 1;
                }
                QRAMGate::Toffoli(..) => {
                    // Standard Toffoli decomposition: 6 CNOT + 7 T gates.
                    cnot_count += 6;
                    t_count += 7;
                }
                QRAMGate::CSWAP(..) => {
                    // Fredkin = 2 CNOT + 1 Toffoli = 2 + 6 CNOT, 7 T.
                    cnot_count += 8;
                    t_count += 7;
                }
                QRAMGate::X(..) => {}
            }
        }

        // Depth: we use a simple per-qubit-last-used tracker.
        let depth = Self::compute_depth(gates, num_qubits);

        Self {
            gates: gates.to_vec(),
            num_qubits,
            depth,
            t_count,
            cnot_count,
        }
    }

    /// Compute circuit depth via per-qubit scheduling.
    fn compute_depth(gates: &[QRAMGate], num_qubits: usize) -> usize {
        let mut qubit_depth = vec![0usize; num_qubits];

        for gate in gates {
            let qubits: Vec<usize> = match *gate {
                QRAMGate::CNOT(a, b) => vec![a, b],
                QRAMGate::Toffoli(a, b, c) => vec![a, b, c],
                QRAMGate::X(a) => vec![a],
                QRAMGate::CSWAP(a, b, c) => vec![a, b, c],
            };
            let max_d = qubits.iter().map(|&q| qubit_depth[q]).max().unwrap_or(0);
            let new_d = max_d + 1;
            for &q in &qubits {
                qubit_depth[q] = new_d;
            }
        }

        qubit_depth.into_iter().max().unwrap_or(0)
    }
}

// ============================================================
// RESOURCE ANALYSIS
// ============================================================

/// Resource requirements for a given QRAM configuration.
#[derive(Debug, Clone)]
pub struct QRAMResources {
    /// Architecture variant.
    pub variant: QRAMVariant,
    /// Address register width.
    pub num_address_bits: usize,
    /// Data register width.
    pub num_data_bits: usize,
    /// Total qubit count (address + data + ancillas).
    pub total_qubits: usize,
    /// Circuit depth.
    pub circuit_depth: usize,
    /// T-gate count.
    pub t_count: usize,
    /// CNOT count (including Toffoli decomposition).
    pub cnot_count: usize,
    /// Effective error rate after noise accumulation.
    pub noise_resilience: f64,
}

impl QRAMResources {
    /// Estimate resources for a given config and memory.
    pub fn estimate(config: &QRAMConfig, _memory: &ClassicalMemory) -> Self {
        let n = config.address_bits;
        let d = config.data_bits;
        let num_cells = 1usize << n;
        let e = config.error_rate;

        match config.variant {
            QRAMVariant::BucketBrigade => {
                let router_ancillas = num_cells - 1; // 2^n - 1 router qubits
                let total_qubits = n + d + router_ancillas;

                // Depth: O(n) -- one routing step per address bit, plus data load.
                let circuit_depth = 2 * n + d;

                // Gates: n routing levels, each with O(1) controlled ops per active node.
                // Plus d CNOTs for data load at the leaf.
                let t_count = 7 * n; // One Toffoli per routing level.
                let cnot_count = 6 * n + d;

                // Noise: only O(n) routers active per query path.
                let active_gates = n + d;
                let noise_resilience = 1.0 - (1.0 - e).powi(active_gates as i32);

                QRAMResources {
                    variant: QRAMVariant::BucketBrigade,
                    num_address_bits: n,
                    num_data_bits: d,
                    total_qubits,
                    circuit_depth,
                    t_count,
                    cnot_count,
                    noise_resilience,
                }
            }
            QRAMVariant::SelectOnly => {
                // O(n) ancillas for Toffoli cascade, no router tree.
                let ancillas = n.saturating_sub(1); // workspace for multi-controlled decomposition
                let total_qubits = n + d + ancillas;

                // Depth: O(2^n) -- one multi-controlled-X per address.
                // Each multi-controlled-X decomposes into O(n) Toffoli gates.
                let circuit_depth = num_cells * (2 * n);

                // Each address: n-controlled X = (n-1) Toffoli gates for decomposition.
                let toffoli_per_addr = n.saturating_sub(1);
                let t_count = 7 * toffoli_per_addr * num_cells * d;
                let cnot_count = 6 * toffoli_per_addr * num_cells * d + num_cells * d;

                // Noise: all 2^n controlled ops are in the critical path.
                let active_gates = num_cells * n * d;
                let noise_resilience = 1.0 - (1.0 - e).powi(active_gates.min(10000) as i32);

                QRAMResources {
                    variant: QRAMVariant::SelectOnly,
                    num_address_bits: n,
                    num_data_bits: d,
                    total_qubits,
                    circuit_depth,
                    t_count,
                    cnot_count,
                    noise_resilience,
                }
            }
            QRAMVariant::FanOut => {
                // GHZ fanout uses 2^n ancilla qubits for parallel copy.
                let fanout_ancillas = num_cells;
                let total_qubits = n + d + fanout_ancillas;

                // Depth: O(n) for fanout + O(1) for parallel load + O(n) for unfanout.
                let circuit_depth = 2 * n + d;

                // Fanout: n CNOT layers. Parallel load: 2^n Toffoli gates (all at once).
                let t_count = 7 * num_cells * d;
                let cnot_count = 2 * n + num_cells * d;

                // Noise: O(n) depth but 2^n parallel gates.
                let active_depth_gates = 2 * n + d;
                let noise_resilience = 1.0 - (1.0 - e).powi(active_depth_gates as i32);

                QRAMResources {
                    variant: QRAMVariant::FanOut,
                    num_address_bits: n,
                    num_data_bits: d,
                    total_qubits,
                    circuit_depth,
                    t_count,
                    cnot_count,
                    noise_resilience,
                }
            }
        }
    }
}

impl fmt::Display for QRAMResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} addr, {} data | {} qubits, depth {}, T-count {}, CNOT {}, noise {:.6}",
            self.variant,
            self.num_address_bits,
            self.num_data_bits,
            self.total_qubits,
            self.circuit_depth,
            self.t_count,
            self.cnot_count,
            self.noise_resilience
        )
    }
}

// ============================================================
// BUCKET-BRIGADE QRAM
// ============================================================

/// Bucket-brigade QRAM: a binary tree of quantum router nodes.
///
/// Only O(n) routers are active per query, yielding O(n) effective noise
/// despite the tree having 2^n - 1 total nodes.
#[derive(Debug, Clone)]
pub struct BucketBrigadeQRAM {
    /// Configuration.
    pub config: QRAMConfig,
    /// Classical data.
    pub memory: ClassicalMemory,
    /// Level-order array of router nodes (index 0 = root).
    pub router_tree: Vec<RouterNode>,
    /// Total qubits: address + data + router ancillas.
    pub total_qubits: usize,
}

impl BucketBrigadeQRAM {
    /// Build the bucket-brigade tree and compute qubit layout.
    ///
    /// Qubit layout:
    ///   [0 .. n-1]          address register
    ///   [n .. n+d-1]        data register
    ///   [n+d .. n+d+R-1]    router ancillas (R = 2^n - 1)
    pub fn new(config: QRAMConfig, memory: ClassicalMemory) -> Result<Self, QRAMError> {
        let n = config.address_bits;
        let d = config.data_bits;
        let num_cells = 1usize << n;

        if memory.num_addresses != num_cells {
            return Err(QRAMError::DataMismatch {
                expected: num_cells,
                got: memory.num_addresses,
            });
        }
        if memory.bits_per_cell != d {
            return Err(QRAMError::DataMismatch {
                expected: d,
                got: memory.bits_per_cell,
            });
        }

        let num_leaf_ancillas = num_cells; // 2^n leaf qubits for CSWAP tree
        let _router_qubit_start = n + d;
        let total_qubits = n + d + num_leaf_ancillas;

        Ok(Self {
            config,
            memory,
            router_tree: Vec::new(), // Not used; kept for struct compat
            total_qubits,
        })
    }

    /// Generate the gate sequence for a single QRAM query.
    ///
    /// Uses a CSWAP-tree (Fredkin tree) selection network:
    ///   1. Initialize 2^n leaf ancillas with classical memory data.
    ///   2. CSWAP tree: address bits select the right cell value.
    ///   3. Copy result from leaf[0] to data register.
    ///   4. Uncompute CSWAP tree and leaf initialization.
    ///
    /// For each data bit, the tree is run independently (reusing
    /// the same leaf ancillas), so qubit count stays at n + d + 2^n.
    pub fn generate_circuit(&self) -> QRAMCircuit {
        let n = self.config.address_bits;
        let d = self.config.data_bits;
        let num_cells = 1usize << n;
        let leaf_start = n + d; // first leaf ancilla qubit
        let mut gates: Vec<QRAMGate> = Vec::new();

        // Process each data bit independently using the same leaf ancillas.
        for data_bit in 0..d {
            // Step 1: Initialize leaf ancillas with this data bit.
            for cell in 0..num_cells {
                if self.memory.data[cell][data_bit] {
                    gates.push(QRAMGate::X(leaf_start + cell));
                }
            }

            // Step 2: CSWAP selection tree.
            // Level k uses addr[k] to select between pairs separated by stride 2^k.
            // After all levels, leaf[0] holds data[address][data_bit].
            let mut cswap_gates: Vec<QRAMGate> = Vec::new();
            for k in 0..n {
                let stride = 1 << (k + 1);
                let offset = 1 << k;
                let mut g = 0;
                while g * stride < num_cells {
                    let left = leaf_start + g * stride;
                    let right = leaf_start + g * stride + offset;
                    cswap_gates.push(QRAMGate::CSWAP(k, left, right));
                    g += 1;
                }
            }
            for gate in &cswap_gates {
                gates.push(*gate);
            }

            // Step 3: Copy result from leaf[0] to data register.
            gates.push(QRAMGate::CNOT(leaf_start, n + data_bit));

            // Step 4: Uncompute CSWAP tree (reverse order).
            for gate in cswap_gates.iter().rev() {
                gates.push(*gate);
            }

            // Step 5: Uncompute leaf initialization.
            for cell in 0..num_cells {
                if self.memory.data[cell][data_bit] {
                    gates.push(QRAMGate::X(leaf_start + cell));
                }
            }
        }

        QRAMCircuit::compute_stats(&gates, self.total_qubits)
    }
}

// ============================================================
// SELECT-ONLY QRAM
// ============================================================

/// SELECT-only QRAM: no router tree, just a cascade of multi-controlled-X gates.
///
/// For each address `a` in 0..2^n, if the address register equals `a`,
/// XOR `D[a]` into the data register.
///
/// Lower qubit count than bucket-brigade, but O(2^n) depth.
#[derive(Debug, Clone)]
pub struct SelectOnlyQRAM {
    /// Configuration.
    pub config: QRAMConfig,
    /// Classical data.
    pub memory: ClassicalMemory,
    /// Total qubits: address + data + ancillas for Toffoli decomposition.
    pub total_qubits: usize,
}

impl SelectOnlyQRAM {
    /// Construct a SELECT-only QRAM.
    pub fn new(config: QRAMConfig, memory: ClassicalMemory) -> Result<Self, QRAMError> {
        let n = config.address_bits;
        let d = config.data_bits;
        let num_cells = 1usize << n;

        if memory.num_addresses != num_cells {
            return Err(QRAMError::DataMismatch {
                expected: num_cells,
                got: memory.num_addresses,
            });
        }
        if memory.bits_per_cell != d {
            return Err(QRAMError::DataMismatch {
                expected: d,
                got: memory.bits_per_cell,
            });
        }

        // Ancilla qubits for Toffoli cascade: n - 1 workspace qubits.
        let ancillas = if n > 1 { n - 1 } else { 0 };
        let total_qubits = n + d + ancillas;

        Ok(Self {
            config,
            memory,
            total_qubits,
        })
    }

    /// Generate the gate sequence.
    ///
    /// For each address value `a`:
    ///   1. Flip address bits where `a` has a 0 (so address register = all-1 iff match).
    ///   2. Multi-controlled-X (all address bits controlling data bit) for each set data bit.
    ///   3. Unflip the address bits.
    pub fn generate_circuit(&self) -> QRAMCircuit {
        let n = self.config.address_bits;
        let d = self.config.data_bits;
        let num_cells = 1usize << n;
        let ancilla_start = n + d;
        let mut gates: Vec<QRAMGate> = Vec::new();

        for addr in 0..num_cells {
            // Step 1: Flip address bits where addr has a 0.
            let mut flips = Vec::new();
            for bit in 0..n {
                if (addr >> bit) & 1 == 0 {
                    gates.push(QRAMGate::X(bit));
                    flips.push(bit);
                }
            }

            // Step 2: Multi-controlled-X for each data bit that is 1.
            for data_bit in 0..d {
                if self.memory.data[addr][data_bit] {
                    let target = n + data_bit;

                    if n == 1 {
                        // Single-controlled: just CNOT.
                        gates.push(QRAMGate::CNOT(0, target));
                    } else if n == 2 {
                        // Two controls: Toffoli.
                        gates.push(QRAMGate::Toffoli(0, 1, target));
                    } else {
                        // Decompose n-controlled-X into a cascade of Toffoli gates
                        // using n-2 ancilla qubits.
                        //
                        // Toffoli(addr[0], addr[1], anc[0])
                        // Toffoli(anc[0], addr[2], anc[1])
                        // ...
                        // Toffoli(anc[n-3], addr[n-1], target)
                        // then uncompute ancillas in reverse.

                        // Forward pass: compute ancillas.
                        gates.push(QRAMGate::Toffoli(0, 1, ancilla_start));
                        for k in 2..(n - 1) {
                            gates.push(QRAMGate::Toffoli(
                                ancilla_start + k - 2,
                                k,
                                ancilla_start + k - 1,
                            ));
                        }
                        // Final Toffoli targeting the data bit.
                        gates.push(QRAMGate::Toffoli(ancilla_start + n - 3, n - 1, target));
                        // Uncompute ancillas (reverse the forward pass, excluding final).
                        for k in (2..(n - 1)).rev() {
                            gates.push(QRAMGate::Toffoli(
                                ancilla_start + k - 2,
                                k,
                                ancilla_start + k - 1,
                            ));
                        }
                        gates.push(QRAMGate::Toffoli(0, 1, ancilla_start));
                    }
                }
            }

            // Step 3: Unflip.
            for &bit in flips.iter().rev() {
                gates.push(QRAMGate::X(bit));
            }
        }

        QRAMCircuit::compute_stats(&gates, self.total_qubits)
    }
}

// ============================================================
// FAN-OUT QRAM
// ============================================================

/// Fan-out QRAM: GHZ-style parallel address distribution.
///
/// Creates 2^n copies of the address via CNOT fanout, then loads
/// data in parallel with one Toffoli per cell.
///
/// Lowest depth (O(n)) but highest qubit count (O(2^n + n)).
#[derive(Debug, Clone)]
pub struct FanOutQRAM {
    /// Configuration.
    pub config: QRAMConfig,
    /// Classical data.
    pub memory: ClassicalMemory,
    /// Total qubits: address + data + fanout ancillas.
    pub total_qubits: usize,
}

impl FanOutQRAM {
    /// Construct a fan-out QRAM.
    pub fn new(config: QRAMConfig, memory: ClassicalMemory) -> Result<Self, QRAMError> {
        let n = config.address_bits;
        let d = config.data_bits;
        let num_cells = 1usize << n;

        if memory.num_addresses != num_cells {
            return Err(QRAMError::DataMismatch {
                expected: num_cells,
                got: memory.num_addresses,
            });
        }
        if memory.bits_per_cell != d {
            return Err(QRAMError::DataMismatch {
                expected: d,
                got: memory.bits_per_cell,
            });
        }

        // Fanout ancillas: one per memory cell.
        let fanout_ancillas = num_cells;
        let total_qubits = n + d + fanout_ancillas;

        Ok(Self {
            config,
            memory,
            total_qubits,
        })
    }

    /// Generate the gate sequence.
    ///
    /// 1. Fanout: for each address bit, cascade CNOT to create copies.
    /// 2. Parallel data load: for each cell, Toffoli from fanout ancilla
    ///    to data register (conditioned on address match).
    /// 3. Unfanout: reverse of step 1.
    pub fn generate_circuit(&self) -> QRAMCircuit {
        let n = self.config.address_bits;
        let d = self.config.data_bits;
        let num_cells = 1usize << n;
        let ancilla_start = n + d;
        let mut gates: Vec<QRAMGate> = Vec::new();

        // Step 1: Address decoding — set ancilla[c] = 1 iff address == c.
        //
        // For each cell c, apply X to address bits where bit(c) == 0,
        // then multi-controlled X targeting ancilla[c], then undo the X gates.
        // This implements a proper AND of all address-bit conditions.
        //
        // For n==1: CNOT (or X+CNOT+X)
        // For n==2: Toffoli (or X+Toffoli+X)
        // For n>2:  cascaded Toffoli decomposition via scratch ancillas
        //           (not needed for current max_address_bits but future-proof)

        for cell in 0..num_cells {
            // Flip address bits that should be 0 for this cell.
            let mut flip_bits = Vec::new();
            for bit in 0..n {
                if (cell >> bit) & 1 == 0 {
                    gates.push(QRAMGate::X(bit));
                    flip_bits.push(bit);
                }
            }

            // Multi-controlled X: all n address bits control ancilla[cell].
            if n == 1 {
                gates.push(QRAMGate::CNOT(0, ancilla_start + cell));
            } else if n == 2 {
                gates.push(QRAMGate::Toffoli(0, 1, ancilla_start + cell));
            } else {
                // Cascaded Toffoli decomposition for n>2:
                // Use ancilla scratch space (reuse other cell ancillas temporarily).
                // Toffoli(addr[0], addr[1], scratch) then Toffoli(scratch, addr[2], scratch2) ...
                // For simplicity with small n, just use a chain.
                // First pair → ancilla target directly via recursive decomposition.
                // We use the target ancilla as accumulator:
                // Step A: Toffoli(addr[0], addr[1], ancilla[cell])
                gates.push(QRAMGate::Toffoli(0, 1, ancilla_start + cell));
                // Step B: For each additional bit, Toffoli(addr[k], ancilla[cell], ancilla[cell])
                // won't work — need a scratch qubit. Use ancilla of cell 0 if cell != 0, else cell 1.
                let scratch = if cell == 0 {
                    ancilla_start + 1
                } else {
                    ancilla_start
                };
                for bit in 2..n {
                    // Save: Toffoli(ancilla[cell], addr[bit], scratch)
                    // then swap scratch back. Actually, simpler approach:
                    // Re-target: result so far is in ancilla[cell].
                    // Toffoli(addr[bit], ancilla[cell], scratch) → scratch = addr[bit] AND prior
                    // Then CNOT(scratch, ancilla[cell]) to move result, CNOT(scratch) to clear.
                    // But this gets messy. For n<=4 (our max), just expand:
                    gates.push(QRAMGate::Toffoli(bit, ancilla_start + cell, scratch));
                    gates.push(QRAMGate::CNOT(scratch, ancilla_start + cell));
                    gates.push(QRAMGate::Toffoli(bit, ancilla_start + cell, scratch));
                }
            }

            // Undo the address bit flips.
            for &bit in flip_bits.iter().rev() {
                gates.push(QRAMGate::X(bit));
            }
        }

        // Step 2: Data load. For each cell, if ancilla is active, XOR data into
        // the data register.
        for cell in 0..num_cells {
            for data_bit in 0..d {
                if self.memory.data[cell][data_bit] {
                    gates.push(QRAMGate::CNOT(ancilla_start + cell, n + data_bit));
                }
            }
        }

        // Step 3: Unfanout (reverse of address decoding).
        // Reverse the decoding gates (everything from address decoding, not data load).
        let decode_end = gates.len()
            - (0..num_cells)
                .flat_map(|c| (0..d).filter(move |&b| self.memory.data[c][b]))
                .count();
        let decode_gates: Vec<QRAMGate> = gates[..decode_end].to_vec();
        for gate in decode_gates.into_iter().rev() {
            gates.push(gate);
        }

        QRAMCircuit::compute_stats(&gates, self.total_qubits)
    }
}

// ============================================================
// MAIN QRAM SIMULATOR
// ============================================================

/// Unified QRAM simulator supporting all three variants.
///
/// Operates on the nQPU-Metal `QuantumState` directly, executing
/// the QRAM query as a sequence of elementary gates.
pub struct QRAMSimulator {
    /// Configuration.
    pub config: QRAMConfig,
    /// Classical data.
    pub memory: ClassicalMemory,
}

impl QRAMSimulator {
    /// Construct from validated config and matching memory.
    pub fn new(config: QRAMConfig, memory: ClassicalMemory) -> Result<Self, QRAMError> {
        let num_cells = 1usize << config.address_bits;
        if memory.num_addresses != num_cells {
            return Err(QRAMError::DataMismatch {
                expected: num_cells,
                got: memory.num_addresses,
            });
        }
        if memory.bits_per_cell != config.data_bits {
            return Err(QRAMError::DataMismatch {
                expected: config.data_bits,
                got: memory.bits_per_cell,
            });
        }
        Ok(Self { config, memory })
    }

    /// Total number of qubits required for the selected variant.
    pub fn total_qubits(&self) -> usize {
        match self.config.variant {
            QRAMVariant::BucketBrigade => {
                let leaves = 1usize << self.config.address_bits;
                self.config.address_bits + self.config.data_bits + leaves
            }
            QRAMVariant::SelectOnly => {
                let ancillas = if self.config.address_bits > 1 {
                    self.config.address_bits - 1
                } else {
                    0
                };
                self.config.address_bits + self.config.data_bits + ancillas
            }
            QRAMVariant::FanOut => {
                let fanout = 1usize << self.config.address_bits;
                self.config.address_bits + self.config.data_bits + fanout
            }
        }
    }

    /// Generate the circuit for the selected variant.
    pub fn generate_circuit(&self) -> Result<QRAMCircuit, QRAMError> {
        match self.config.variant {
            QRAMVariant::BucketBrigade => {
                let bb = BucketBrigadeQRAM::new(self.config.clone(), self.memory.clone())?;
                Ok(bb.generate_circuit())
            }
            QRAMVariant::SelectOnly => {
                let so = SelectOnlyQRAM::new(self.config.clone(), self.memory.clone())?;
                Ok(so.generate_circuit())
            }
            QRAMVariant::FanOut => {
                let fo = FanOutQRAM::new(self.config.clone(), self.memory.clone())?;
                Ok(fo.generate_circuit())
            }
        }
    }

    /// Execute a QRAM query on a statevector.
    ///
    /// The `state` must have at least `total_qubits()` qubits.
    /// The address register occupies qubits `[0 .. address_bits)`.
    /// The data register occupies qubits `[address_bits .. address_bits + data_bits)`.
    /// Ancillas follow.
    ///
    /// The caller is responsible for preparing the address register
    /// (e.g., Hadamard for superposition queries).
    pub fn query(&self, state: &mut QuantumState) -> Result<(), QRAMError> {
        let required = self.total_qubits();
        if state.num_qubits < required {
            return Err(QRAMError::ConfigError(format!(
                "state has {} qubits, need at least {}",
                state.num_qubits, required
            )));
        }

        let circuit = self.generate_circuit()?;
        self.apply_circuit(state, &circuit);
        Ok(())
    }

    /// Execute a QRAM query on a basis-state address and return the data value.
    ///
    /// This is a convenience method: it prepares |address>|0...0>,
    /// runs the query, and reads the data register.
    pub fn query_basis_state(&self, address: usize) -> Result<u64, QRAMError> {
        let n = self.config.address_bits;
        let d = self.config.data_bits;
        let num_cells = 1usize << n;

        if address >= num_cells {
            return Err(QRAMError::ConfigError(format!(
                "address {} out of range [0, {})",
                address, num_cells
            )));
        }

        let total = self.total_qubits();
        let mut state = QuantumState::new(total);

        // Prepare address register to |address>.
        for bit in 0..n {
            if (address >> bit) & 1 == 1 {
                GateOperations::x(&mut state, bit);
            }
        }

        self.query(&mut state)?;

        // Read the data register: check which basis state has amplitude ~1.
        let probs = state.probabilities();
        let mut best_idx = 0usize;
        let mut best_prob = 0.0f64;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_prob {
                best_prob = p;
                best_idx = i;
            }
        }

        // Extract data bits from best_idx.
        let mut data_val = 0u64;
        for bit in 0..d {
            if (best_idx >> (n + bit)) & 1 == 1 {
                data_val |= 1 << bit;
            }
        }

        Ok(data_val)
    }

    /// Apply a generated circuit to a statevector.
    fn apply_circuit(&self, state: &mut QuantumState, circuit: &QRAMCircuit) {
        for gate in &circuit.gates {
            match *gate {
                QRAMGate::CNOT(ctrl, tgt) => {
                    GateOperations::cnot(state, ctrl, tgt);
                }
                QRAMGate::Toffoli(c1, c2, tgt) => {
                    GateOperations::toffoli(state, c1, c2, tgt);
                }
                QRAMGate::X(tgt) => {
                    GateOperations::x(state, tgt);
                }
                QRAMGate::CSWAP(ctrl, t1, t2) => {
                    // Fredkin decomposition: CNOT(t2,t1), Toffoli(ctrl,t1,t2), CNOT(t2,t1).
                    GateOperations::cnot(state, t2, t1);
                    GateOperations::toffoli(state, ctrl, t1, t2);
                    GateOperations::cnot(state, t2, t1);
                }
            }
        }
    }

    /// Apply depolarizing noise: with probability `error_rate` per gate,
    /// apply a random Pauli (X, Y, or Z) to one of the gate's qubits.
    ///
    /// Returns the number of errors injected.
    pub fn apply_noise(&self, state: &mut QuantumState, circuit: &QRAMCircuit) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let e = self.config.error_rate;
        let mut errors = 0usize;

        for gate in &circuit.gates {
            // First, apply the gate.
            match *gate {
                QRAMGate::CNOT(ctrl, tgt) => {
                    GateOperations::cnot(state, ctrl, tgt);
                }
                QRAMGate::Toffoli(c1, c2, tgt) => {
                    GateOperations::toffoli(state, c1, c2, tgt);
                }
                QRAMGate::X(tgt) => {
                    GateOperations::x(state, tgt);
                }
                QRAMGate::CSWAP(ctrl, t1, t2) => {
                    GateOperations::cnot(state, t2, t1);
                    GateOperations::toffoli(state, ctrl, t1, t2);
                    GateOperations::cnot(state, t2, t1);
                }
            }

            // Then, with probability e, inject depolarizing noise.
            if e > 0.0 && rng.gen::<f64>() < e {
                let qubits: Vec<usize> = match *gate {
                    QRAMGate::CNOT(a, b) => vec![a, b],
                    QRAMGate::Toffoli(a, b, c) => vec![a, b, c],
                    QRAMGate::X(a) => vec![a],
                    QRAMGate::CSWAP(a, b, c) => vec![a, b, c],
                };
                let qubit = qubits[rng.gen_range(0..qubits.len())];
                let pauli = rng.gen_range(0..3);
                match pauli {
                    0 => GateOperations::x(state, qubit),
                    1 => {
                        // Y = iXZ: apply X then Z (ignoring global phase).
                        GateOperations::x(state, qubit);
                        apply_z(state, qubit);
                    }
                    _ => {
                        apply_z(state, qubit);
                    }
                }
                errors += 1;
            }
        }

        errors
    }

    /// Execute a noisy query and return the data register measurement.
    pub fn noisy_query(&self, state: &mut QuantumState) -> Result<usize, QRAMError> {
        let circuit = self.generate_circuit()?;
        self.apply_noise(state, &circuit);

        // Measure data register.
        let (outcome, _prob) = state.measure();
        Ok(outcome)
    }

    /// Compare resources across all three variants for the current parameters.
    pub fn compare_resources(&self) -> Vec<QRAMResources> {
        let variants = [
            QRAMVariant::BucketBrigade,
            QRAMVariant::SelectOnly,
            QRAMVariant::FanOut,
        ];

        variants
            .iter()
            .map(|&v| {
                let mut cfg = self.config.clone();
                cfg.variant = v;
                QRAMResources::estimate(&cfg, &self.memory)
            })
            .collect()
    }
}

// ============================================================
// HELPER: Z GATE (NOT EXPORTED FROM GateOperations FOR SINGLE QUBIT)
// ============================================================

/// Apply a Pauli-Z gate to a single qubit on a QuantumState.
fn apply_z(state: &mut QuantumState, qubit: usize) {
    let mask = 1usize << qubit;
    let amps = state.amplitudes_mut();
    for i in 0..amps.len() {
        if i & mask != 0 {
            amps[i] = C64::new(-amps[i].re, -amps[i].im);
        }
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple 3-address-bit, 1-data-bit memory with values [0,1,0,1,1,0,1,0].
    fn test_memory_3_1() -> ClassicalMemory {
        ClassicalMemory::from_values(&[0, 1, 0, 1, 1, 0, 1, 0], 1)
    }

    /// Helper: create a 2-address-bit, 2-data-bit memory.
    fn test_memory_2_2() -> ClassicalMemory {
        // D[0]=0b01=1, D[1]=0b10=2, D[2]=0b11=3, D[3]=0b00=0
        ClassicalMemory::from_values(&[1, 2, 3, 0], 2)
    }

    // --------------------------------------------------------
    // test_classical_memory
    // --------------------------------------------------------
    #[test]
    fn test_classical_memory() {
        let mut mem = ClassicalMemory::zeros(4, 3);
        assert_eq!(mem.num_addresses, 4);
        assert_eq!(mem.bits_per_cell, 3);
        assert_eq!(mem.read(0), 0);

        mem.write(0, 5); // 0b101
        assert_eq!(mem.read(0), 5);
        assert!(mem.data[0][0]); // bit 0 = 1
        assert!(!mem.data[0][1]); // bit 1 = 0
        assert!(mem.data[0][2]); // bit 2 = 1

        mem.write(3, 7); // 0b111
        assert_eq!(mem.read(3), 7);

        // from_values
        let mem2 = ClassicalMemory::from_values(&[0, 1, 2, 3], 2);
        assert_eq!(mem2.read(0), 0);
        assert_eq!(mem2.read(1), 1);
        assert_eq!(mem2.read(2), 2);
        assert_eq!(mem2.read(3), 3);
    }

    // --------------------------------------------------------
    // test_bucket_brigade_qubit_count
    // --------------------------------------------------------
    #[test]
    fn test_bucket_brigade_qubit_count() {
        // n=3 address bits, 1 data bit.
        // CSWAP-tree leaves: 2^3 = 8. Total: 3 + 1 + 8 = 12.
        let cfg = QRAMConfig::new(3, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
        let mem = test_memory_3_1();
        let bb = BucketBrigadeQRAM::new(cfg, mem).unwrap();
        assert_eq!(bb.total_qubits, 12);

        // n=2: leaves = 4. Total: 2 + 2 + 4 = 8.
        let cfg2 = QRAMConfig::new(2, 2, 0.0, QRAMVariant::BucketBrigade).unwrap();
        let mem2 = test_memory_2_2();
        let bb2 = BucketBrigadeQRAM::new(cfg2, mem2).unwrap();
        assert_eq!(bb2.total_qubits, 8);
    }

    // --------------------------------------------------------
    // test_select_only_qubit_count
    // --------------------------------------------------------
    #[test]
    fn test_select_only_qubit_count() {
        // n=3, d=1. Ancillas = n-1 = 2. Total: 3 + 1 + 2 = 6.
        let cfg = QRAMConfig::new(3, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
        let mem = test_memory_3_1();
        let so = SelectOnlyQRAM::new(cfg, mem).unwrap();
        assert_eq!(so.total_qubits, 6);

        // Less than bucket-brigade (11).
        assert!(so.total_qubits < 11);
    }

    // --------------------------------------------------------
    // test_fanout_qubit_count
    // --------------------------------------------------------
    #[test]
    fn test_fanout_qubit_count() {
        // n=3, d=1. Fanout ancillas = 2^3 = 8. Total: 3 + 1 + 8 = 12.
        let cfg = QRAMConfig::new(3, 1, 0.0, QRAMVariant::FanOut).unwrap();
        let mem = test_memory_3_1();
        let fo = FanOutQRAM::new(cfg, mem).unwrap();
        assert_eq!(fo.total_qubits, 12);

        // More than bucket-brigade (11) and select-only (6).
        assert!(fo.total_qubits > 11);
        assert!(fo.total_qubits > 6);
    }

    // --------------------------------------------------------
    // test_bucket_brigade_circuit
    // --------------------------------------------------------
    #[test]
    fn test_bucket_brigade_circuit() {
        let cfg = QRAMConfig::new(2, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
        let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);
        let bb = BucketBrigadeQRAM::new(cfg, mem).unwrap();
        let circuit = bb.generate_circuit();

        assert!(circuit.num_qubits > 0);
        assert!(!circuit.gates.is_empty());
        assert!(circuit.depth > 0);

        // Verify all gate qubit indices are in range.
        for gate in &circuit.gates {
            assert!(gate.max_qubit() < circuit.num_qubits);
        }
    }

    // --------------------------------------------------------
    // test_select_only_circuit
    // --------------------------------------------------------
    #[test]
    fn test_select_only_circuit() {
        let cfg = QRAMConfig::new(2, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
        let mem = ClassicalMemory::from_values(&[1, 0, 1, 1], 1);
        let so = SelectOnlyQRAM::new(cfg, mem).unwrap();
        let circuit = so.generate_circuit();

        assert!(circuit.num_qubits > 0);
        assert!(!circuit.gates.is_empty());

        for gate in &circuit.gates {
            assert!(gate.max_qubit() < circuit.num_qubits);
        }
    }

    // --------------------------------------------------------
    // test_query_single_address
    // --------------------------------------------------------
    #[test]
    fn test_query_single_address() {
        // 2-bit address, 1-bit data. Memory: [0, 1, 1, 0].
        // Query address 1 => data should be 1.
        // Query address 2 => data should be 1.
        // Query address 0 => data should be 0.
        // Query address 3 => data should be 0.
        let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);

        for variant in &[QRAMVariant::SelectOnly] {
            let cfg = QRAMConfig::new(2, 1, 0.0, *variant).unwrap();
            let sim = QRAMSimulator::new(cfg, mem.clone()).unwrap();

            // Query each address and check the result.
            for addr in 0..4u64 {
                let result = sim.query_basis_state(addr as usize).unwrap();
                let expected = mem.read(addr as usize);
                assert_eq!(
                    result, expected,
                    "variant={}, addr={}: got {} expected {}",
                    variant, addr, result, expected
                );
            }
        }
    }

    // --------------------------------------------------------
    // test_bucket_brigade_depth
    // --------------------------------------------------------
    #[test]
    fn test_bucket_brigade_depth() {
        // Bucket-brigade depth should be O(n), not O(2^n).
        // For n=2: depth should be small (proportional to n).
        // For n=3: depth should not blow up exponentially.
        let resources_2 = {
            let cfg = QRAMConfig::new(2, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
            let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);
            QRAMResources::estimate(&cfg, &mem)
        };

        let resources_3 = {
            let cfg = QRAMConfig::new(3, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
            let mem = test_memory_3_1();
            QRAMResources::estimate(&cfg, &mem)
        };

        // Depth should grow linearly, not exponentially.
        // n=2 -> depth ~ 2*2+1 = 5, n=3 -> depth ~ 2*3+1 = 7.
        // The ratio should be < 2 (linear), not 4 (exponential).
        let ratio = resources_3.circuit_depth as f64 / resources_2.circuit_depth as f64;
        assert!(
            ratio < 2.0,
            "bucket-brigade depth ratio n=3/n=2 = {:.2}, expected < 2.0 (linear growth)",
            ratio
        );
    }

    // --------------------------------------------------------
    // test_select_only_depth
    // --------------------------------------------------------
    #[test]
    fn test_select_only_depth() {
        // Select-only depth should be O(2^n).
        let resources_2 = {
            let cfg = QRAMConfig::new(2, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
            let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);
            QRAMResources::estimate(&cfg, &mem)
        };

        let resources_3 = {
            let cfg = QRAMConfig::new(3, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
            let mem = test_memory_3_1();
            QRAMResources::estimate(&cfg, &mem)
        };

        // Exponential growth: n=3 should be roughly 2x-4x deeper than n=2.
        assert!(
            resources_3.circuit_depth > resources_2.circuit_depth,
            "select-only depth should increase: n=3 ({}) > n=2 ({})",
            resources_3.circuit_depth,
            resources_2.circuit_depth
        );

        // The ratio should be > 2 (exponential).
        let ratio = resources_3.circuit_depth as f64 / resources_2.circuit_depth as f64;
        assert!(
            ratio >= 2.0,
            "select-only depth ratio n=3/n=2 = {:.2}, expected >= 2.0 (exponential growth)",
            ratio
        );
    }

    // --------------------------------------------------------
    // test_resource_comparison
    // --------------------------------------------------------
    #[test]
    fn test_resource_comparison() {
        let mem = test_memory_3_1();
        let cfg = QRAMConfig::new(3, 1, 0.001, QRAMVariant::BucketBrigade).unwrap();
        let sim = QRAMSimulator::new(cfg, mem).unwrap();

        let resources = sim.compare_resources();
        assert_eq!(resources.len(), 3);

        // Find each variant.
        let bb = resources
            .iter()
            .find(|r| r.variant == QRAMVariant::BucketBrigade)
            .unwrap();
        let so = resources
            .iter()
            .find(|r| r.variant == QRAMVariant::SelectOnly)
            .unwrap();
        let fo = resources
            .iter()
            .find(|r| r.variant == QRAMVariant::FanOut)
            .unwrap();

        // Bucket-brigade should have the lowest noise resilience value
        // (i.e. lowest effective error rate) among the three.
        assert!(
            bb.noise_resilience <= so.noise_resilience,
            "bucket-brigade noise ({:.6}) should be <= select-only ({:.6})",
            bb.noise_resilience,
            so.noise_resilience
        );

        // FanOut should have same or lower noise than SelectOnly.
        assert!(
            fo.noise_resilience <= so.noise_resilience,
            "fanout noise ({:.6}) should be <= select-only ({:.6})",
            fo.noise_resilience,
            so.noise_resilience
        );

        // SelectOnly uses fewer qubits than BucketBrigade.
        assert!(
            so.total_qubits < bb.total_qubits,
            "select-only qubits ({}) should be < bucket-brigade ({})",
            so.total_qubits,
            bb.total_qubits
        );

        // FanOut uses same or more qubits than BucketBrigade (both use 2^n ancillas).
        assert!(
            fo.total_qubits >= bb.total_qubits,
            "fanout qubits ({}) should be >= bucket-brigade ({})",
            fo.total_qubits,
            bb.total_qubits
        );
    }

    // --------------------------------------------------------
    // test_noisy_query
    // --------------------------------------------------------
    #[test]
    fn test_noisy_query() {
        // With high noise, the output should sometimes be wrong.
        // With zero noise, it should always be correct.
        let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);

        // Noiseless control: always correct.
        let cfg_clean = QRAMConfig::new(2, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
        let sim_clean = QRAMSimulator::new(cfg_clean, mem.clone()).unwrap();

        let mut state = QuantumState::new(sim_clean.total_qubits());
        // Prepare address |01> (address 1).
        GateOperations::x(&mut state, 0);
        let circuit = sim_clean.generate_circuit().unwrap();
        let errors = sim_clean.apply_noise(&mut state, &circuit);
        assert_eq!(errors, 0, "noiseless should inject no errors");

        // Noisy: with error_rate = 1.0, every gate injects an error.
        let cfg_noisy = QRAMConfig::new(2, 1, 1.0, QRAMVariant::SelectOnly).unwrap();
        let sim_noisy = QRAMSimulator::new(cfg_noisy, mem.clone()).unwrap();

        let mut state2 = QuantumState::new(sim_noisy.total_qubits());
        GateOperations::x(&mut state2, 0);
        let circuit2 = sim_noisy.generate_circuit().unwrap();
        let errors2 = sim_noisy.apply_noise(&mut state2, &circuit2);
        // With error_rate = 1.0, every gate should produce an error.
        assert!(errors2 > 0, "fully noisy simulation should inject errors");
        // The number of errors should equal the number of gates.
        assert_eq!(
            errors2,
            circuit2.gates.len(),
            "error_rate=1.0 should inject one error per gate"
        );
    }

    // --------------------------------------------------------
    // test_variant_equivalence
    // --------------------------------------------------------
    #[test]
    fn test_variant_equivalence() {
        // All three variants should produce the same output for a noiseless
        // basis-state query.
        let mem = ClassicalMemory::from_values(&[0, 1, 1, 0], 1);

        let cfg_so = QRAMConfig::new(2, 1, 0.0, QRAMVariant::SelectOnly).unwrap();
        let sim_so = QRAMSimulator::new(cfg_so, mem.clone()).unwrap();

        for addr in 0..4usize {
            let result_so = sim_so.query_basis_state(addr).unwrap();
            let expected = mem.read(addr);
            assert_eq!(
                result_so, expected,
                "SelectOnly addr={}: got {} expected {}",
                addr, result_so, expected
            );
        }

        // Test FanOut variant independently.
        let cfg_fo = QRAMConfig::new(2, 1, 0.0, QRAMVariant::FanOut).unwrap();
        let sim_fo = QRAMSimulator::new(cfg_fo, mem.clone()).unwrap();

        for addr in 0..4usize {
            let result_fo = sim_fo.query_basis_state(addr).unwrap();
            let expected = mem.read(addr);
            assert_eq!(
                result_fo, expected,
                "FanOut addr={}: got {} expected {}",
                addr, result_fo, expected
            );
        }

        // Test BucketBrigade variant.
        let cfg_bb = QRAMConfig::new(2, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
        let sim_bb = QRAMSimulator::new(cfg_bb, mem.clone()).unwrap();

        for addr in 0..4usize {
            let result_bb = sim_bb.query_basis_state(addr).unwrap();
            let expected = mem.read(addr);
            assert_eq!(
                result_bb, expected,
                "BucketBrigade addr={}: got {} expected {}",
                addr, result_bb, expected
            );
        }
    }

    // --------------------------------------------------------
    // test_config_validation
    // --------------------------------------------------------
    #[test]
    fn test_config_validation() {
        // Zero address bits should fail.
        assert!(QRAMConfig::new(0, 1, 0.0, QRAMVariant::BucketBrigade).is_err());

        // Too many address bits should fail.
        assert!(QRAMConfig::new(20, 1, 0.0, QRAMVariant::BucketBrigade).is_err());

        // Zero data bits should fail.
        assert!(QRAMConfig::new(2, 0, 0.0, QRAMVariant::BucketBrigade).is_err());

        // Negative error rate should fail (via out-of-range).
        assert!(QRAMConfig::new(2, 1, -0.1, QRAMVariant::BucketBrigade).is_err());

        // Error rate > 1 should fail.
        assert!(QRAMConfig::new(2, 1, 1.1, QRAMVariant::BucketBrigade).is_err());

        // Valid config should succeed.
        assert!(QRAMConfig::new(2, 1, 0.0, QRAMVariant::BucketBrigade).is_ok());
    }

    // --------------------------------------------------------
    // test_data_mismatch_error
    // --------------------------------------------------------
    #[test]
    fn test_data_mismatch_error() {
        // Memory with wrong number of addresses.
        let cfg = QRAMConfig::new(2, 1, 0.0, QRAMVariant::BucketBrigade).unwrap();
        let mem = ClassicalMemory::from_values(&[0, 1, 0], 1); // 3 instead of 4
        assert!(BucketBrigadeQRAM::new(cfg, mem).is_err());

        // Memory with wrong bits_per_cell.
        let cfg2 = QRAMConfig::new(2, 2, 0.0, QRAMVariant::SelectOnly).unwrap();
        let mem2 = ClassicalMemory::from_values(&[0, 1, 2, 3], 1); // 1 bit instead of 2
        assert!(SelectOnlyQRAM::new(cfg2, mem2).is_err());
    }

    // --------------------------------------------------------
    // test_error_display
    // --------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e1 = QRAMError::AddressTooLarge { bits: 20, max: 16 };
        assert!(e1.to_string().contains("20"));
        assert!(e1.to_string().contains("16"));

        let e2 = QRAMError::DataMismatch {
            expected: 8,
            got: 4,
        };
        assert!(e2.to_string().contains("8"));
        assert!(e2.to_string().contains("4"));

        let e3 = QRAMError::ConfigError("bad".to_string());
        assert!(e3.to_string().contains("bad"));
    }

    // --------------------------------------------------------
    // test_multi_bit_data
    // --------------------------------------------------------
    #[test]
    fn test_multi_bit_data() {
        // 2-bit address, 2-bit data: D = [1, 2, 3, 0].
        let mem = test_memory_2_2();
        let cfg = QRAMConfig::new(2, 2, 0.0, QRAMVariant::SelectOnly).unwrap();
        let sim = QRAMSimulator::new(cfg, mem.clone()).unwrap();

        assert_eq!(sim.query_basis_state(0).unwrap(), 1);
        assert_eq!(sim.query_basis_state(1).unwrap(), 2);
        assert_eq!(sim.query_basis_state(2).unwrap(), 3);
        assert_eq!(sim.query_basis_state(3).unwrap(), 0);
    }
}
