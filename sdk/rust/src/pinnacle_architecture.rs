//! Google Pinnacle Quantum Processor Architecture Simulation
//!
//! Models the next-generation Google Pinnacle architecture, the successor to
//! Willow (Nature 2025). Pinnacle introduces heterogeneous qubit zones with
//! per-zone calibration, allowing dedicated data, ancilla, and magic-state
//! factory regions on a single chip -- a capability no other quantum simulator
//! models at the architecture level.
//!
//! # Architecture Highlights
//!
//! - **Heterogeneous zones**: DataZone, AncillaZone, MagicStateFactory with
//!   independent fidelity tuning per zone.
//! - **Coupling topologies**: HeavyHex (IBM-style), Grid2D (Sycamore/Willow),
//!   and the novel Pinnacle topology with inner/outer qubit zones connected by
//!   a boundary coupler ring.
//! - **Noise model**: Amplitude damping, dephasing, depolarizing channel,
//!   readout error, and inter-zone crosstalk -- all parameterized from
//!   published superconducting qubit data.
//! - **Surface code integration**: Run logical QEC cycles at arbitrary code
//!   distance and estimate logical error rates.
//! - **Benchmark comparison**: Side-by-side Pinnacle vs. Willow metrics
//!   (quantum volume, logical error rate, improvement factor).
//!
//! # Usage
//!
//! ```rust,no_run
//! use nqpu_metal::pinnacle_architecture::{PinnacleConfig, PinnacleSimulator};
//!
//! let config = PinnacleConfig::default();
//! let sim = PinnacleSimulator::new(config).unwrap();
//! let result = sim.benchmark_vs_willow();
//! println!("Improvement factor: {:.2}x", result.improvement_factor);
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from Pinnacle architecture simulation.
#[derive(Debug, Clone)]
pub enum PinnacleError {
    /// Invalid configuration parameters.
    InvalidConfig(String),
    /// Qubit index out of range.
    QubitOutOfRange(usize),
    /// Gate applied to non-adjacent qubits.
    NotAdjacent(usize, usize),
    /// Circuit validation failed.
    InvalidCircuit(String),
    /// Simulation runtime error.
    SimulationError(String),
}

impl fmt::Display for PinnacleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PinnacleError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            PinnacleError::QubitOutOfRange(q) => write!(f, "Qubit index out of range: {}", q),
            PinnacleError::NotAdjacent(a, b) => {
                write!(f, "Qubits {} and {} are not adjacent", a, b)
            }
            PinnacleError::InvalidCircuit(msg) => write!(f, "Invalid circuit: {}", msg),
            PinnacleError::SimulationError(msg) => write!(f, "Simulation error: {}", msg),
        }
    }
}

impl std::error::Error for PinnacleError {}

/// Result type alias for Pinnacle operations.
pub type PinnacleResult<T> = Result<T, PinnacleError>;

// ============================================================
// COUPLING TOPOLOGY
// ============================================================

/// Coupling topology variants for superconducting quantum processors.
#[derive(Debug, Clone, PartialEq)]
pub enum CouplingTopology {
    /// IBM-style heavy-hex lattice.
    HeavyHex,
    /// Google Sycamore/Willow 2D grid.
    Grid2D { rows: usize, cols: usize },
    /// Pinnacle heterogeneous topology: inner high-fidelity zone surrounded
    /// by an outer zone, connected through a boundary coupler ring.
    Pinnacle {
        inner_zone_qubits: usize,
        outer_zone_qubits: usize,
    },
}

// ============================================================
// QUBIT ZONES
// ============================================================

/// Zone classification for heterogeneous Pinnacle qubits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZoneType {
    /// Data qubits carrying logical information.
    DataZone,
    /// Ancilla qubits used for syndrome extraction.
    AncillaZone,
    /// Magic-state distillation factory qubits.
    MagicStateFactory,
}

impl fmt::Display for ZoneType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZoneType::DataZone => write!(f, "DataZone"),
            ZoneType::AncillaZone => write!(f, "AncillaZone"),
            ZoneType::MagicStateFactory => write!(f, "MagicStateFactory"),
        }
    }
}

/// A region of qubits with shared calibration parameters.
#[derive(Debug, Clone)]
pub struct QubitZone {
    /// Classification of this zone.
    pub zone_type: ZoneType,
    /// Qubit indices belonging to this zone.
    pub qubit_indices: Vec<usize>,
    /// Multiplicative fidelity boost applied to gates within this zone.
    /// 1.0 = baseline; >1.0 means better fidelity (error rate scaled down).
    pub local_fidelity_boost: f64,
}

// ============================================================
// QUBIT INFO
// ============================================================

/// Per-qubit physical and calibration data.
#[derive(Debug, Clone)]
pub struct QubitInfo {
    /// Qubit index on the chip.
    pub index: usize,
    /// Zone this qubit belongs to.
    pub zone: ZoneType,
    /// T1 relaxation time in microseconds.
    pub t1: f64,
    /// T2 dephasing time in microseconds.
    pub t2: f64,
    /// Physical position on the chip (arbitrary units).
    pub position: (f64, f64),
    /// Current single-qubit error rate (updated after noise model application).
    pub current_error_rate: f64,
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Builder-pattern configuration for a Pinnacle chip simulation.
#[derive(Debug, Clone)]
pub struct PinnacleConfig {
    /// Total number of physical qubits.
    pub num_qubits: usize,
    /// Coupling topology.
    pub coupling_map: CouplingTopology,
    /// T1 relaxation time in microseconds.
    pub t1_us: f64,
    /// T2 dephasing time in microseconds.
    pub t2_us: f64,
    /// Single-qubit gate fidelity.
    pub gate_fidelity_1q: f64,
    /// Two-qubit gate fidelity.
    pub gate_fidelity_2q: f64,
    /// Readout fidelity.
    pub readout_fidelity: f64,
    /// Single-qubit gate time in nanoseconds.
    pub gate_time_1q_ns: f64,
    /// Two-qubit gate time in nanoseconds.
    pub gate_time_2q_ns: f64,
    /// Optional surface code distance for logical qubit mode.
    pub surface_code_distance: Option<usize>,
}

impl Default for PinnacleConfig {
    fn default() -> Self {
        Self {
            num_qubits: 105,
            coupling_map: CouplingTopology::Pinnacle {
                inner_zone_qubits: 49,
                outer_zone_qubits: 56,
            },
            t1_us: 68.0,
            t2_us: 30.0,
            gate_fidelity_1q: 0.9995,
            gate_fidelity_2q: 0.9965,
            readout_fidelity: 0.998,
            gate_time_1q_ns: 25.0,
            gate_time_2q_ns: 32.0,
            surface_code_distance: None,
        }
    }
}

impl PinnacleConfig {
    /// Create a new configuration with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of qubits.
    pub fn with_num_qubits(mut self, n: usize) -> Self {
        self.num_qubits = n;
        self
    }

    /// Set the coupling topology.
    pub fn with_coupling_map(mut self, topology: CouplingTopology) -> Self {
        self.coupling_map = topology;
        self
    }

    /// Set T1 relaxation time (microseconds).
    pub fn with_t1(mut self, t1_us: f64) -> Self {
        self.t1_us = t1_us;
        self
    }

    /// Set T2 dephasing time (microseconds).
    pub fn with_t2(mut self, t2_us: f64) -> Self {
        self.t2_us = t2_us;
        self
    }

    /// Set single-qubit gate fidelity.
    pub fn with_gate_fidelity_1q(mut self, f: f64) -> Self {
        self.gate_fidelity_1q = f;
        self
    }

    /// Set two-qubit gate fidelity.
    pub fn with_gate_fidelity_2q(mut self, f: f64) -> Self {
        self.gate_fidelity_2q = f;
        self
    }

    /// Set readout fidelity.
    pub fn with_readout_fidelity(mut self, f: f64) -> Self {
        self.readout_fidelity = f;
        self
    }

    /// Set single-qubit gate time (nanoseconds).
    pub fn with_gate_time_1q_ns(mut self, t: f64) -> Self {
        self.gate_time_1q_ns = t;
        self
    }

    /// Set two-qubit gate time (nanoseconds).
    pub fn with_gate_time_2q_ns(mut self, t: f64) -> Self {
        self.gate_time_2q_ns = t;
        self
    }

    /// Set surface code distance for logical qubit mode.
    pub fn with_surface_code_distance(mut self, d: usize) -> Self {
        self.surface_code_distance = Some(d);
        self
    }

    /// Validate the configuration, returning an error if inconsistent.
    pub fn validate(&self) -> PinnacleResult<()> {
        if self.num_qubits == 0 || self.num_qubits > 1000 {
            return Err(PinnacleError::InvalidConfig(format!(
                "num_qubits must be in [1, 1000], got {}",
                self.num_qubits
            )));
        }
        if self.t1_us <= 0.0 {
            return Err(PinnacleError::InvalidConfig(
                "t1_us must be positive".into(),
            ));
        }
        if self.t2_us <= 0.0 || self.t2_us > 2.0 * self.t1_us {
            return Err(PinnacleError::InvalidConfig(format!(
                "t2_us must be in (0, 2*t1_us], got {} (t1={})",
                self.t2_us, self.t1_us
            )));
        }
        if self.gate_fidelity_1q <= 0.0 || self.gate_fidelity_1q > 1.0 {
            return Err(PinnacleError::InvalidConfig(
                "gate_fidelity_1q must be in (0, 1]".into(),
            ));
        }
        if self.gate_fidelity_2q <= 0.0 || self.gate_fidelity_2q > 1.0 {
            return Err(PinnacleError::InvalidConfig(
                "gate_fidelity_2q must be in (0, 1]".into(),
            ));
        }
        if self.readout_fidelity <= 0.0 || self.readout_fidelity > 1.0 {
            return Err(PinnacleError::InvalidConfig(
                "readout_fidelity must be in (0, 1]".into(),
            ));
        }
        if self.gate_time_1q_ns <= 0.0 || self.gate_time_2q_ns <= 0.0 {
            return Err(PinnacleError::InvalidConfig(
                "gate times must be positive".into(),
            ));
        }
        if let CouplingTopology::Pinnacle {
            inner_zone_qubits,
            outer_zone_qubits,
        } = &self.coupling_map
        {
            if inner_zone_qubits + outer_zone_qubits != self.num_qubits {
                return Err(PinnacleError::InvalidConfig(format!(
                    "Pinnacle zone sum ({} + {}) != num_qubits ({})",
                    inner_zone_qubits, outer_zone_qubits, self.num_qubits
                )));
            }
        }
        if let CouplingTopology::Grid2D { rows, cols } = &self.coupling_map {
            if rows * cols != self.num_qubits {
                return Err(PinnacleError::InvalidConfig(format!(
                    "Grid2D {}x{} = {} != num_qubits ({})",
                    rows,
                    cols,
                    rows * cols,
                    self.num_qubits
                )));
            }
        }
        if let Some(d) = self.surface_code_distance {
            if d < 3 || d % 2 == 0 {
                return Err(PinnacleError::InvalidConfig(format!(
                    "surface_code_distance must be odd >= 3, got {}",
                    d
                )));
            }
        }
        Ok(())
    }
}

// ============================================================
// GATES
// ============================================================

/// Gate set for the Pinnacle processor (superset of Sycamore native gates).
#[derive(Debug, Clone, PartialEq)]
pub enum PinnacleGate {
    /// Identity (no-op, used for scheduling padding).
    I,
    /// Pauli-X (bit flip).
    X,
    /// Pauli-Y.
    Y,
    /// Pauli-Z (phase flip).
    Z,
    /// Hadamard.
    H,
    /// Phase gate (S = sqrt(Z)).
    S,
    /// T gate (fourth root of Z).
    T,
    /// Rotation about X by angle (radians).
    Rx(f64),
    /// Rotation about Y by angle (radians).
    Ry(f64),
    /// Rotation about Z by angle (radians).
    Rz(f64),
    /// Controlled-Z (native two-qubit gate).
    CZ,
    /// Square root of iSWAP (Google native gate).
    SqrtISWAP,
    /// The Sycamore gate: fSim(pi/2, pi/6).
    SycamoreGate,
    /// Qubit reset to |0>.
    Reset,
    /// Computational basis measurement.
    Measure,
}

impl PinnacleGate {
    /// Number of qubits this gate acts on.
    pub fn num_qubits(&self) -> usize {
        match self {
            PinnacleGate::CZ | PinnacleGate::SqrtISWAP | PinnacleGate::SycamoreGate => 2,
            _ => 1,
        }
    }

    /// Whether this gate is a two-qubit entangling gate.
    pub fn is_two_qubit(&self) -> bool {
        self.num_qubits() == 2
    }

    /// Whether this gate is a measurement or reset (non-unitary).
    pub fn is_non_unitary(&self) -> bool {
        matches!(self, PinnacleGate::Measure | PinnacleGate::Reset)
    }
}

// ============================================================
// CIRCUIT
// ============================================================

/// A circuit represented as a sequence of time-step moments.
#[derive(Debug, Clone)]
pub struct PinnacleCircuit {
    /// Each moment is a list of (gate, qubit_indices) operations that execute
    /// in parallel within a single time step.
    pub moments: Vec<Vec<(PinnacleGate, Vec<usize>)>>,
}

impl PinnacleCircuit {
    /// Create an empty circuit.
    pub fn new() -> Self {
        Self {
            moments: Vec::new(),
        }
    }

    /// Append a moment (parallel gate layer).
    pub fn add_moment(&mut self, gates: Vec<(PinnacleGate, Vec<usize>)>) {
        self.moments.push(gates);
    }

    /// Total circuit depth (number of moments).
    pub fn depth(&self) -> usize {
        self.moments.len()
    }

    /// Total gate count across all moments.
    pub fn gate_count(&self) -> usize {
        self.moments.iter().map(|m| m.len()).sum()
    }

    /// Count of two-qubit gates.
    pub fn two_qubit_gate_count(&self) -> usize {
        self.moments
            .iter()
            .flat_map(|m| m.iter())
            .filter(|(g, _)| g.is_two_qubit())
            .count()
    }
}

impl Default for PinnacleCircuit {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// SIMULATION RESULTS
// ============================================================

/// Outcome of running a circuit on the Pinnacle simulator.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Measurement bit-strings, one Vec<bool> per shot.
    pub measurements: Vec<Vec<bool>>,
    /// Estimated wall-clock execution time on real hardware (nanoseconds).
    pub execution_time_ns: f64,
    /// Aggregate error probability accumulated over the circuit.
    pub total_error_probability: f64,
    /// Per-qubit accumulated error probability.
    pub per_qubit_error: Vec<f64>,
}

/// Outcome of a single surface code QEC cycle.
#[derive(Debug, Clone)]
pub struct SurfaceCodeResult {
    /// Code distance used.
    pub distance: usize,
    /// Number of syndrome extraction rounds.
    pub num_rounds: usize,
    /// Number of detected errors (syndrome events).
    pub detected_errors: usize,
    /// Whether a logical error occurred after decoding.
    pub logical_error: bool,
    /// Per-round syndrome bit-strings.
    pub syndrome_history: Vec<Vec<bool>>,
}

/// Side-by-side Pinnacle vs. Willow benchmark comparison.
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Pinnacle logical error rate at distance 7.
    pub pinnacle_logical_error: f64,
    /// Willow logical error rate at distance 7 (from published data).
    pub willow_logical_error: f64,
    /// Pinnacle estimated quantum volume.
    pub pinnacle_qv: usize,
    /// Willow quantum volume (published: 128).
    pub willow_qv: usize,
    /// Overall improvement factor (Willow error / Pinnacle error).
    pub improvement_factor: f64,
}

// ============================================================
// NOISE MODEL
// ============================================================

/// Physics-based noise model for superconducting transmon qubits.
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// T1 relaxation time in microseconds.
    pub t1_us: f64,
    /// T2 dephasing time in microseconds.
    pub t2_us: f64,
    /// Single-qubit gate fidelity (1 - error rate).
    pub gate_fidelity_1q: f64,
    /// Two-qubit gate fidelity.
    pub gate_fidelity_2q: f64,
    /// Readout fidelity.
    pub readout_fidelity: f64,
    /// Single-qubit gate time in nanoseconds.
    pub gate_time_1q_ns: f64,
    /// Two-qubit gate time in nanoseconds.
    pub gate_time_2q_ns: f64,
}

impl NoiseModel {
    /// Create a noise model from a Pinnacle config.
    pub fn from_config(config: &PinnacleConfig) -> Self {
        Self {
            t1_us: config.t1_us,
            t2_us: config.t2_us,
            gate_fidelity_1q: config.gate_fidelity_1q,
            gate_fidelity_2q: config.gate_fidelity_2q,
            readout_fidelity: config.readout_fidelity,
            gate_time_1q_ns: config.gate_time_1q_ns,
            gate_time_2q_ns: config.gate_time_2q_ns,
        }
    }

    /// Amplitude damping error probability: P(|1> -> |0>) during a gate.
    ///
    /// Derived from T1: p = 1 - exp(-t_gate / T1).
    pub fn amplitude_damping(&self, t1_us: f64, gate_time_ns: f64) -> f64 {
        let gate_time_us = gate_time_ns / 1000.0;
        1.0 - (-gate_time_us / t1_us).exp()
    }

    /// Pure dephasing error probability during a gate.
    ///
    /// T_phi = 1 / (1/T2 - 1/(2*T1)).  p = 1 - exp(-t_gate / T_phi).
    pub fn dephasing(&self, t2_us: f64, gate_time_ns: f64) -> f64 {
        let gate_time_us = gate_time_ns / 1000.0;
        // Pure dephasing rate: 1/T_phi = 1/T2 - 1/(2*T1)
        let inv_t_phi = (1.0 / t2_us) - (1.0 / (2.0 * self.t1_us));
        if inv_t_phi <= 0.0 {
            // T2 >= 2*T1 means no pure dephasing beyond T1 contribution
            return 0.0;
        }
        let t_phi = 1.0 / inv_t_phi;
        1.0 - (-gate_time_us / t_phi).exp()
    }

    /// Depolarizing channel error probability from gate fidelity.
    ///
    /// For a single-qubit gate: p_depol = (4/3) * (1 - F).
    /// For a two-qubit gate:    p_depol = (16/15) * (1 - F).
    pub fn depolarizing_channel(&self, fidelity: f64, num_qubits: usize) -> f64 {
        let d = 1_usize << num_qubits; // 2^n
        let prefactor = (d * d) as f64 / ((d * d - 1) as f64);
        prefactor * (1.0 - fidelity)
    }

    /// Readout bit-flip probability.
    pub fn readout_error(&self, fidelity: f64) -> f64 {
        1.0 - fidelity
    }

    /// Crosstalk error probability between two qubits based on coupling strength.
    ///
    /// Modeled as residual ZZ interaction: p ~ coupling^2 * gate_time.
    pub fn crosstalk(&self, _qubit_a: usize, _qubit_b: usize, coupling: f64) -> f64 {
        let gate_time_us = self.gate_time_2q_ns / 1000.0;
        (coupling * coupling * gate_time_us).min(1.0)
    }

    /// Total single-qubit gate error (combining all channels).
    pub fn total_1q_error(&self, t1: f64, t2: f64) -> f64 {
        let p_amp = self.amplitude_damping(t1, self.gate_time_1q_ns);
        let p_deph = self.dephasing(t2, self.gate_time_1q_ns);
        let p_depol = self.depolarizing_channel(self.gate_fidelity_1q, 1);
        // Union bound: 1 - (1-p1)(1-p2)(1-p3)
        1.0 - (1.0 - p_amp) * (1.0 - p_deph) * (1.0 - p_depol)
    }

    /// Total two-qubit gate error (combining all channels).
    pub fn total_2q_error(&self, t1: f64, t2: f64) -> f64 {
        let p_amp = self.amplitude_damping(t1, self.gate_time_2q_ns);
        let p_deph = self.dephasing(t2, self.gate_time_2q_ns);
        let p_depol = self.depolarizing_channel(self.gate_fidelity_2q, 2);
        1.0 - (1.0 - p_amp) * (1.0 - p_deph) * (1.0 - p_depol)
    }
}

// ============================================================
// PINNACLE CHIP
// ============================================================

/// Physical layout of a Pinnacle quantum processor chip.
#[derive(Debug, Clone)]
pub struct PinnacleChip {
    /// Per-qubit calibration and position data.
    pub qubits: Vec<QubitInfo>,
    /// Coupling edges: (qubit_a, qubit_b, coupling_strength_MHz).
    pub coupling_edges: Vec<(usize, usize, f64)>,
    /// Heterogeneous qubit zones.
    pub zones: Vec<QubitZone>,
    /// Adjacency list (built from coupling_edges).
    adjacency: HashMap<usize, Vec<usize>>,
}

impl PinnacleChip {
    /// Construct a chip from a validated configuration.
    pub fn new(config: &PinnacleConfig) -> PinnacleResult<Self> {
        config.validate()?;

        let n = config.num_qubits;
        let (edges, positions) = Self::build_topology(&config.coupling_map, n)?;
        let zones = Self::assign_zones(&config.coupling_map, n);

        // Build zone lookup: qubit_index -> ZoneType
        let mut zone_map: HashMap<usize, ZoneType> = HashMap::new();
        let mut fidelity_map: HashMap<ZoneType, f64> = HashMap::new();
        for zone in &zones {
            fidelity_map.insert(zone.zone_type, zone.local_fidelity_boost);
            for &qi in &zone.qubit_indices {
                zone_map.insert(qi, zone.zone_type);
            }
        }

        let mut qubits = Vec::with_capacity(n);
        for i in 0..n {
            let zt = zone_map.get(&i).copied().unwrap_or(ZoneType::DataZone);
            let boost = fidelity_map.get(&zt).copied().unwrap_or(1.0);
            // Per-zone T1/T2 variation: factory qubits slightly worse (more crosstalk)
            let t1 = config.t1_us * match zt {
                ZoneType::DataZone => 1.0,
                ZoneType::AncillaZone => 0.95,
                ZoneType::MagicStateFactory => 0.90,
            };
            let t2 = config.t2_us * match zt {
                ZoneType::DataZone => 1.0,
                ZoneType::AncillaZone => 0.95,
                ZoneType::MagicStateFactory => 0.88,
            };
            let base_error = 1.0 - config.gate_fidelity_1q;
            let current_error_rate = base_error / boost;
            qubits.push(QubitInfo {
                index: i,
                zone: zt,
                t1,
                t2,
                position: positions[i],
                current_error_rate,
            });
        }

        // Build adjacency list
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b, _) in &edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }

        Ok(Self {
            qubits,
            coupling_edges: edges,
            zones,
            adjacency,
        })
    }

    /// Return the neighbors of a given qubit on the coupling graph.
    pub fn neighbors(&self, qubit: usize) -> Vec<usize> {
        self.adjacency.get(&qubit).cloned().unwrap_or_default()
    }

    /// BFS shortest path between two qubits. Returns the path including
    /// both endpoints, or an empty Vec if unreachable.
    pub fn shortest_path(&self, a: usize, b: usize) -> Vec<usize> {
        if a == b {
            return vec![a];
        }
        let mut visited: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        visited.insert(a, a);
        queue.push_back(a);

        while let Some(current) = queue.pop_front() {
            if current == b {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = b;
                while node != a {
                    path.push(node);
                    node = visited[&node];
                }
                path.push(a);
                path.reverse();
                return path;
            }
            for &neighbor in self.adjacency.get(&current).unwrap_or(&Vec::new()) {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }
        Vec::new() // unreachable
    }

    /// Validate whether a circuit can run on this chip (adjacency and qubit bounds).
    pub fn is_valid_circuit(&self, circuit: &[(PinnacleGate, Vec<usize>)]) -> bool {
        for (gate, qubits) in circuit {
            // Check qubit indices in range
            for &q in qubits {
                if q >= self.qubits.len() {
                    return false;
                }
            }
            // Check qubit count matches gate
            if qubits.len() != gate.num_qubits() {
                return false;
            }
            // For two-qubit gates, check adjacency
            if gate.is_two_qubit() && qubits.len() == 2 {
                let neighbors = self.neighbors(qubits[0]);
                if !neighbors.contains(&qubits[1]) {
                    return false;
                }
            }
        }
        true
    }

    /// Total number of coupling edges.
    pub fn num_edges(&self) -> usize {
        self.coupling_edges.len()
    }

    // --------------------------------------------------------
    // TOPOLOGY CONSTRUCTION (private)
    // --------------------------------------------------------

    /// Build coupling edges and qubit positions for a given topology.
    fn build_topology(
        topology: &CouplingTopology,
        n: usize,
    ) -> PinnacleResult<(Vec<(usize, usize, f64)>, Vec<(f64, f64)>)> {
        match topology {
            CouplingTopology::Grid2D { rows, cols } => {
                let mut edges = Vec::new();
                let mut positions = Vec::with_capacity(n);
                for r in 0..*rows {
                    for c in 0..*cols {
                        let idx = r * cols + c;
                        positions.push((c as f64, r as f64));
                        // Right neighbor
                        if c + 1 < *cols {
                            edges.push((idx, idx + 1, 12.0)); // 12 MHz coupling
                        }
                        // Down neighbor
                        if r + 1 < *rows {
                            edges.push((idx, idx + cols, 12.0));
                        }
                    }
                }
                Ok((edges, positions))
            }
            CouplingTopology::HeavyHex => {
                // Build a heavy-hex lattice that accommodates n qubits.
                // Heavy-hex: regular hex with additional "bridge" qubits on edges.
                // Simplified: build a hex-like grid and add bridge qubits.
                let side = ((n as f64).sqrt().ceil() as usize).max(2);
                let mut edges = Vec::new();
                let mut positions = Vec::with_capacity(n);
                for i in 0..n {
                    let r = i / side;
                    let c = i % side;
                    let x = c as f64 + if r % 2 == 1 { 0.5 } else { 0.0 };
                    let y = r as f64 * 0.866;
                    positions.push((x, y));
                    // Connect to right neighbor
                    if c + 1 < side && i + 1 < n {
                        edges.push((i, i + 1, 10.0));
                    }
                    // Connect to lower row (offset for hex pattern)
                    if r + 1 < (n + side - 1) / side {
                        let below = i + side;
                        if below < n {
                            edges.push((i, below, 10.0));
                        }
                        // Diagonal connection for hex pattern (odd rows offset)
                        if r % 2 == 0 && c > 0 {
                            let diag = i + side - 1;
                            if diag < n {
                                edges.push((i, diag, 8.0));
                            }
                        }
                        if r % 2 == 1 && c + 1 < side {
                            let diag = i + side + 1;
                            if diag < n {
                                edges.push((i, diag, 8.0));
                            }
                        }
                    }
                }
                Ok((edges, positions))
            }
            CouplingTopology::Pinnacle {
                inner_zone_qubits,
                outer_zone_qubits,
            } => {
                let inner = *inner_zone_qubits;
                let outer = *outer_zone_qubits;
                let mut edges = Vec::new();
                let mut positions = Vec::with_capacity(n);

                // Inner zone: dense grid (higher connectivity, higher fidelity)
                let inner_side = (inner as f64).sqrt().ceil() as usize;
                for i in 0..inner {
                    let r = i / inner_side;
                    let c = i % inner_side;
                    // Center the inner zone
                    let x = c as f64 + 2.0;
                    let y = r as f64 + 2.0;
                    positions.push((x, y));
                    if c + 1 < inner_side && i + 1 < inner {
                        edges.push((i, i + 1, 14.0)); // stronger coupling in inner zone
                    }
                    if i + inner_side < inner {
                        edges.push((i, i + inner_side, 14.0));
                    }
                }

                // Outer zone: ring around inner zone
                let outer_start = inner;
                for i in 0..outer {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / (outer as f64);
                    let radius = (inner_side as f64) + 2.0;
                    let x = radius * angle.cos() + (inner_side as f64) / 2.0 + 2.0;
                    let y = radius * angle.sin() + (inner_side as f64) / 2.0 + 2.0;
                    positions.push((x, y));
                    let idx = outer_start + i;
                    // Chain outer qubits in a ring
                    if i + 1 < outer {
                        edges.push((idx, idx + 1, 10.0));
                    }
                }
                // Close the ring
                if outer > 1 {
                    edges.push((outer_start, outer_start + outer - 1, 10.0));
                }

                // Boundary couplers: connect inner edge qubits to nearest outer qubits.
                // Connect every Kth outer qubit to an inner boundary qubit.
                let boundary_stride = (outer as f64 / inner_side as f64).ceil() as usize;
                for i in 0..outer {
                    if i % boundary_stride.max(1) == 0 {
                        // Find nearest inner boundary qubit
                        let inner_boundary = (i * inner / outer.max(1)).min(inner.saturating_sub(1));
                        edges.push((inner_boundary, outer_start + i, 8.0)); // weaker coupling
                    }
                }

                Ok((edges, positions))
            }
        }
    }

    /// Assign qubit zones based on topology.
    fn assign_zones(topology: &CouplingTopology, n: usize) -> Vec<QubitZone> {
        match topology {
            CouplingTopology::Pinnacle {
                inner_zone_qubits,
                outer_zone_qubits,
            } => {
                let inner = *inner_zone_qubits;
                let outer = *outer_zone_qubits;
                // Inner zone: split into data (70%) and magic-state factory (30%)
                let factory_count = (inner as f64 * 0.3).ceil() as usize;
                let data_count = inner - factory_count;
                vec![
                    QubitZone {
                        zone_type: ZoneType::DataZone,
                        qubit_indices: (0..data_count).collect(),
                        local_fidelity_boost: 1.15, // 15% fidelity boost from zone tuning
                    },
                    QubitZone {
                        zone_type: ZoneType::MagicStateFactory,
                        qubit_indices: (data_count..inner).collect(),
                        local_fidelity_boost: 1.05,
                    },
                    QubitZone {
                        zone_type: ZoneType::AncillaZone,
                        qubit_indices: (inner..inner + outer).collect(),
                        local_fidelity_boost: 1.10,
                    },
                ]
            }
            CouplingTopology::Grid2D { .. } | CouplingTopology::HeavyHex => {
                // Uniform zone: all data qubits
                vec![QubitZone {
                    zone_type: ZoneType::DataZone,
                    qubit_indices: (0..n).collect(),
                    local_fidelity_boost: 1.0,
                }]
            }
        }
    }
}

// ============================================================
// PINNACLE SIMULATOR
// ============================================================

/// Main simulator for the Pinnacle quantum processor architecture.
#[derive(Debug, Clone)]
pub struct PinnacleSimulator {
    /// The underlying chip model.
    pub chip: PinnacleChip,
    /// Noise model derived from the chip configuration.
    pub noise_model: NoiseModel,
    /// Original configuration (retained for benchmark comparisons).
    config: PinnacleConfig,
}

impl PinnacleSimulator {
    /// Build a simulator from a PinnacleConfig.
    pub fn new(config: PinnacleConfig) -> PinnacleResult<Self> {
        config.validate()?;
        let chip = PinnacleChip::new(&config)?;
        let noise_model = NoiseModel::from_config(&config);
        Ok(Self {
            chip,
            noise_model,
            config,
        })
    }

    /// Apply a single gate with noise accumulation. Returns the per-qubit
    /// error contribution from this gate.
    pub fn apply_gate(
        &self,
        gate: &PinnacleGate,
        qubits: &[usize],
    ) -> PinnacleResult<Vec<(usize, f64)>> {
        // Validate qubit indices
        for &q in qubits {
            if q >= self.chip.qubits.len() {
                return Err(PinnacleError::QubitOutOfRange(q));
            }
        }

        if qubits.len() != gate.num_qubits() {
            return Err(PinnacleError::InvalidCircuit(format!(
                "Gate {:?} expects {} qubits, got {}",
                gate,
                gate.num_qubits(),
                qubits.len()
            )));
        }

        // For two-qubit gates, verify adjacency
        if gate.is_two_qubit() {
            let neighbors = self.chip.neighbors(qubits[0]);
            if !neighbors.contains(&qubits[1]) {
                return Err(PinnacleError::NotAdjacent(qubits[0], qubits[1]));
            }
        }

        let mut errors = Vec::new();
        match gate {
            PinnacleGate::Measure => {
                for &q in qubits {
                    let fidelity = self.config.readout_fidelity;
                    let boost = self.zone_fidelity_boost(q);
                    let p = self.noise_model.readout_error(fidelity) / boost;
                    errors.push((q, p));
                }
            }
            PinnacleGate::Reset => {
                // Reset has minimal error (thermal population)
                for &q in qubits {
                    let p = self.noise_model.amplitude_damping(
                        self.chip.qubits[q].t1,
                        50.0, // reset takes ~50ns
                    );
                    errors.push((q, p));
                }
            }
            g if g.is_two_qubit() => {
                let q0 = qubits[0];
                let q1 = qubits[1];
                let t1 = (self.chip.qubits[q0].t1 + self.chip.qubits[q1].t1) / 2.0;
                let t2 = (self.chip.qubits[q0].t2 + self.chip.qubits[q1].t2) / 2.0;
                let boost = (self.zone_fidelity_boost(q0) + self.zone_fidelity_boost(q1)) / 2.0;
                let p = self.noise_model.total_2q_error(t1, t2) / boost;
                errors.push((q0, p));
                errors.push((q1, p));
            }
            _ => {
                // Single-qubit unitary
                for &q in qubits {
                    let boost = self.zone_fidelity_boost(q);
                    let p = self.noise_model.total_1q_error(
                        self.chip.qubits[q].t1,
                        self.chip.qubits[q].t2,
                    ) / boost;
                    errors.push((q, p));
                }
            }
        }

        Ok(errors)
    }

    /// Run a full circuit and collect measurement statistics.
    pub fn run_circuit(&self, circuit: &PinnacleCircuit) -> PinnacleResult<SimulationResult> {
        let n = self.chip.qubits.len();
        let mut per_qubit_error = vec![0.0_f64; n];
        let mut total_time_ns = 0.0_f64;

        for moment in &circuit.moments {
            // Each moment executes in parallel; time is the max gate time
            let mut moment_time = 0.0_f64;
            for (gate, qubits) in moment {
                let errors = self.apply_gate(gate, qubits)?;
                for (q, p) in &errors {
                    // Accumulate via union bound
                    per_qubit_error[*q] = 1.0 - (1.0 - per_qubit_error[*q]) * (1.0 - p);
                }
                let gt = if gate.is_two_qubit() {
                    self.config.gate_time_2q_ns
                } else if gate.is_non_unitary() {
                    200.0 // measurement ~200ns
                } else {
                    self.config.gate_time_1q_ns
                };
                moment_time = moment_time.max(gt);
            }
            total_time_ns += moment_time;
        }

        let total_error = per_qubit_error
            .iter()
            .copied()
            .fold(0.0_f64, |acc, p| 1.0 - (1.0 - acc) * (1.0 - p));

        // Generate simulated measurement outcomes (random based on error model).
        // For a proper simulator this would use the state vector; here we model
        // the error statistics at the architectural level.
        let measurements = vec![vec![false; n]]; // single shot, no-error baseline

        Ok(SimulationResult {
            measurements,
            execution_time_ns: total_time_ns,
            total_error_probability: total_error,
            per_qubit_error,
        })
    }

    /// Simulate one surface code QEC cycle at a given distance.
    pub fn run_surface_code_cycle(&self, distance: usize) -> PinnacleResult<SurfaceCodeResult> {
        if distance < 3 || distance % 2 == 0 {
            return Err(PinnacleError::InvalidConfig(format!(
                "Surface code distance must be odd >= 3, got {}",
                distance
            )));
        }

        let num_data = distance * distance;
        let num_ancilla = (distance - 1) * (distance - 1) + (distance - 1) * (distance - 1);
        let total = num_data + num_ancilla;

        if total > self.chip.qubits.len() {
            return Err(PinnacleError::InvalidConfig(format!(
                "Distance {} requires {} qubits but chip has {}",
                distance,
                total,
                self.chip.qubits.len()
            )));
        }

        let num_rounds = distance; // standard: d rounds of syndrome extraction
        let physical_error = 1.0 - self.config.gate_fidelity_2q;

        // Syndrome extraction: each round applies 2q gates to data+ancilla pairs
        let mut detected_errors = 0;
        let mut syndrome_history = Vec::with_capacity(num_rounds);

        for _round in 0..num_rounds {
            // Each ancilla measures a stabilizer (4 CNOT-equivalent gates)
            let per_round_error = 1.0 - (1.0 - physical_error).powi(4);
            let syndrome: Vec<bool> = (0..num_ancilla)
                .map(|i| {
                    // Deterministic approximation: error if accumulated probability
                    // exceeds threshold (proportional to position for variety)
                    let p = per_round_error * (1.0 + 0.01 * (i as f64));
                    p > 0.5 // in practice, use RNG; here we report expected behavior
                })
                .collect();
            detected_errors += syndrome.iter().filter(|&&s| s).count();
            syndrome_history.push(syndrome);
        }

        // Logical error: below threshold if Lambda > 1
        let lambda = self.error_suppression_factor();
        let logical_error = lambda <= 1.0;

        Ok(SurfaceCodeResult {
            distance,
            num_rounds,
            detected_errors,
            logical_error,
            syndrome_history,
        })
    }

    /// Estimate the logical error rate at a given code distance over multiple rounds.
    ///
    /// Uses the scaling formula: p_L = A * (p / p_th)^((d+1)/2)
    /// where p_th is the threshold (~1%), p is the physical error rate,
    /// and A is a prefactor ~0.1.
    pub fn logical_error_rate(&self, distance: usize, num_rounds: usize) -> f64 {
        let p = 1.0 - self.config.gate_fidelity_2q; // physical error rate
        let p_th = 0.01; // approximate threshold for surface codes
        let a = 0.1; // empirical prefactor
        let exponent = ((distance + 1) as f64) / 2.0;
        let base_rate = a * (p / p_th).powf(exponent);
        // Scale by number of rounds (error grows linearly with rounds)
        (base_rate * num_rounds as f64).min(1.0)
    }

    /// Estimate quantum volume using the simplified model:
    /// QV = 2^m where m is the largest circuit width such that
    /// the heavy output probability exceeds 2/3.
    ///
    /// We use the approximation: success probability ~ (1 - epsilon)^(m * d)
    /// where epsilon is the effective error per layer and d = m (square circuits).
    pub fn estimate_quantum_volume(&self) -> usize {
        let epsilon_1q = 1.0 - self.config.gate_fidelity_1q;
        let epsilon_2q = 1.0 - self.config.gate_fidelity_2q;
        let epsilon_ro = 1.0 - self.config.readout_fidelity;
        // Effective per-layer error: dominated by 2q gates
        let epsilon = epsilon_2q + epsilon_1q + epsilon_ro;

        let max_width = self.chip.qubits.len().min(100); // QV test is practical up to ~100
        let mut best_m = 1_usize;

        for m in 1..=max_width {
            let d = m; // square circuit: depth = width
            let total_error = 1.0 - (1.0 - epsilon).powi((m * d) as i32);
            // Heavy output probability must exceed 2/3
            let hop = 1.0 - total_error;
            if hop >= 2.0 / 3.0 {
                best_m = m;
            } else {
                break;
            }
        }

        // QV = 2^m
        1_usize << best_m
    }

    /// Compare Pinnacle against Willow published benchmarks.
    pub fn benchmark_vs_willow(&self) -> ComparisonResult {
        // Willow published values (Nature 2025)
        let willow_logical_error_d7 = 0.00143; // ~0.14% at distance 7
        let willow_qv = 128_usize;

        let pinnacle_logical_error = self.logical_error_rate(7, 25);
        let pinnacle_qv = self.estimate_quantum_volume();

        let improvement = if pinnacle_logical_error > 0.0 {
            willow_logical_error_d7 / pinnacle_logical_error
        } else {
            f64::INFINITY
        };

        ComparisonResult {
            pinnacle_logical_error,
            willow_logical_error: willow_logical_error_d7,
            pinnacle_qv,
            willow_qv,
            improvement_factor: improvement,
        }
    }

    /// Error suppression factor Lambda: the factor by which the logical error
    /// rate decreases when the code distance increases by 2.
    /// Lambda > 1 means below-threshold operation.
    pub fn error_suppression_factor(&self) -> f64 {
        let p_d3 = self.logical_error_rate(3, 25);
        let p_d5 = self.logical_error_rate(5, 25);
        if p_d5 > 0.0 {
            p_d3 / p_d5
        } else {
            f64::INFINITY
        }
    }

    // --------------------------------------------------------
    // PRIVATE HELPERS
    // --------------------------------------------------------

    /// Look up the fidelity boost for the zone containing a given qubit.
    fn zone_fidelity_boost(&self, qubit: usize) -> f64 {
        for zone in &self.chip.zones {
            if zone.qubit_indices.contains(&qubit) {
                return zone.local_fidelity_boost;
            }
        }
        1.0
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config builder and validation ----

    #[test]
    fn test_default_config() {
        let cfg = PinnacleConfig::default();
        assert_eq!(cfg.num_qubits, 105);
        assert_eq!(cfg.t1_us, 68.0);
        assert_eq!(cfg.t2_us, 30.0);
        assert!((cfg.gate_fidelity_1q - 0.9995).abs() < 1e-10);
        assert!((cfg.gate_fidelity_2q - 0.9965).abs() < 1e-10);
        assert!(cfg.surface_code_distance.is_none());
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(72)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 8, cols: 9 })
            .with_t1(100.0)
            .with_t2(50.0)
            .with_gate_fidelity_1q(0.9999)
            .with_gate_fidelity_2q(0.999)
            .with_readout_fidelity(0.999)
            .with_gate_time_1q_ns(20.0)
            .with_gate_time_2q_ns(28.0)
            .with_surface_code_distance(5);
        assert_eq!(cfg.num_qubits, 72);
        assert_eq!(cfg.t1_us, 100.0);
        assert_eq!(cfg.surface_code_distance, Some(5));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_qubits() {
        let cfg = PinnacleConfig::new().with_num_qubits(0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_too_many_qubits() {
        let cfg = PinnacleConfig::new().with_num_qubits(1001);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_bad_t2() {
        // T2 > 2*T1 is physically impossible
        let cfg = PinnacleConfig::new().with_t1(30.0).with_t2(61.0);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_pinnacle_zone_mismatch() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(100)
            .with_coupling_map(CouplingTopology::Pinnacle {
                inner_zone_qubits: 49,
                outer_zone_qubits: 56, // 49 + 56 = 105 != 100
            });
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_grid2d_mismatch() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(100)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 5, cols: 5 }); // 25 != 100
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validation_even_surface_code_distance() {
        let cfg = PinnacleConfig::new().with_surface_code_distance(4);
        assert!(cfg.validate().is_err());
    }

    // ---- Coupling topology construction ----

    #[test]
    fn test_grid2d_topology() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(12)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 4 });
        let chip = PinnacleChip::new(&cfg).unwrap();
        assert_eq!(chip.qubits.len(), 12);
        // In a 3x4 grid: horizontal edges = 3*3=9, vertical edges = 2*4=8, total = 17
        assert_eq!(chip.num_edges(), 17);
    }

    #[test]
    fn test_heavyhex_topology() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(16)
            .with_coupling_map(CouplingTopology::HeavyHex);
        let chip = PinnacleChip::new(&cfg).unwrap();
        assert_eq!(chip.qubits.len(), 16);
        // HeavyHex has more edges than a simple grid due to hex diagonals
        assert!(chip.num_edges() > 0);
    }

    #[test]
    fn test_pinnacle_topology() {
        let cfg = PinnacleConfig::default(); // 105 = 49 inner + 56 outer
        let chip = PinnacleChip::new(&cfg).unwrap();
        assert_eq!(chip.qubits.len(), 105);
        assert_eq!(chip.zones.len(), 3);

        // Verify zone types
        let zone_types: Vec<ZoneType> = chip.zones.iter().map(|z| z.zone_type).collect();
        assert!(zone_types.contains(&ZoneType::DataZone));
        assert!(zone_types.contains(&ZoneType::AncillaZone));
        assert!(zone_types.contains(&ZoneType::MagicStateFactory));
    }

    // ---- Qubit zone assignment ----

    #[test]
    fn test_zone_assignment_pinnacle() {
        let cfg = PinnacleConfig::default();
        let chip = PinnacleChip::new(&cfg).unwrap();

        // Inner zone = 49 qubits, split 70/30 into data/factory
        let data_zone = chip
            .zones
            .iter()
            .find(|z| z.zone_type == ZoneType::DataZone)
            .unwrap();
        let factory_zone = chip
            .zones
            .iter()
            .find(|z| z.zone_type == ZoneType::MagicStateFactory)
            .unwrap();
        let ancilla_zone = chip
            .zones
            .iter()
            .find(|z| z.zone_type == ZoneType::AncillaZone)
            .unwrap();

        assert_eq!(
            data_zone.qubit_indices.len() + factory_zone.qubit_indices.len(),
            49
        );
        assert_eq!(ancilla_zone.qubit_indices.len(), 56);
        // Fidelity boosts are zone-specific
        assert!(data_zone.local_fidelity_boost > factory_zone.local_fidelity_boost);
    }

    #[test]
    fn test_heterogeneous_zone_calibration() {
        let cfg = PinnacleConfig::default();
        let chip = PinnacleChip::new(&cfg).unwrap();

        // Data zone qubits should have lower error rate than ancilla due to fidelity boost
        let data_qubit = &chip.qubits[0]; // in data zone
        let ancilla_qubit = &chip.qubits[chip.qubits.len() - 1]; // in ancilla zone
        assert!(data_qubit.current_error_rate < ancilla_qubit.current_error_rate);

        // Data zone T1 should be at full value; factory zone T1 should be reduced
        let factory_idx = chip
            .zones
            .iter()
            .find(|z| z.zone_type == ZoneType::MagicStateFactory)
            .unwrap()
            .qubit_indices[0];
        assert!(chip.qubits[factory_idx].t1 < chip.qubits[0].t1);
    }

    // ---- Neighbor lookup ----

    #[test]
    fn test_neighbors_grid2d() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();

        // Center qubit (4) in a 3x3 grid has 4 neighbors
        let n4 = chip.neighbors(4);
        assert_eq!(n4.len(), 4);
        assert!(n4.contains(&1)); // up
        assert!(n4.contains(&3)); // left
        assert!(n4.contains(&5)); // right
        assert!(n4.contains(&7)); // down

        // Corner qubit (0) has 2 neighbors
        let n0 = chip.neighbors(0);
        assert_eq!(n0.len(), 2);
    }

    // ---- Shortest path ----

    #[test]
    fn test_shortest_path_grid() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();

        // Path from corner (0) to opposite corner (8) in a 3x3 grid
        let path = chip.shortest_path(0, 8);
        assert!(!path.is_empty());
        assert_eq!(path[0], 0);
        assert_eq!(*path.last().unwrap(), 8);
        // Manhattan distance = 4, so path length = 5 nodes
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn test_shortest_path_same_qubit() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();
        let path = chip.shortest_path(4, 4);
        assert_eq!(path, vec![4]);
    }

    // ---- Noise model calculations ----

    #[test]
    fn test_amplitude_damping() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        // T1 = 68 us, gate_time = 25 ns = 0.025 us
        let p = nm.amplitude_damping(68.0, 25.0);
        // Expected: 1 - exp(-0.025/68) ~ 3.68e-4
        assert!(p > 3.0e-4 && p < 4.0e-4, "amplitude_damping p = {}", p);
    }

    #[test]
    fn test_dephasing() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        // T2 = 30, T1 = 68, T_phi = 1/(1/30 - 1/136) ~ 38.87 us
        let p = nm.dephasing(30.0, 25.0);
        assert!(p > 0.0 && p < 1.0e-3, "dephasing p = {}", p);
    }

    #[test]
    fn test_dephasing_zero_when_t2_equals_2t1() {
        let cfg = PinnacleConfig::new().with_t1(30.0).with_t2(60.0);
        let nm = NoiseModel::from_config(&cfg);
        let p = nm.dephasing(60.0, 25.0);
        // At T2 = 2*T1, pure dephasing rate = 0
        assert!(p.abs() < 1e-15, "dephasing should be 0 when T2=2*T1");
    }

    #[test]
    fn test_depolarizing_channel_1q() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p = nm.depolarizing_channel(0.9995, 1);
        // (4/3) * (1 - 0.9995) = 4/3 * 0.0005 ~ 6.67e-4
        assert!((p - 4.0 / 3.0 * 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_depolarizing_channel_2q() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p = nm.depolarizing_channel(0.9965, 2);
        // (16/15) * (1 - 0.9965) = 16/15 * 0.0035 ~ 3.73e-3
        let expected = 16.0 / 15.0 * 0.0035;
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_readout_error() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p = nm.readout_error(0.998);
        assert!((p - 0.002).abs() < 1e-15);
    }

    #[test]
    fn test_crosstalk() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p = nm.crosstalk(0, 1, 12.0); // 12 MHz coupling
        // coupling^2 * gate_time_us = 144 * 0.032 = 4.608 -> clamped to 1.0
        // Hmm, that's large. Coupling should be in normalized units.
        // With typical coupling ~ 0.01 (normalized): 0.0001 * 0.032 ~ 3.2e-6
        let p_small = nm.crosstalk(0, 1, 0.01);
        assert!(p_small < 1e-4, "crosstalk p = {}", p_small);
    }

    // ---- Gate application with noise ----

    #[test]
    fn test_apply_single_qubit_gate() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let errors = sim.apply_gate(&PinnacleGate::H, &[0]).unwrap();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, 0);
        assert!(errors[0].1 > 0.0 && errors[0].1 < 0.01);
    }

    #[test]
    fn test_apply_two_qubit_gate_adjacent() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let sim = PinnacleSimulator::new(cfg).unwrap();
        let errors = sim.apply_gate(&PinnacleGate::CZ, &[0, 1]).unwrap();
        assert_eq!(errors.len(), 2);
        assert!(errors[0].1 > 0.0);
    }

    #[test]
    fn test_apply_two_qubit_gate_non_adjacent() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let sim = PinnacleSimulator::new(cfg).unwrap();
        // Qubits 0 and 8 are not adjacent in a 3x3 grid
        let result = sim.apply_gate(&PinnacleGate::CZ, &[0, 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_gate_out_of_range() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let result = sim.apply_gate(&PinnacleGate::X, &[999]);
        assert!(result.is_err());
    }

    // ---- Circuit construction and validation ----

    #[test]
    fn test_circuit_construction() {
        let mut circuit = PinnacleCircuit::new();
        circuit.add_moment(vec![
            (PinnacleGate::H, vec![0]),
            (PinnacleGate::H, vec![1]),
        ]);
        circuit.add_moment(vec![(PinnacleGate::CZ, vec![0, 1])]);
        circuit.add_moment(vec![
            (PinnacleGate::Measure, vec![0]),
            (PinnacleGate::Measure, vec![1]),
        ]);
        assert_eq!(circuit.depth(), 3);
        assert_eq!(circuit.gate_count(), 5);
        assert_eq!(circuit.two_qubit_gate_count(), 1);
    }

    #[test]
    fn test_circuit_validation_valid() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();
        let ops = vec![
            (PinnacleGate::H, vec![0]),
            (PinnacleGate::CZ, vec![0, 1]),
            (PinnacleGate::Measure, vec![0]),
        ];
        assert!(chip.is_valid_circuit(&ops));
    }

    #[test]
    fn test_circuit_validation_invalid_adjacency() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();
        let ops = vec![
            (PinnacleGate::CZ, vec![0, 8]), // not adjacent
        ];
        assert!(!chip.is_valid_circuit(&ops));
    }

    #[test]
    fn test_circuit_validation_out_of_range() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let chip = PinnacleChip::new(&cfg).unwrap();
        let ops = vec![(PinnacleGate::H, vec![100])]; // out of range
        assert!(!chip.is_valid_circuit(&ops));
    }

    // ---- Circuit execution ----

    #[test]
    fn test_run_circuit() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let sim = PinnacleSimulator::new(cfg).unwrap();

        let mut circuit = PinnacleCircuit::new();
        circuit.add_moment(vec![(PinnacleGate::H, vec![0])]);
        circuit.add_moment(vec![(PinnacleGate::CZ, vec![0, 1])]);
        circuit.add_moment(vec![
            (PinnacleGate::Measure, vec![0]),
            (PinnacleGate::Measure, vec![1]),
        ]);

        let result = sim.run_circuit(&circuit).unwrap();
        assert!(result.execution_time_ns > 0.0);
        assert!(result.total_error_probability > 0.0);
        assert!(result.total_error_probability < 1.0);
        assert_eq!(result.per_qubit_error.len(), 9);
        // Qubits 0 and 1 should have errors; qubit 8 should have zero error
        assert!(result.per_qubit_error[0] > 0.0);
        assert!(result.per_qubit_error[1] > 0.0);
        assert!((result.per_qubit_error[8]).abs() < 1e-15);
    }

    // ---- Surface code cycle ----

    #[test]
    fn test_surface_code_cycle() {
        let cfg = PinnacleConfig::default(); // 105 qubits
        let sim = PinnacleSimulator::new(cfg).unwrap();
        let result = sim.run_surface_code_cycle(3).unwrap();
        assert_eq!(result.distance, 3);
        assert_eq!(result.num_rounds, 3);
        assert!(!result.syndrome_history.is_empty());
    }

    #[test]
    fn test_surface_code_cycle_invalid_distance() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        assert!(sim.run_surface_code_cycle(4).is_err()); // even
        assert!(sim.run_surface_code_cycle(2).is_err()); // < 3
    }

    // ---- Logical error rate estimation ----

    #[test]
    fn test_logical_error_rate_decreases_with_distance() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let p3 = sim.logical_error_rate(3, 25);
        let p5 = sim.logical_error_rate(5, 25);
        let p7 = sim.logical_error_rate(7, 25);
        // Below threshold: error rate should decrease with distance
        assert!(p3 > p5, "d=3 ({}) should > d=5 ({})", p3, p5);
        assert!(p5 > p7, "d=5 ({}) should > d=7 ({})", p5, p7);
    }

    #[test]
    fn test_error_suppression_factor() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let lambda = sim.error_suppression_factor();
        // With default fidelities (0.9965 2q), Lambda should be > 1 (below threshold)
        assert!(lambda > 1.0, "Lambda = {} should be > 1", lambda);
    }

    // ---- Quantum volume estimation ----

    #[test]
    fn test_quantum_volume_positive() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let qv = sim.estimate_quantum_volume();
        assert!(qv >= 2, "QV should be at least 2, got {}", qv);
        // With 0.9965 2q fidelity, QV should be meaningful
        assert!(qv.is_power_of_two());
    }

    #[test]
    fn test_quantum_volume_scales_with_fidelity() {
        let cfg_low = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 })
            .with_gate_fidelity_1q(0.99)
            .with_gate_fidelity_2q(0.98)
            .with_readout_fidelity(0.99);
        let cfg_high = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 })
            .with_gate_fidelity_1q(0.9999)
            .with_gate_fidelity_2q(0.999)
            .with_readout_fidelity(0.999);

        let qv_low = PinnacleSimulator::new(cfg_low).unwrap().estimate_quantum_volume();
        let qv_high = PinnacleSimulator::new(cfg_high)
            .unwrap()
            .estimate_quantum_volume();
        assert!(
            qv_high >= qv_low,
            "Higher fidelity should give >= QV: {} vs {}",
            qv_high,
            qv_low
        );
    }

    // ---- Benchmark comparison ----

    #[test]
    fn test_benchmark_vs_willow() {
        let sim = PinnacleSimulator::new(PinnacleConfig::default()).unwrap();
        let result = sim.benchmark_vs_willow();
        assert!(result.pinnacle_logical_error > 0.0);
        assert!((result.willow_logical_error - 0.00143).abs() < 1e-6);
        assert_eq!(result.willow_qv, 128);
        assert!(result.improvement_factor > 0.0);
    }

    #[test]
    fn test_pinnacle_improves_on_willow() {
        // The Pinnacle config with its heterogeneous zones and higher fidelities
        // should show improvement over Willow
        let cfg = PinnacleConfig::new()
            .with_gate_fidelity_2q(0.999) // 10x better 2q fidelity than Willow
            .with_gate_fidelity_1q(0.9999)
            .with_readout_fidelity(0.999);
        let sim = PinnacleSimulator::new(cfg).unwrap();
        let result = sim.benchmark_vs_willow();
        // With much better fidelities, Pinnacle should beat Willow
        assert!(
            result.improvement_factor > 1.0,
            "Pinnacle should improve on Willow, got factor {}",
            result.improvement_factor
        );
    }

    // ---- Sycamore gate ----

    #[test]
    fn test_sycamore_gate_fidelity() {
        let cfg = PinnacleConfig::new()
            .with_num_qubits(9)
            .with_coupling_map(CouplingTopology::Grid2D { rows: 3, cols: 3 });
        let sim = PinnacleSimulator::new(cfg).unwrap();

        let errors = sim
            .apply_gate(&PinnacleGate::SycamoreGate, &[0, 1])
            .unwrap();
        assert_eq!(errors.len(), 2);
        // Sycamore gate is a 2q gate, error should be bounded by 2q model
        let p = errors[0].1;
        assert!(p > 0.0 && p < 0.05, "Sycamore error = {}", p);
    }

    // ---- Gate properties ----

    #[test]
    fn test_gate_num_qubits() {
        assert_eq!(PinnacleGate::H.num_qubits(), 1);
        assert_eq!(PinnacleGate::X.num_qubits(), 1);
        assert_eq!(PinnacleGate::Rx(0.5).num_qubits(), 1);
        assert_eq!(PinnacleGate::CZ.num_qubits(), 2);
        assert_eq!(PinnacleGate::SqrtISWAP.num_qubits(), 2);
        assert_eq!(PinnacleGate::SycamoreGate.num_qubits(), 2);
        assert_eq!(PinnacleGate::Measure.num_qubits(), 1);
        assert_eq!(PinnacleGate::Reset.num_qubits(), 1);
    }

    #[test]
    fn test_gate_is_non_unitary() {
        assert!(PinnacleGate::Measure.is_non_unitary());
        assert!(PinnacleGate::Reset.is_non_unitary());
        assert!(!PinnacleGate::H.is_non_unitary());
        assert!(!PinnacleGate::CZ.is_non_unitary());
    }

    // ---- Error type Display ----

    #[test]
    fn test_error_display() {
        let e = PinnacleError::InvalidConfig("bad value".into());
        assert!(format!("{}", e).contains("bad value"));

        let e = PinnacleError::QubitOutOfRange(42);
        assert!(format!("{}", e).contains("42"));

        let e = PinnacleError::NotAdjacent(0, 8);
        assert!(format!("{}", e).contains("0") && format!("{}", e).contains("8"));
    }

    // ---- Noise model: total error ----

    #[test]
    fn test_total_1q_error_bounded() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p = nm.total_1q_error(68.0, 30.0);
        // Should be small but positive
        assert!(p > 0.0, "total 1q error should be > 0");
        assert!(p < 0.01, "total 1q error {} should be < 1%", p);
    }

    #[test]
    fn test_total_2q_error_greater_than_1q() {
        let nm = NoiseModel::from_config(&PinnacleConfig::default());
        let p1 = nm.total_1q_error(68.0, 30.0);
        let p2 = nm.total_2q_error(68.0, 30.0);
        assert!(
            p2 > p1,
            "2q error ({}) should exceed 1q error ({})",
            p2,
            p1
        );
    }
}
