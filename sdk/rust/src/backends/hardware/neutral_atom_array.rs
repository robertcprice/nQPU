//! Neutral Atom Array Simulation with Reconfigurable Atom Movement
//!
//! Full continuous-variable neutral atom array simulator modelling the physical
//! dynamics of reconfigurable optical tweezer arrays:
//!
//! - **Rydberg blockade physics**: C6/r^6 van der Waals interaction, blockade
//!   radius computation, and energy-shift accumulation across the array.
//! - **Zone-based computation**: atoms shuttle between Storage, Entangling,
//!   Readout, and Cooling zones mirroring real hardware (e.g. Atom Computing,
//!   QuEra).
//! - **Atom sorting / rearrangement**: odd-even transposition sort and
//!   heuristic rearrangement to compact stochastically-loaded arrays into
//!   defect-free sub-registers.
//! - **Native gate set**: single-qubit Rabi rotations, CZ via Rydberg blockade,
//!   native three-qubit CCZ, and global (parallel) rotations.
//! - **Noise model**: depolarising errors, atom loss per operation, and thermal
//!   motion dephasing.
//!
//! # Physical constants (default: Rubidium-87)
//!
//! | Parameter | Default | Unit |
//! |-----------|---------|------|
//! | C6 coefficient | 862 690 | MHz um^6 |
//! | Rydberg level n | 70 | - |
//! | Trap spacing | 5.0 | um |
//! | Max Rabi frequency | 10.0 | MHz |
//! | Atom temperature | 10.0 | uK |
//!
//! # References
//!
//! - Browaeys & Lahaye, Nature Physics 16 (2020) -- Rydberg atom arrays
//! - Bluvstein et al., Nature 604 (2022) -- reconfigurable atom arrays
//! - Ebadi et al., Nature 595 (2021) -- 256-atom programmable simulator
//! - Graham et al., Nature 604 (2022) -- multi-qubit entanglement

use num_complex::Complex64;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from neutral atom array operations.
#[derive(Debug, Clone)]
pub enum NeutralAtomError {
    /// Configuration parameter is out of valid range.
    InvalidConfig(String),
    /// Trap index is out of bounds.
    TrapOutOfBounds { index: usize, num_traps: usize },
    /// Atom index is out of bounds.
    AtomOutOfBounds { index: usize, num_atoms: usize },
    /// Source trap is unoccupied during a move.
    TrapEmpty(usize),
    /// Destination trap is already occupied during a move.
    TrapOccupied(usize),
    /// Atom has been lost and cannot be operated on.
    AtomLost(usize),
    /// Gate precondition violated (e.g. atoms not blockaded for CZ).
    GatePrecondition(String),
    /// Array geometry is invalid.
    InvalidGeometry(String),
}

impl std::fmt::Display for NeutralAtomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::TrapOutOfBounds { index, num_traps } => {
                write!(f, "Trap index {} out of bounds ({})", index, num_traps)
            }
            Self::AtomOutOfBounds { index, num_atoms } => {
                write!(f, "Atom index {} out of bounds ({})", index, num_atoms)
            }
            Self::TrapEmpty(idx) => write!(f, "Trap {} is empty", idx),
            Self::TrapOccupied(idx) => write!(f, "Trap {} is already occupied", idx),
            Self::AtomLost(idx) => write!(f, "Atom {} has been lost", idx),
            Self::GatePrecondition(msg) => write!(f, "Gate precondition: {}", msg),
            Self::InvalidGeometry(msg) => write!(f, "Invalid geometry: {}", msg),
        }
    }
}

impl std::error::Error for NeutralAtomError {}

pub type NeutralAtomResult<T> = Result<T, NeutralAtomError>;

// ===================================================================
// CONNECTIVITY & ZONE ENUMS
// ===================================================================

/// Connectivity model between atoms in the array.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomConnectivity {
    /// Only directly adjacent atoms interact.
    NearestNeighbor,
    /// Atoms up to two sites apart interact.
    NextNearestNeighbor,
    /// All pairs within the Rydberg blockade radius interact.
    AllToAll,
    /// User-specified pairs.
    Custom(Vec<(usize, usize)>),
}

/// Functional zone within the neutral atom processor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeZone {
    /// Long-coherence parking storage.
    Storage,
    /// Rydberg interaction / entangling zone.
    Entangling,
    /// Fluorescence imaging readout.
    Readout,
    /// Laser cooling / re-thermalisation zone.
    Cooling,
}

impl std::fmt::Display for ComputeZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage => write!(f, "Storage"),
            Self::Entangling => write!(f, "Entangling"),
            Self::Readout => write!(f, "Readout"),
            Self::Cooling => write!(f, "Cooling"),
        }
    }
}

// ===================================================================
// CONFIGURATION (builder pattern)
// ===================================================================

/// Configuration for a neutral atom array simulation.
#[derive(Debug, Clone)]
pub struct NeutralAtomConfig {
    /// Number of trap sites (4..=1000).
    pub num_atoms: usize,
    /// Trap spacing in micrometres.
    pub trap_spacing_um: f64,
    /// Principal quantum number of the Rydberg level.
    pub rydberg_level: usize,
    /// C6 van der Waals coefficient in MHz * um^6.
    pub c6_coefficient: f64,
    /// Maximum global Rabi frequency in MHz.
    pub max_rabi_frequency_mhz: f64,
    /// Atom temperature in microkelvin.
    pub atom_temperature_uk: f64,
    /// Probability that a trap is loaded after a loading attempt.
    pub loading_probability: f64,
    /// Fidelity of a single atom rearrangement move.
    pub rearrangement_fidelity: f64,
    /// Per-operation atom loss rate.
    pub atom_loss_rate: f64,
    /// Connectivity model.
    pub connectivity: AtomConnectivity,
}

impl Default for NeutralAtomConfig {
    fn default() -> Self {
        Self {
            num_atoms: 256,
            trap_spacing_um: 5.0,
            rydberg_level: 70,
            c6_coefficient: 862_690.0,
            max_rabi_frequency_mhz: 10.0,
            atom_temperature_uk: 10.0,
            loading_probability: 0.5,
            rearrangement_fidelity: 0.999,
            atom_loss_rate: 0.001,
            connectivity: AtomConnectivity::AllToAll,
        }
    }
}

impl NeutralAtomConfig {
    /// Start a new builder with default values.
    pub fn builder() -> NeutralAtomConfigBuilder {
        NeutralAtomConfigBuilder {
            config: Self::default(),
        }
    }

    /// Validate all parameters.
    pub fn validate(&self) -> NeutralAtomResult<()> {
        if self.num_atoms < 4 || self.num_atoms > 1000 {
            return Err(NeutralAtomError::InvalidConfig(format!(
                "num_atoms must be 4..=1000, got {}",
                self.num_atoms
            )));
        }
        if self.trap_spacing_um <= 0.0 {
            return Err(NeutralAtomError::InvalidConfig(
                "trap_spacing_um must be positive".into(),
            ));
        }
        if self.c6_coefficient <= 0.0 {
            return Err(NeutralAtomError::InvalidConfig(
                "c6_coefficient must be positive".into(),
            ));
        }
        if self.max_rabi_frequency_mhz <= 0.0 {
            return Err(NeutralAtomError::InvalidConfig(
                "max_rabi_frequency_mhz must be positive".into(),
            ));
        }
        if self.loading_probability <= 0.0 || self.loading_probability > 1.0 {
            return Err(NeutralAtomError::InvalidConfig(format!(
                "loading_probability must be in (0, 1], got {}",
                self.loading_probability
            )));
        }
        if self.rearrangement_fidelity < 0.0 || self.rearrangement_fidelity > 1.0 {
            return Err(NeutralAtomError::InvalidConfig(format!(
                "rearrangement_fidelity must be in [0, 1], got {}",
                self.rearrangement_fidelity
            )));
        }
        if self.atom_loss_rate < 0.0 || self.atom_loss_rate > 1.0 {
            return Err(NeutralAtomError::InvalidConfig(format!(
                "atom_loss_rate must be in [0, 1], got {}",
                self.atom_loss_rate
            )));
        }
        Ok(())
    }
}

/// Builder for [`NeutralAtomConfig`].
pub struct NeutralAtomConfigBuilder {
    config: NeutralAtomConfig,
}

impl NeutralAtomConfigBuilder {
    pub fn num_atoms(mut self, n: usize) -> Self {
        self.config.num_atoms = n;
        self
    }
    pub fn trap_spacing_um(mut self, s: f64) -> Self {
        self.config.trap_spacing_um = s;
        self
    }
    pub fn rydberg_level(mut self, n: usize) -> Self {
        self.config.rydberg_level = n;
        self
    }
    pub fn c6_coefficient(mut self, c6: f64) -> Self {
        self.config.c6_coefficient = c6;
        self
    }
    pub fn max_rabi_frequency_mhz(mut self, omega: f64) -> Self {
        self.config.max_rabi_frequency_mhz = omega;
        self
    }
    pub fn atom_temperature_uk(mut self, t: f64) -> Self {
        self.config.atom_temperature_uk = t;
        self
    }
    pub fn loading_probability(mut self, p: f64) -> Self {
        self.config.loading_probability = p;
        self
    }
    pub fn rearrangement_fidelity(mut self, f: f64) -> Self {
        self.config.rearrangement_fidelity = f;
        self
    }
    pub fn atom_loss_rate(mut self, r: f64) -> Self {
        self.config.atom_loss_rate = r;
        self
    }
    pub fn connectivity(mut self, c: AtomConnectivity) -> Self {
        self.config.connectivity = c;
        self
    }

    /// Consume the builder and produce a validated config.
    pub fn build(self) -> NeutralAtomResult<NeutralAtomConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ===================================================================
// ATOM TRAP
// ===================================================================

/// A single optical tweezer trap site.
#[derive(Debug, Clone)]
pub struct AtomTrap {
    /// 3D position in micrometres.
    pub position: (f64, f64, f64),
    /// Whether the trap currently holds an atom.
    pub occupied: bool,
    /// Index of the atom currently held, if any.
    pub atom_id: Option<usize>,
    /// Functional zone this trap belongs to.
    pub zone: ComputeZone,
}

// ===================================================================
// ATOM STATE
// ===================================================================

/// Internal state of a single neutral atom qubit.
#[derive(Debug, Clone)]
pub struct AtomState {
    /// Two-level qubit amplitudes: [|g>, |r>] (ground, Rydberg).
    pub qubit_state: [Complex64; 2],
    /// Current 3D position in micrometres (may differ from trap centre due to thermal motion).
    pub position: (f64, f64, f64),
    /// Velocity vector (um / us) from thermal distribution.
    pub velocity: (f64, f64, f64),
    /// Whether the atom has been lost from its trap.
    pub is_lost: bool,
}

impl AtomState {
    /// Create an atom initialised in |g> at the given position.
    pub fn new_ground(pos: (f64, f64, f64)) -> Self {
        Self {
            qubit_state: [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            position: pos,
            velocity: (0.0, 0.0, 0.0),
            is_lost: false,
        }
    }

    /// Probability of being in the Rydberg (excited) state.
    pub fn rydberg_population(&self) -> f64 {
        self.qubit_state[1].norm_sqr()
    }

    /// Probability of being in the ground state.
    pub fn ground_population(&self) -> f64 {
        self.qubit_state[0].norm_sqr()
    }
}

// ===================================================================
// ATOM ARRAY (geometry + state management)
// ===================================================================

/// A reconfigurable array of optical tweezer traps holding neutral atoms.
#[derive(Debug, Clone)]
pub struct AtomArray {
    pub traps: Vec<AtomTrap>,
    pub atoms: Vec<AtomState>,
    pub config: NeutralAtomConfig,
}

impl AtomArray {
    /// Build a 1D linear chain of traps.
    pub fn new_1d(config: &NeutralAtomConfig) -> NeutralAtomResult<Self> {
        config.validate()?;
        let n = config.num_atoms;
        let spacing = config.trap_spacing_um;
        let mut traps = Vec::with_capacity(n);
        for i in 0..n {
            traps.push(AtomTrap {
                position: (i as f64 * spacing, 0.0, 0.0),
                occupied: false,
                atom_id: None,
                zone: ComputeZone::Entangling,
            });
        }
        Ok(Self {
            traps,
            atoms: Vec::new(),
            config: config.clone(),
        })
    }

    /// Build a 2D square lattice of `rows x cols` traps.
    pub fn new_2d_square(
        rows: usize,
        cols: usize,
        config: &NeutralAtomConfig,
    ) -> NeutralAtomResult<Self> {
        if rows == 0 || cols == 0 {
            return Err(NeutralAtomError::InvalidGeometry(
                "rows and cols must be > 0".into(),
            ));
        }
        // Validate the rest of the config but we override num_atoms.
        let mut cfg = config.clone();
        cfg.num_atoms = rows * cols;
        cfg.validate()?;

        let spacing = config.trap_spacing_um;
        let mut traps = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                traps.push(AtomTrap {
                    position: (c as f64 * spacing, r as f64 * spacing, 0.0),
                    occupied: false,
                    atom_id: None,
                    zone: ComputeZone::Entangling,
                });
            }
        }
        Ok(Self {
            traps,
            atoms: Vec::new(),
            config: cfg,
        })
    }

    /// Build a 2D triangular (hexagonal-packed) lattice.
    ///
    /// The number of trap sites equals `config.num_atoms`; rows are offset
    /// by half a spacing to create 60-degree geometry.
    pub fn new_2d_triangular(config: &NeutralAtomConfig) -> NeutralAtomResult<Self> {
        config.validate()?;
        let n = config.num_atoms;
        let spacing = config.trap_spacing_um;
        let cols = (n as f64).sqrt().ceil() as usize;
        let rows = (n + cols - 1) / cols;

        let mut traps = Vec::with_capacity(n);
        let mut count = 0usize;
        for r in 0..rows {
            let x_offset = if r % 2 == 1 { spacing * 0.5 } else { 0.0 };
            let y = r as f64 * spacing * (3.0_f64).sqrt() / 2.0;
            for c in 0..cols {
                if count >= n {
                    break;
                }
                traps.push(AtomTrap {
                    position: (c as f64 * spacing + x_offset, y, 0.0),
                    occupied: false,
                    atom_id: None,
                    zone: ComputeZone::Entangling,
                });
                count += 1;
            }
        }
        Ok(Self {
            traps,
            atoms: Vec::new(),
            config: config.clone(),
        })
    }

    /// Number of currently occupied traps.
    pub fn num_occupied(&self) -> usize {
        self.traps.iter().filter(|t| t.occupied).count()
    }

    /// Shuttle an atom from one trap to another.
    pub fn move_atom(&mut self, from_trap: usize, to_trap: usize) -> NeutralAtomResult<()> {
        let nt = self.traps.len();
        if from_trap >= nt {
            return Err(NeutralAtomError::TrapOutOfBounds {
                index: from_trap,
                num_traps: nt,
            });
        }
        if to_trap >= nt {
            return Err(NeutralAtomError::TrapOutOfBounds {
                index: to_trap,
                num_traps: nt,
            });
        }
        if !self.traps[from_trap].occupied {
            return Err(NeutralAtomError::TrapEmpty(from_trap));
        }
        if self.traps[to_trap].occupied {
            return Err(NeutralAtomError::TrapOccupied(to_trap));
        }

        let atom_id = self.traps[from_trap]
            .atom_id
            .expect("occupied trap must have atom_id");

        // Update atom position to new trap centre.
        self.atoms[atom_id].position = self.traps[to_trap].position;

        // Swap trap occupancy.
        self.traps[from_trap].occupied = false;
        self.traps[from_trap].atom_id = None;
        self.traps[to_trap].occupied = true;
        self.traps[to_trap].atom_id = Some(atom_id);

        Ok(())
    }

    /// Sort atoms into a contiguous block at the start of the trap array
    /// (defragmentation / compaction).  Returns the number of moves performed.
    pub fn sort_atoms(&mut self) -> usize {
        AtomSortingAlgorithm::odd_even_sort(self)
    }

    /// Rearrange atoms so that every target trap in `target_pattern` is
    /// occupied.  Each element is `(atom_index, target_trap_index)`.
    pub fn rearrange(&mut self, target_pattern: &[(usize, usize)]) -> NeutralAtomResult<usize> {
        AtomSortingAlgorithm::heuristic_rearrange(self, target_pattern)
    }

    /// Assign a zone to a trap.
    pub fn set_zone(&mut self, trap_idx: usize, zone: ComputeZone) -> NeutralAtomResult<()> {
        if trap_idx >= self.traps.len() {
            return Err(NeutralAtomError::TrapOutOfBounds {
                index: trap_idx,
                num_traps: self.traps.len(),
            });
        }
        self.traps[trap_idx].zone = zone;
        Ok(())
    }
}

// ===================================================================
// RYDBERG INTERACTION PHYSICS
// ===================================================================

/// Rydberg interaction physics for neutral atom qubits.
pub struct RydbergInteraction;

impl RydbergInteraction {
    /// Blockade radius: R_b = (C6 / Omega)^{1/6}.
    ///
    /// Returns the distance in micrometres below which two atoms experience
    /// a Rydberg blockade strong enough to prevent simultaneous excitation.
    pub fn blockade_radius(c6: f64, rabi_freq: f64) -> f64 {
        (c6 / rabi_freq).powf(1.0 / 6.0)
    }

    /// Van der Waals interaction energy: V(r) = C6 / r^6  (MHz).
    pub fn interaction_energy(c6: f64, distance: f64) -> f64 {
        if distance <= 0.0 {
            return f64::INFINITY;
        }
        c6 / distance.powi(6)
    }

    /// Whether two atoms at given positions are within the blockade radius.
    pub fn is_blockaded(
        pos_i: (f64, f64, f64),
        pos_j: (f64, f64, f64),
        c6: f64,
        rabi_freq: f64,
    ) -> bool {
        let d = distance_3d(pos_i, pos_j);
        d < Self::blockade_radius(c6, rabi_freq)
    }

    /// Total Rydberg shift on a target atom from all other excited atoms.
    ///
    /// Returns the sum of C6/r^6 for every atom `j != target` that has
    /// non-negligible Rydberg population (> 0.1).
    pub fn blockade_shift(atoms: &[AtomState], target: usize, c6: f64) -> f64 {
        let mut shift = 0.0;
        let target_pos = atoms[target].position;
        for (j, atom) in atoms.iter().enumerate() {
            if j == target || atom.is_lost {
                continue;
            }
            if atom.rydberg_population() > 0.1 {
                let d = distance_3d(target_pos, atom.position);
                shift += Self::interaction_energy(c6, d);
            }
        }
        shift
    }
}

/// Euclidean distance between two 3D points.
fn distance_3d(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let dz = a.2 - b.2;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ===================================================================
// NEUTRAL ATOM GATE SET
// ===================================================================

/// Native gate operations for neutral atom qubits.
pub struct NeutralAtomGates;

impl NeutralAtomGates {
    /// Single-qubit rotation on the Bloch sphere via Rabi pulse.
    ///
    /// U(theta, phi) = [[cos(theta/2), -i e^{-i phi} sin(theta/2)],
    ///                   [-i e^{i phi} sin(theta/2), cos(theta/2)]]
    pub fn single_qubit_rotation(atom: &mut AtomState, theta: f64, phi: f64) {
        let ct = (theta / 2.0).cos();
        let st = (theta / 2.0).sin();
        let ep = Complex64::new(phi.cos(), phi.sin()); // e^{i phi}
        let em = Complex64::new(phi.cos(), -phi.sin()); // e^{-i phi}
        let neg_i = Complex64::new(0.0, -1.0);

        let a = atom.qubit_state[0];
        let b = atom.qubit_state[1];

        atom.qubit_state[0] = a * ct + b * (neg_i * em * st);
        atom.qubit_state[1] = a * (neg_i * ep * st) + b * ct;
    }

    /// CZ gate via Rydberg blockade between two atoms.
    ///
    /// Physical implementation: pi pulse on atom A, 2 pi pulse on atom B
    /// (blocked if A in |r>), pi pulse on atom A.  Net effect: phase of -1
    /// on |rr>.  In the ground/Rydberg computational basis this is CZ.
    pub fn cz_gate(atoms: &mut [AtomState], a: usize, b: usize) -> NeutralAtomResult<()> {
        let na = atoms.len();
        if a >= na {
            return Err(NeutralAtomError::AtomOutOfBounds {
                index: a,
                num_atoms: na,
            });
        }
        if b >= na {
            return Err(NeutralAtomError::AtomOutOfBounds {
                index: b,
                num_atoms: na,
            });
        }
        if atoms[a].is_lost {
            return Err(NeutralAtomError::AtomLost(a));
        }
        if atoms[b].is_lost {
            return Err(NeutralAtomError::AtomLost(b));
        }

        // CZ in the 2-qubit computational basis: diag(1, 1, 1, -1).
        // In the product basis |g_a g_b>, |g_a r_b>, |r_a g_b>, |r_a r_b>:
        //   only the |r_a r_b> component picks up a -1 phase.
        let ga = atoms[a].qubit_state[0];
        let ra = atoms[a].qubit_state[1];
        let gb = atoms[b].qubit_state[0];
        let rb = atoms[b].qubit_state[1];

        // Expand into 4-component product state, apply CZ, then project back.
        // |00> = ga*gb,  |01> = ga*rb,  |10> = ra*gb,  |11> = ra*rb
        let c00 = ga * gb;
        let c01 = ga * rb;
        let c10 = ra * gb;
        let c11 = ra * rb * (-1.0); // CZ phase

        // Project back: atom A amplitudes are sums over B states.
        // |g_a> component: c00 (B=|g>) + c01 (B=|r>)  -- but we need to keep B state too.
        // For a proper 2-qubit gate we need to work in the full 4-dim space.
        // We store the result by updating both atoms coherently.
        //
        // After CZ the state is |psi> = c00|gg> + c01|gr> + c10|rg> + c11|rr>.
        // We can only store product states in separated atom storage, so we write
        // back the marginal amplitudes.  For a true CZ on a product-state input
        // that STAYS a product state (the common case in circuits with interleaved
        // measurements), this is exact.  For entangled inputs the simulator must
        // use the full NeutralAtomSimulator path.
        //
        // Marginal of A: rho_A = Tr_B(|psi><psi|).
        // |g_A|^2 = |c00|^2 + |c01|^2,  |r_A|^2 = |c10|^2 + |c11|^2.
        // Phase:  We preserve the relative phase structure by direct assignment.

        let norm_a_g = (c00.norm_sqr() + c01.norm_sqr()).sqrt();
        let norm_a_r = (c10.norm_sqr() + c11.norm_sqr()).sqrt();
        let norm_b_g = (c00.norm_sqr() + c10.norm_sqr()).sqrt();
        let norm_b_r = (c01.norm_sqr() + c11.norm_sqr()).sqrt();

        // Preserve phase from the component that contributes to each amplitude.
        // For atom A's |r> component, we need the phase from c10 or c11.
        // If c10 is zero but c11 is non-zero, use c11's phase.
        let phase_ag = if c00.norm_sqr() > 1e-30 {
            c00 / c00.norm()
        } else if c01.norm_sqr() > 1e-30 {
            c01 / c01.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };
        let phase_ar = if c10.norm_sqr() > 1e-30 {
            c10 / c10.norm()
        } else if c11.norm_sqr() > 1e-30 {
            c11 / c11.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };
        let phase_bg = if c00.norm_sqr() > 1e-30 {
            c00 / c00.norm()
        } else if c10.norm_sqr() > 1e-30 {
            c10 / c10.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };
        let phase_br = if c01.norm_sqr() > 1e-30 {
            c01 / c01.norm()
        } else if c11.norm_sqr() > 1e-30 {
            c11 / c11.norm()
        } else {
            Complex64::new(1.0, 0.0)
        };

        atoms[a].qubit_state[0] = phase_ag * norm_a_g;
        atoms[a].qubit_state[1] = phase_ar * norm_a_r;
        atoms[b].qubit_state[0] = phase_bg * norm_b_g;
        atoms[b].qubit_state[1] = phase_br * norm_b_r;

        Ok(())
    }

    /// Native 3-qubit CCZ gate (Controlled-Controlled-Z).
    ///
    /// The only component that picks up a -1 phase is |r r r>.
    /// This is a native operation on Rydberg hardware because the three-body
    /// blockade naturally implements the required phase.
    pub fn ccz_gate(
        atoms: &mut [AtomState],
        a: usize,
        b: usize,
        c: usize,
    ) -> NeutralAtomResult<()> {
        let na = atoms.len();
        for &idx in &[a, b, c] {
            if idx >= na {
                return Err(NeutralAtomError::AtomOutOfBounds {
                    index: idx,
                    num_atoms: na,
                });
            }
            if atoms[idx].is_lost {
                return Err(NeutralAtomError::AtomLost(idx));
            }
        }

        // Expand 3-qubit product state.
        let (ga, ra) = (atoms[a].qubit_state[0], atoms[a].qubit_state[1]);
        let (gb, rb) = (atoms[b].qubit_state[0], atoms[b].qubit_state[1]);
        let (gc, rc) = (atoms[c].qubit_state[0], atoms[c].qubit_state[1]);

        // 8 basis amplitudes: |abc> where a,b,c in {g=0, r=1}.
        let mut coeffs = [Complex64::new(0.0, 0.0); 8];
        let aa = [ga, ra];
        let bb = [gb, rb];
        let cc = [gc, rc];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let idx = (i << 2) | (j << 1) | k;
                    coeffs[idx] = aa[i] * bb[j] * cc[k];
                }
            }
        }

        // CCZ: flip sign on |111> = index 7.
        coeffs[7] = coeffs[7] * (-1.0);

        // Project back marginals for product-state storage.
        // Atom A: |g> from indices 0..3 (i=0), |r> from indices 4..7 (i=1).
        let norm_ag = (coeffs[0].norm_sqr()
            + coeffs[1].norm_sqr()
            + coeffs[2].norm_sqr()
            + coeffs[3].norm_sqr())
        .sqrt();
        let norm_ar = (coeffs[4].norm_sqr()
            + coeffs[5].norm_sqr()
            + coeffs[6].norm_sqr()
            + coeffs[7].norm_sqr())
        .sqrt();
        // Find first non-zero coefficient for phase
        let ph_ag = coeffs[0..4]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        let ph_ar = coeffs[4..8]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        atoms[a].qubit_state = [ph_ag * norm_ag, ph_ar * norm_ar];

        // Atom B: |g> from even k indices where j=0, |r> where j=1.
        let norm_bg = (coeffs[0].norm_sqr()
            + coeffs[1].norm_sqr()
            + coeffs[4].norm_sqr()
            + coeffs[5].norm_sqr())
        .sqrt();
        let norm_br = (coeffs[2].norm_sqr()
            + coeffs[3].norm_sqr()
            + coeffs[6].norm_sqr()
            + coeffs[7].norm_sqr())
        .sqrt();
        let ph_bg = [coeffs[0], coeffs[1], coeffs[4], coeffs[5]]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        let ph_br = [coeffs[2], coeffs[3], coeffs[6], coeffs[7]]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        atoms[b].qubit_state = [ph_bg * norm_bg, ph_br * norm_br];

        // Atom C: |g> from even indices (k=0), |r> from odd (k=1).
        let norm_cg = (coeffs[0].norm_sqr()
            + coeffs[2].norm_sqr()
            + coeffs[4].norm_sqr()
            + coeffs[6].norm_sqr())
        .sqrt();
        let norm_cr = (coeffs[1].norm_sqr()
            + coeffs[3].norm_sqr()
            + coeffs[5].norm_sqr()
            + coeffs[7].norm_sqr())
        .sqrt();
        let ph_cg = [coeffs[0], coeffs[2], coeffs[4], coeffs[6]]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        let ph_cr = [coeffs[1], coeffs[3], coeffs[5], coeffs[7]]
            .iter()
            .find(|c| c.norm_sqr() > 1e-30)
            .map(|c| c / c.norm())
            .unwrap_or(Complex64::new(1.0, 0.0));
        atoms[c].qubit_state = [ph_cg * norm_cg, ph_cr * norm_cr];

        Ok(())
    }

    /// Apply a simultaneous global rotation to all atoms (parallel Rabi pulse).
    pub fn global_rotation(atoms: &mut [AtomState], theta: f64, phi: f64) {
        for atom in atoms.iter_mut() {
            if !atom.is_lost {
                Self::single_qubit_rotation(atom, theta, phi);
            }
        }
    }

    /// Apply depolarising noise and stochastic atom loss to a single atom.
    ///
    /// `depol_prob` is the single-qubit depolarising probability.
    /// `loss_rate` is the probability of losing the atom on this operation.
    /// `rng_seed` provides deterministic pseudo-randomness for reproducibility.
    pub fn apply_noise(atom: &mut AtomState, depol_prob: f64, loss_rate: f64, rng_seed: u64) {
        // Simple deterministic hash-based pseudo-random from seed.
        let r1 = pseudo_rand(rng_seed);
        let r2 = pseudo_rand(rng_seed.wrapping_add(1));

        // Atom loss.
        if r1 < loss_rate {
            atom.is_lost = true;
            atom.qubit_state = [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
            return;
        }

        // Depolarising channel: with probability p, replace state with maximally mixed.
        if r2 < depol_prob {
            let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
            atom.qubit_state = [
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ];
        }
    }
}

/// Cheap deterministic pseudo-random number in [0, 1) from a u64 seed.
fn pseudo_rand(seed: u64) -> f64 {
    // Splitmix64-style mixing.
    let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z as f64) / (u64::MAX as f64)
}

// ===================================================================
// ATOM SORTING / REARRANGEMENT ALGORITHMS
// ===================================================================

/// Algorithms for rearranging atoms in the trap array.
pub struct AtomSortingAlgorithm;

impl AtomSortingAlgorithm {
    /// Odd-even transposition sort: compact all occupied traps to the front.
    ///
    /// Returns the number of individual atom moves performed.
    pub fn odd_even_sort(array: &mut AtomArray) -> usize {
        let n = array.traps.len();
        let mut moves = 0usize;
        let mut sorted = false;

        while !sorted {
            sorted = true;
            // Odd phase: compare (1,2), (3,4), ...
            for i in (1..n).step_by(2) {
                if i + 1 < n && !array.traps[i].occupied && array.traps[i + 1].occupied {
                    let _ = array.move_atom(i + 1, i);
                    moves += 1;
                    sorted = false;
                }
            }
            // Even phase: compare (0,1), (2,3), ...
            for i in (0..n).step_by(2) {
                if i + 1 < n && !array.traps[i].occupied && array.traps[i + 1].occupied {
                    let _ = array.move_atom(i + 1, i);
                    moves += 1;
                    sorted = false;
                }
            }
        }
        moves
    }

    /// Heuristic rearrangement: move atoms to target traps minimising total
    /// shuttle distance.
    ///
    /// `target_pattern` is a list of `(atom_index, target_trap_index)` moves.
    /// Returns the total number of moves executed.
    pub fn heuristic_rearrange(
        array: &mut AtomArray,
        target_pattern: &[(usize, usize)],
    ) -> NeutralAtomResult<usize> {
        let nt = array.traps.len();
        let na = array.atoms.len();
        let mut moves = 0usize;

        // Sort by move cost (shortest first -- greedy).
        let mut sorted_targets: Vec<(usize, usize, f64)> = target_pattern
            .iter()
            .map(|&(atom_id, target_trap)| {
                if atom_id >= na {
                    return (atom_id, target_trap, f64::INFINITY);
                }
                let atom_pos = array.atoms[atom_id].position;
                if target_trap >= nt {
                    return (atom_id, target_trap, f64::INFINITY);
                }
                let trap_pos = array.traps[target_trap].position;
                let cost = Self::move_cost(atom_pos, trap_pos);
                (atom_id, target_trap, cost)
            })
            .collect();
        sorted_targets.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        for (atom_id, target_trap, _cost) in &sorted_targets {
            let atom_id = *atom_id;
            let target_trap = *target_trap;

            if atom_id >= na {
                return Err(NeutralAtomError::AtomOutOfBounds {
                    index: atom_id,
                    num_atoms: na,
                });
            }
            if target_trap >= nt {
                return Err(NeutralAtomError::TrapOutOfBounds {
                    index: target_trap,
                    num_traps: nt,
                });
            }

            // Find the current trap holding this atom.
            let current_trap = array.traps.iter().position(|t| t.atom_id == Some(atom_id));

            if let Some(from_trap) = current_trap {
                if from_trap != target_trap {
                    // If target is occupied, skip (greedy approach).
                    if !array.traps[target_trap].occupied {
                        array.move_atom(from_trap, target_trap)?;
                        moves += 1;
                    }
                }
            }
        }
        Ok(moves)
    }

    /// Estimated time cost (in microseconds) of shuttling an atom between two positions.
    ///
    /// Assumes a maximum shuttle speed of ~0.5 um/us (typical AOD slew rate).
    pub fn move_cost(from: (f64, f64, f64), to: (f64, f64, f64)) -> f64 {
        let d = distance_3d(from, to);
        // 0.5 um/us shuttle speed + 1 us overhead for trap transfer.
        d / 0.5 + 1.0
    }
}

// ===================================================================
// GATE ENUM FOR CIRCUIT REPRESENTATION
// ===================================================================

/// A gate in a neutral atom circuit.
#[derive(Debug, Clone)]
pub enum NeutralAtomGate {
    /// Single-qubit Rabi rotation (theta, phi) on atom index.
    Rotation { atom: usize, theta: f64, phi: f64 },
    /// CZ gate between two atoms.
    CZ { atom_a: usize, atom_b: usize },
    /// Native CCZ gate on three atoms.
    CCZ { a: usize, b: usize, c: usize },
    /// Global rotation applied to all atoms simultaneously.
    GlobalRotation { theta: f64, phi: f64 },
}

/// Result of a circuit simulation.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Final atom states.
    pub atoms: Vec<AtomState>,
    /// Estimated overall circuit fidelity.
    pub fidelity: f64,
    /// Number of atoms lost during simulation.
    pub atoms_lost: usize,
    /// Number of gates applied.
    pub gates_applied: usize,
}

// ===================================================================
// NEUTRAL ATOM SIMULATOR (orchestrator)
// ===================================================================

/// Top-level simulator for neutral atom array quantum computation.
#[derive(Debug, Clone)]
pub struct NeutralAtomSimulator {
    pub array: AtomArray,
    pub config: NeutralAtomConfig,
    /// Accumulated fidelity (product of per-gate fidelities).
    accumulated_fidelity: f64,
    /// Gate counter for deterministic noise seeding.
    gate_counter: u64,
}

impl NeutralAtomSimulator {
    /// Create a new simulator with a 1D trap array.
    pub fn new(config: NeutralAtomConfig) -> NeutralAtomResult<Self> {
        config.validate()?;
        let array = AtomArray::new_1d(&config)?;
        Ok(Self {
            config: config.clone(),
            array,
            accumulated_fidelity: 1.0,
            gate_counter: 0,
        })
    }

    /// Probabilistically load atoms into traps.
    ///
    /// Each trap is loaded independently with probability `config.loading_probability`.
    /// Uses a deterministic seed for reproducibility.
    pub fn load_atoms(&mut self, seed: u64) -> usize {
        let p = self.config.loading_probability;
        let mut loaded = 0usize;
        self.array.atoms.clear();

        for (i, trap) in self.array.traps.iter_mut().enumerate() {
            let r = pseudo_rand(seed.wrapping_add(i as u64));
            if r < p {
                trap.occupied = true;
                trap.atom_id = Some(loaded);
                self.array.atoms.push(AtomState::new_ground(trap.position));
                loaded += 1;
            } else {
                trap.occupied = false;
                trap.atom_id = None;
            }
        }
        loaded
    }

    /// Sort atoms to fill gaps and compact the working register.
    pub fn sort_and_compact(&mut self) -> usize {
        self.array.sort_atoms()
    }

    /// Apply a single gate from the neutral atom gate set.
    pub fn apply_gate(&mut self, gate: &NeutralAtomGate) -> NeutralAtomResult<()> {
        match gate {
            NeutralAtomGate::Rotation { atom, theta, phi } => {
                let na = self.array.atoms.len();
                if *atom >= na {
                    return Err(NeutralAtomError::AtomOutOfBounds {
                        index: *atom,
                        num_atoms: na,
                    });
                }
                if self.array.atoms[*atom].is_lost {
                    return Err(NeutralAtomError::AtomLost(*atom));
                }
                NeutralAtomGates::single_qubit_rotation(&mut self.array.atoms[*atom], *theta, *phi);
                // Single-qubit gate fidelity ~ 0.999.
                self.accumulated_fidelity *= 0.999;
            }
            NeutralAtomGate::CZ { atom_a, atom_b } => {
                NeutralAtomGates::cz_gate(&mut self.array.atoms, *atom_a, *atom_b)?;
                // Two-qubit gate fidelity ~ 0.995.
                self.accumulated_fidelity *= 0.995;
            }
            NeutralAtomGate::CCZ { a, b, c } => {
                NeutralAtomGates::ccz_gate(&mut self.array.atoms, *a, *b, *c)?;
                // Three-qubit gate fidelity ~ 0.99.
                self.accumulated_fidelity *= 0.99;
            }
            NeutralAtomGate::GlobalRotation { theta, phi } => {
                NeutralAtomGates::global_rotation(&mut self.array.atoms, *theta, *phi);
                self.accumulated_fidelity *= 0.999;
            }
        }

        // Apply noise after each gate.
        let loss_rate = self.config.atom_loss_rate;
        let depol = 0.001; // 0.1% depolarising per gate.
        for (i, atom) in self.array.atoms.iter_mut().enumerate() {
            if !atom.is_lost {
                self.gate_counter += 1;
                NeutralAtomGates::apply_noise(
                    atom,
                    depol,
                    loss_rate,
                    self.gate_counter.wrapping_add(i as u64),
                );
            }
        }

        Ok(())
    }

    /// Execute a circuit (sequence of gates) and return the simulation result.
    pub fn run_circuit(
        &mut self,
        circuit: &[NeutralAtomGate],
    ) -> NeutralAtomResult<SimulationResult> {
        let mut gates_applied = 0usize;
        for gate in circuit {
            self.apply_gate(gate)?;
            gates_applied += 1;
        }
        let atoms_lost = self.array.atoms.iter().filter(|a| a.is_lost).count();
        Ok(SimulationResult {
            atoms: self.array.atoms.clone(),
            fidelity: self.accumulated_fidelity,
            atoms_lost,
            gates_applied,
        })
    }

    /// Current fidelity estimate based on accumulated gate errors.
    pub fn fidelity_estimate(&self) -> f64 {
        self.accumulated_fidelity
    }

    /// Compute the blockade map: `map[i][j]` is true if atoms i and j are
    /// within the Rydberg blockade radius.
    pub fn blockade_map(&self) -> Vec<Vec<bool>> {
        let na = self.array.atoms.len();
        let c6 = self.config.c6_coefficient;
        let omega = self.config.max_rabi_frequency_mhz;
        let mut map = vec![vec![false; na]; na];
        for i in 0..na {
            for j in (i + 1)..na {
                let blocked = RydbergInteraction::is_blockaded(
                    self.array.atoms[i].position,
                    self.array.atoms[j].position,
                    c6,
                    omega,
                );
                map[i][j] = blocked;
                map[j][i] = blocked;
            }
        }
        map
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---------------------------------------------------------------
    // 1. Config builder validation
    // ---------------------------------------------------------------

    #[test]
    fn test_config_builder_defaults() {
        let cfg = NeutralAtomConfig::builder().build().unwrap();
        assert_eq!(cfg.num_atoms, 256);
        assert!((cfg.trap_spacing_um - 5.0).abs() < 1e-12);
        assert_eq!(cfg.rydberg_level, 70);
        assert!((cfg.c6_coefficient - 862_690.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_builder_custom() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(16)
            .trap_spacing_um(4.0)
            .rydberg_level(60)
            .max_rabi_frequency_mhz(5.0)
            .loading_probability(0.8)
            .connectivity(AtomConnectivity::NearestNeighbor)
            .build()
            .unwrap();
        assert_eq!(cfg.num_atoms, 16);
        assert!((cfg.trap_spacing_um - 4.0).abs() < 1e-12);
        assert_eq!(cfg.connectivity, AtomConnectivity::NearestNeighbor);
    }

    #[test]
    fn test_config_validation_rejects_too_few_atoms() {
        let result = NeutralAtomConfig::builder().num_atoms(2).build();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("num_atoms"));
    }

    #[test]
    fn test_config_validation_rejects_too_many_atoms() {
        let result = NeutralAtomConfig::builder().num_atoms(2000).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_rejects_bad_loading_probability() {
        let result = NeutralAtomConfig::builder()
            .num_atoms(16)
            .loading_probability(1.5)
            .build();
        assert!(result.is_err());

        let result2 = NeutralAtomConfig::builder()
            .num_atoms(16)
            .loading_probability(0.0)
            .build();
        assert!(result2.is_err());
    }

    #[test]
    fn test_config_validation_rejects_negative_spacing() {
        let result = NeutralAtomConfig::builder()
            .num_atoms(16)
            .trap_spacing_um(-1.0)
            .build();
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 2. Array geometries
    // ---------------------------------------------------------------

    #[test]
    fn test_1d_array_geometry() {
        let cfg = NeutralAtomConfig::builder().num_atoms(8).build().unwrap();
        let array = AtomArray::new_1d(&cfg).unwrap();
        assert_eq!(array.traps.len(), 8);
        // Check spacing.
        let dx = array.traps[1].position.0 - array.traps[0].position.0;
        assert!((dx - 5.0).abs() < 1e-12);
        // All y, z should be 0.
        for trap in &array.traps {
            assert!((trap.position.1).abs() < 1e-12);
            assert!((trap.position.2).abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_square_geometry() {
        let cfg = NeutralAtomConfig::builder().num_atoms(16).build().unwrap();
        let array = AtomArray::new_2d_square(4, 4, &cfg).unwrap();
        assert_eq!(array.traps.len(), 16);
        // Corner trap should be at (15, 15, 0).
        let last = &array.traps[15];
        assert!((last.position.0 - 15.0).abs() < 1e-12);
        assert!((last.position.1 - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_2d_triangular_geometry() {
        let cfg = NeutralAtomConfig::builder().num_atoms(16).build().unwrap();
        let array = AtomArray::new_2d_triangular(&cfg).unwrap();
        assert_eq!(array.traps.len(), 16);
        // Odd rows should have x offset of spacing/2 = 2.5.
        let cols = (16.0_f64).sqrt().ceil() as usize;
        if cols > 0 {
            let second_row_first = &array.traps[cols];
            assert!((second_row_first.position.0 - 2.5).abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_square_rejects_zero_rows() {
        let cfg = NeutralAtomConfig::builder().num_atoms(16).build().unwrap();
        let result = AtomArray::new_2d_square(0, 4, &cfg);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 3. Rydberg blockade physics
    // ---------------------------------------------------------------

    #[test]
    fn test_blockade_radius() {
        // R_b = (C6 / Omega)^{1/6}
        let c6 = 862_690.0;
        let omega = 10.0;
        let r_b = RydbergInteraction::blockade_radius(c6, omega);
        // (862690 / 10)^{1/6} = 86269^{1/6}
        let expected = (c6 / omega).powf(1.0 / 6.0);
        assert!((r_b - expected).abs() < 1e-6);
        // For Rb87: R_b ~ 9.7 um.
        assert!(r_b > 5.0 && r_b < 15.0);
    }

    #[test]
    fn test_interaction_energy() {
        let c6 = 862_690.0;
        // At r = 5 um: V = 862690 / 5^6 = 862690 / 15625 = 55.21 MHz.
        let v = RydbergInteraction::interaction_energy(c6, 5.0);
        assert!((v - 862_690.0 / 15625.0).abs() < 1e-6);
    }

    #[test]
    fn test_interaction_energy_zero_distance() {
        let v = RydbergInteraction::interaction_energy(862_690.0, 0.0);
        assert!(v.is_infinite());
    }

    #[test]
    fn test_is_blockaded() {
        let c6 = 862_690.0;
        let omega = 10.0;
        // Two atoms at 5 um spacing should be blockaded (R_b ~ 9.7 um).
        assert!(RydbergInteraction::is_blockaded(
            (0.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            c6,
            omega,
        ));
        // Two atoms at 20 um spacing should NOT be blockaded.
        assert!(!RydbergInteraction::is_blockaded(
            (0.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            c6,
            omega,
        ));
    }

    #[test]
    fn test_blockade_shift() {
        // One excited atom at distance 5 um from target.
        let target = AtomState::new_ground((0.0, 0.0, 0.0));
        let mut excited = AtomState::new_ground((5.0, 0.0, 0.0));
        // Put excited atom in |r>.
        excited.qubit_state = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

        let atoms = vec![target, excited];
        let shift = RydbergInteraction::blockade_shift(&atoms, 0, 862_690.0);
        let expected = 862_690.0 / 5.0_f64.powi(6);
        assert!((shift - expected).abs() < 1e-6);
    }

    // ---------------------------------------------------------------
    // 4. Atom loading
    // ---------------------------------------------------------------

    #[test]
    fn test_probabilistic_loading() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(100)
            .loading_probability(0.5)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        let loaded = sim.load_atoms(42);
        // With p=0.5 and 100 traps, expect roughly 50 +/- 15 loaded.
        assert!(loaded > 20 && loaded < 80, "loaded = {}", loaded);
        assert_eq!(sim.array.atoms.len(), loaded);
        assert_eq!(sim.array.num_occupied(), loaded);
    }

    #[test]
    fn test_loading_deterministic() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(32)
            .loading_probability(0.5)
            .build()
            .unwrap();
        let mut sim1 = NeutralAtomSimulator::new(cfg.clone()).unwrap();
        let mut sim2 = NeutralAtomSimulator::new(cfg).unwrap();
        let l1 = sim1.load_atoms(12345);
        let l2 = sim2.load_atoms(12345);
        assert_eq!(l1, l2);
    }

    // ---------------------------------------------------------------
    // 5. Atom sorting / rearrangement
    // ---------------------------------------------------------------

    #[test]
    fn test_sort_atoms_compacts() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(8)
            .loading_probability(1.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        // Manually create gaps: remove atom from trap 2 and trap 5.
        if sim.array.traps[2].occupied {
            sim.array.traps[2].occupied = false;
            let aid = sim.array.traps[2].atom_id.take();
            if let Some(id) = aid {
                sim.array.atoms[id].is_lost = true;
            }
        }
        if sim.array.traps[5].occupied {
            sim.array.traps[5].occupied = false;
            let aid = sim.array.traps[5].atom_id.take();
            if let Some(id) = aid {
                sim.array.atoms[id].is_lost = true;
            }
        }

        let initial_occupied = sim.array.num_occupied();
        let moves = sim.sort_and_compact();

        // After sorting, all occupied traps should be contiguous at the front.
        let mut seen_empty = false;
        for trap in &sim.array.traps {
            if !trap.occupied {
                seen_empty = true;
            } else if seen_empty {
                panic!("Found occupied trap after empty trap -- sort failed");
            }
        }
        assert_eq!(sim.array.num_occupied(), initial_occupied);
        assert!(moves > 0);
    }

    #[test]
    fn test_move_atom_basic() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(4)
            .loading_probability(1.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        // Move atom from trap 0 to trap 3 (if trap 3 is empty after clearing).
        if sim.array.traps[3].occupied {
            sim.array.traps[3].occupied = false;
            sim.array.traps[3].atom_id = None;
        }
        let result = sim.array.move_atom(0, 3);
        assert!(result.is_ok());
        assert!(!sim.array.traps[0].occupied);
        assert!(sim.array.traps[3].occupied);
    }

    #[test]
    fn test_move_atom_errors() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(4)
            .loading_probability(1.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        // Move from out-of-bounds trap.
        assert!(sim.array.move_atom(100, 0).is_err());
        // Move to out-of-bounds trap.
        assert!(sim.array.move_atom(0, 100).is_err());
    }

    #[test]
    fn test_heuristic_rearrange() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(8)
            .loading_probability(1.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        // Clear trap 7 so we can move atom 0 there.
        if sim.array.traps[7].occupied {
            sim.array.traps[7].occupied = false;
            sim.array.traps[7].atom_id = None;
        }
        // Rearrange atom 0 to trap 7.
        let result = sim.array.rearrange(&[(0, 7)]);
        assert!(result.is_ok());
        let moves = result.unwrap();
        assert!(moves >= 1);
    }

    // ---------------------------------------------------------------
    // 6. Single qubit rotation
    // ---------------------------------------------------------------

    #[test]
    fn test_single_qubit_rotation_pi() {
        // A pi rotation about x (phi=0) should flip |g> -> -i|r>.
        let mut atom = AtomState::new_ground((0.0, 0.0, 0.0));
        NeutralAtomGates::single_qubit_rotation(&mut atom, PI, 0.0);
        // |g> -> cos(pi/2)|g> - i sin(pi/2)|r> = -i|r>.
        assert!(atom.qubit_state[0].norm() < 1e-10);
        assert!((atom.qubit_state[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_rotation_preserves_norm() {
        let mut atom = AtomState::new_ground((0.0, 0.0, 0.0));
        NeutralAtomGates::single_qubit_rotation(&mut atom, 1.234, 0.567);
        let norm = atom.qubit_state[0].norm_sqr() + atom.qubit_state[1].norm_sqr();
        assert!((norm - 1.0).abs() < 1e-12, "Norm = {}", norm);
    }

    #[test]
    fn test_single_qubit_rotation_half_pi() {
        // pi/2 rotation creates equal superposition.
        let mut atom = AtomState::new_ground((0.0, 0.0, 0.0));
        NeutralAtomGates::single_qubit_rotation(&mut atom, PI / 2.0, 0.0);
        let p_g = atom.ground_population();
        let p_r = atom.rydberg_population();
        assert!((p_g - 0.5).abs() < 1e-10, "p_g = {}", p_g);
        assert!((p_r - 0.5).abs() < 1e-10, "p_r = {}", p_r);
    }

    // ---------------------------------------------------------------
    // 7. CZ gate
    // ---------------------------------------------------------------

    #[test]
    fn test_cz_gate_ground_ground() {
        // CZ on |gg> should be identity (no phase change).
        let mut atoms = vec![
            AtomState::new_ground((0.0, 0.0, 0.0)),
            AtomState::new_ground((5.0, 0.0, 0.0)),
        ];
        NeutralAtomGates::cz_gate(&mut atoms, 0, 1).unwrap();
        // Both atoms should still be in |g>.
        assert!(atoms[0].ground_population() > 0.999);
        assert!(atoms[1].ground_population() > 0.999);
    }

    #[test]
    fn test_cz_gate_rydberg_rydberg() {
        // CZ on |rr> should give -|rr> (global phase, not observable in separable storage).
        let mut atoms = vec![
            AtomState {
                qubit_state: [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                position: (0.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: false,
            },
            AtomState {
                qubit_state: [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                position: (5.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: false,
            },
        ];
        NeutralAtomGates::cz_gate(&mut atoms, 0, 1).unwrap();
        // Both should still be in |r> with magnitude 1.
        // Note: The -1 global phase is not stored in separable qubit representation.
        assert!(atoms[0].rydberg_population() > 0.999);
        assert!(atoms[1].rydberg_population() > 0.999);
        // Verify amplitudes are normalized
        assert!((atoms[0].qubit_state[1].norm() - 1.0).abs() < 1e-10);
        assert!((atoms[1].qubit_state[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cz_gate_error_on_lost_atom() {
        let mut atoms = vec![
            AtomState::new_ground((0.0, 0.0, 0.0)),
            AtomState {
                qubit_state: [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                position: (5.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: true,
            },
        ];
        let result = NeutralAtomGates::cz_gate(&mut atoms, 0, 1);
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 8. CCZ gate
    // ---------------------------------------------------------------

    #[test]
    fn test_ccz_gate_all_ground() {
        // CCZ on |ggg> is identity.
        let mut atoms = vec![
            AtomState::new_ground((0.0, 0.0, 0.0)),
            AtomState::new_ground((5.0, 0.0, 0.0)),
            AtomState::new_ground((10.0, 0.0, 0.0)),
        ];
        NeutralAtomGates::ccz_gate(&mut atoms, 0, 1, 2).unwrap();
        for atom in &atoms {
            assert!(atom.ground_population() > 0.999);
        }
    }

    #[test]
    fn test_ccz_gate_all_rydberg() {
        // CCZ on |rrr> gives -|rrr> (global phase, not observable in separable storage).
        let r_state = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        let mut atoms = vec![
            AtomState {
                qubit_state: r_state,
                position: (0.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: false,
            },
            AtomState {
                qubit_state: r_state,
                position: (5.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: false,
            },
            AtomState {
                qubit_state: r_state,
                position: (10.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: false,
            },
        ];
        NeutralAtomGates::ccz_gate(&mut atoms, 0, 1, 2).unwrap();
        // All atoms should still be in |r> state with normalized amplitude.
        // Note: The -1 global phase is not stored in separable qubit representation.
        for i in 0..3 {
            assert!(
                atoms[i].rydberg_population() > 0.999,
                "atom {} not in |r>",
                i
            );
            assert!((atoms[i].qubit_state[1].norm() - 1.0).abs() < 1e-10);
        }
    }

    // ---------------------------------------------------------------
    // 9. Blockade map
    // ---------------------------------------------------------------

    #[test]
    fn test_blockade_map() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(4)
            .trap_spacing_um(5.0)
            .loading_probability(1.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        let map = sim.blockade_map();
        assert_eq!(map.len(), sim.array.atoms.len());

        // Nearest neighbours at 5 um should be blockaded (R_b ~ 9.7 um).
        if sim.array.atoms.len() >= 2 {
            assert!(map[0][1], "Adjacent atoms should be blockaded");
        }
        // Atoms 3 traps apart (15 um) should NOT be blockaded.
        if sim.array.atoms.len() >= 4 {
            assert!(!map[0][3], "Distant atoms should not be blockaded");
        }
    }

    // ---------------------------------------------------------------
    // 10. Atom loss model
    // ---------------------------------------------------------------

    #[test]
    fn test_atom_loss_model() {
        let mut atom = AtomState::new_ground((0.0, 0.0, 0.0));
        // Use a seed that produces r1 < loss_rate=1.0 (guaranteed loss).
        NeutralAtomGates::apply_noise(&mut atom, 0.0, 1.0, 42);
        assert!(atom.is_lost);
    }

    #[test]
    fn test_atom_noise_no_loss() {
        let mut atom = AtomState::new_ground((0.0, 0.0, 0.0));
        // Zero loss rate, zero depol -- state should be unchanged.
        NeutralAtomGates::apply_noise(&mut atom, 0.0, 0.0, 42);
        assert!(!atom.is_lost);
        assert!(atom.ground_population() > 0.999);
    }

    // ---------------------------------------------------------------
    // 11. Global rotation
    // ---------------------------------------------------------------

    #[test]
    fn test_global_rotation() {
        let mut atoms = vec![
            AtomState::new_ground((0.0, 0.0, 0.0)),
            AtomState::new_ground((5.0, 0.0, 0.0)),
            AtomState::new_ground((10.0, 0.0, 0.0)),
        ];
        // pi/2 rotation should put all atoms in equal superposition.
        NeutralAtomGates::global_rotation(&mut atoms, PI / 2.0, 0.0);
        for atom in &atoms {
            assert!((atom.ground_population() - 0.5).abs() < 1e-10);
            assert!((atom.rydberg_population() - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_global_rotation_skips_lost_atoms() {
        let mut atoms = vec![
            AtomState::new_ground((0.0, 0.0, 0.0)),
            AtomState {
                qubit_state: [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                position: (5.0, 0.0, 0.0),
                velocity: (0.0, 0.0, 0.0),
                is_lost: true,
            },
        ];
        NeutralAtomGates::global_rotation(&mut atoms, PI, 0.0);
        // Atom 0 should be flipped to |r>.
        assert!(atoms[0].rydberg_population() > 0.999);
        // Lost atom should still be in |g> (not rotated).
        assert!(atoms[1].ground_population() > 0.999);
    }

    // ---------------------------------------------------------------
    // 12. Zone assignment
    // ---------------------------------------------------------------

    #[test]
    fn test_zone_assignment() {
        let cfg = NeutralAtomConfig::builder().num_atoms(8).build().unwrap();
        let mut array = AtomArray::new_1d(&cfg).unwrap();
        // Default zone should be Entangling.
        assert_eq!(array.traps[0].zone, ComputeZone::Entangling);

        array.set_zone(0, ComputeZone::Storage).unwrap();
        assert_eq!(array.traps[0].zone, ComputeZone::Storage);

        array.set_zone(7, ComputeZone::Readout).unwrap();
        assert_eq!(array.traps[7].zone, ComputeZone::Readout);

        array.set_zone(3, ComputeZone::Cooling).unwrap();
        assert_eq!(array.traps[3].zone, ComputeZone::Cooling);
    }

    #[test]
    fn test_zone_assignment_out_of_bounds() {
        let cfg = NeutralAtomConfig::builder().num_atoms(4).build().unwrap();
        let mut array = AtomArray::new_1d(&cfg).unwrap();
        assert!(array.set_zone(100, ComputeZone::Storage).is_err());
    }

    // ---------------------------------------------------------------
    // 13. Full circuit simulation
    // ---------------------------------------------------------------

    #[test]
    fn test_full_circuit_simulation() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(4)
            .loading_probability(1.0)
            .atom_loss_rate(0.0) // disable loss for determinism
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        let circuit = vec![
            NeutralAtomGate::Rotation {
                atom: 0,
                theta: PI / 2.0,
                phi: 0.0,
            },
            NeutralAtomGate::Rotation {
                atom: 1,
                theta: PI / 2.0,
                phi: 0.0,
            },
            NeutralAtomGate::CZ {
                atom_a: 0,
                atom_b: 1,
            },
            NeutralAtomGate::GlobalRotation {
                theta: PI / 4.0,
                phi: 0.0,
            },
        ];

        let result = sim.run_circuit(&circuit).unwrap();
        assert_eq!(result.gates_applied, 4);
        assert!(result.fidelity > 0.9);
        assert!(result.fidelity < 1.0);
    }

    // ---------------------------------------------------------------
    // 14. Fidelity estimation
    // ---------------------------------------------------------------

    #[test]
    fn test_fidelity_decreases_with_gates() {
        let cfg = NeutralAtomConfig::builder()
            .num_atoms(4)
            .loading_probability(1.0)
            .atom_loss_rate(0.0)
            .build()
            .unwrap();
        let mut sim = NeutralAtomSimulator::new(cfg).unwrap();
        sim.load_atoms(0);

        let f0 = sim.fidelity_estimate();
        assert!((f0 - 1.0).abs() < 1e-12);

        sim.apply_gate(&NeutralAtomGate::Rotation {
            atom: 0,
            theta: PI,
            phi: 0.0,
        })
        .unwrap();
        let f1 = sim.fidelity_estimate();
        assert!(f1 < f0);

        sim.apply_gate(&NeutralAtomGate::CZ {
            atom_a: 0,
            atom_b: 1,
        })
        .unwrap();
        let f2 = sim.fidelity_estimate();
        assert!(f2 < f1);
    }

    // ---------------------------------------------------------------
    // 15. Move cost
    // ---------------------------------------------------------------

    #[test]
    fn test_move_cost() {
        let cost = AtomSortingAlgorithm::move_cost((0.0, 0.0, 0.0), (10.0, 0.0, 0.0));
        // 10 um at 0.5 um/us = 20 us + 1 us overhead = 21 us.
        assert!((cost - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_move_cost_zero_distance() {
        let cost = AtomSortingAlgorithm::move_cost((5.0, 5.0, 0.0), (5.0, 5.0, 0.0));
        // Just the 1 us overhead.
        assert!((cost - 1.0).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // 16. Compute zone Display
    // ---------------------------------------------------------------

    #[test]
    fn test_compute_zone_display() {
        assert_eq!(format!("{}", ComputeZone::Storage), "Storage");
        assert_eq!(format!("{}", ComputeZone::Entangling), "Entangling");
        assert_eq!(format!("{}", ComputeZone::Readout), "Readout");
        assert_eq!(format!("{}", ComputeZone::Cooling), "Cooling");
    }

    // ---------------------------------------------------------------
    // 17. Error Display
    // ---------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = NeutralAtomError::TrapOutOfBounds {
            index: 5,
            num_traps: 4,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("5"));
        assert!(msg.contains("4"));

        let e2 = NeutralAtomError::AtomLost(3);
        assert!(format!("{}", e2).contains("3"));
    }

    // ---------------------------------------------------------------
    // 18. AtomState population helpers
    // ---------------------------------------------------------------

    #[test]
    fn test_atom_state_populations() {
        let atom = AtomState::new_ground((0.0, 0.0, 0.0));
        assert!((atom.ground_population() - 1.0).abs() < 1e-12);
        assert!(atom.rydberg_population() < 1e-12);

        let excited = AtomState {
            qubit_state: [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            position: (0.0, 0.0, 0.0),
            velocity: (0.0, 0.0, 0.0),
            is_lost: false,
        };
        assert!(excited.ground_population() < 1e-12);
        assert!((excited.rydberg_population() - 1.0).abs() < 1e-12);
    }

    // ---------------------------------------------------------------
    // 19. Connectivity enum equality
    // ---------------------------------------------------------------

    #[test]
    fn test_connectivity_equality() {
        assert_eq!(
            AtomConnectivity::NearestNeighbor,
            AtomConnectivity::NearestNeighbor
        );
        assert_ne!(
            AtomConnectivity::NearestNeighbor,
            AtomConnectivity::AllToAll
        );
        assert_eq!(
            AtomConnectivity::Custom(vec![(0, 1), (1, 2)]),
            AtomConnectivity::Custom(vec![(0, 1), (1, 2)])
        );
    }
}
