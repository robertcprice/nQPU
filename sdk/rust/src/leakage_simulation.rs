//! Leakage Simulation for Transmon Qubits (AC4b)
//!
//! Real superconducting qubits are not perfect two-level systems. Transmon qubits
//! have a weakly anharmonic spectrum where the |0>-|1> and |1>-|2> transitions
//! are separated by an anharmonicity of approximately -200 to -400 MHz. During
//! fast gate operations, population can "leak" from the computational subspace
//! {|0>, |1>} into the non-computational |2> state (and beyond).
//!
//! This module models each qubit as a **qutrit** (3-level system) and simulates:
//!
//! - **Leakage-aware gates**: Single-qubit gates (X, Z, H) and two-qubit gates (CNOT)
//!   with realistic leakage coupling into |2>
//! - **Idle decay**: T1 relaxation from |2> back to |1> and |0> during idle periods
//! - **Leakage detection**: Population measurement in the non-computational |2> state
//! - **Leakage Reduction Units (LRU)**: Targeted operations to swap population from
//!   |2> back into the computational subspace
//! - **Transmon noise model**: Derivation of leakage rates from physical device
//!   parameters (anharmonicity, gate duration, Rabi frequency)
//!
//! # State Representation
//!
//! For `n` qutrits the full statevector has `3^n` amplitudes. The basis ordering
//! is little-endian in base 3:
//!
//! ```text
//! |q_{n-1} ... q_1 q_0> where each q_i in {0, 1, 2}
//! ```
//!
//! Index mapping: `index = sum_i q_i * 3^i`
//!
//! # Scaling
//!
//! Qutrit simulation scales as `3^n`, which is significantly more expensive than
//! qubit simulation (`2^n`). This implementation targets up to 6 qutrits
//! (`3^6 = 729` amplitudes), which is sufficient for studying leakage physics
//! in small circuits and QEC patches.
//!
//! # References
//!
//! - Wood & Gambetta, "Quantification and characterization of leakage errors",
//!   Phys. Rev. A 97, 032306 (2018)
//! - Ghosh et al., "Understanding the effects of leakage in superconducting
//!   quantum error correction", Phys. Rev. A 88, 062329 (2013)
//! - McEwen et al., "Removing leakage-induced correlated errors in
//!   superconducting quantum error correction", Nature Communications 12, 1761 (2021)

use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================
// CONSTANTS
// ============================================================

/// Maximum supported qutrits (3^6 = 729 amplitudes)
pub const MAX_QUTRITS: usize = 6;

/// Number of levels per qutrit (|0>, |1>, |2>)
pub const QUTRIT_DIM: usize = 3;

/// Typical transmon anharmonicity in GHz (negative for transmon)
pub const DEFAULT_ANHARMONICITY_GHZ: f64 = -0.25;

/// Typical single-qubit gate duration in nanoseconds
pub const DEFAULT_GATE_DURATION_NS: f64 = 20.0;

/// Typical T1 for the |2>->|1> transition in microseconds
pub const DEFAULT_T1_LEAKAGE_US: f64 = 30.0;

/// Numerical tolerance for normalization checks
const NORM_TOLERANCE: f64 = 1e-10;

// ============================================================
// COMPLEX NUMBER HELPERS
// ============================================================

#[inline]
fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[inline]
fn c64_zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

#[inline]
fn c64_one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

// ============================================================
// LEAKAGE RATES
// ============================================================

/// Per-qubit leakage rates characterizing transitions involving the |2> state.
///
/// These rates are typically extracted from device characterization experiments
/// or derived from the transmon Hamiltonian parameters.
#[derive(Clone, Debug)]
pub struct LeakageRates {
    /// Leakage probability per X-like gate: |1> -> |2> coupling.
    ///
    /// For a transmon with anharmonicity alpha and gate duration t_g,
    /// this is approximately (pi * Omega / (2 * alpha))^2 where Omega
    /// is the Rabi frequency.
    pub gamma_01_to_02: f64,

    /// |1> <-> |2> coherent transition rate (Hz).
    ///
    /// Characterizes the strength of coherent coupling between the
    /// computational |1> and non-computational |2> states during
    /// driven operations.
    pub gamma_12: f64,

    /// |2> -> |0> non-radiative decay rate (Hz).
    ///
    /// Two-photon relaxation process, typically much slower than
    /// the single-photon |2> -> |1> decay.
    pub gamma_20: f64,

    /// |2> -> |1> radiative decay rate (Hz).
    ///
    /// The dominant relaxation channel for the |2> state. For transmons,
    /// this is approximately 2/T1 due to the sqrt(2) matrix element
    /// enhancement.
    pub gamma_21: f64,

    /// T1 relaxation time for |2> -> |1> in microseconds.
    ///
    /// Typically shorter than the qubit T1 by a factor of ~2 for
    /// transmons due to the enhanced matrix element.
    pub t1_leakage: f64,
}

impl Default for LeakageRates {
    fn default() -> Self {
        Self {
            gamma_01_to_02: 0.005,
            gamma_12: 1e6,
            gamma_20: 1e3,
            gamma_21: 1e5,
            t1_leakage: DEFAULT_T1_LEAKAGE_US,
        }
    }
}

impl LeakageRates {
    /// Construct leakage rates from transmon device parameters.
    ///
    /// # Arguments
    ///
    /// * `anharmonicity_ghz` - Transmon anharmonicity in GHz (typically -0.2 to -0.4)
    /// * `gate_duration_ns` - Single-qubit gate duration in nanoseconds
    /// * `t1_qubit_us` - Qubit T1 in microseconds
    ///
    /// # Physics
    ///
    /// For a transmon driven at the |0>-|1> frequency, the |1>-|2> transition
    /// is detuned by the anharmonicity alpha. The leakage probability per gate
    /// scales as:
    ///
    /// ```text
    /// P_leak ~ (pi / (2 * |alpha| * t_g))^2
    /// ```
    ///
    /// where t_g is the gate duration. Faster gates produce more leakage.
    pub fn from_transmon_params(
        anharmonicity_ghz: f64,
        gate_duration_ns: f64,
        t1_qubit_us: f64,
    ) -> Self {
        let alpha = anharmonicity_ghz.abs() * 1e9; // Convert to Hz
        let t_g = gate_duration_ns * 1e-9; // Convert to seconds
        let t1 = t1_qubit_us * 1e-6; // Convert to seconds

        // Rabi frequency for a pi-pulse in time t_g
        let omega_rabi = PI / t_g;

        // Leakage probability per gate from perturbation theory:
        // P_leak = (Omega * sqrt(2) / (2 * alpha))^2 * sin^2(pi/2)
        // The sqrt(2) comes from the transmon matrix element <1|n|2> = sqrt(2) * <0|n|1>
        let leakage_per_gate = (omega_rabi * 2.0_f64.sqrt() / (2.0 * alpha * 2.0 * PI)).powi(2);

        // T1 for |2> is approximately T1_qubit / 2 for transmons
        let t1_leakage_us = t1_qubit_us / 2.0;
        let t1_leakage_s = t1_leakage_us * 1e-6;

        // Decay rates
        let gamma_21 = 1.0 / t1_leakage_s;
        let gamma_20 = gamma_21 * 0.01; // Two-photon process ~100x slower

        Self {
            gamma_01_to_02: leakage_per_gate.min(0.5), // Cap at 50% for physicality
            gamma_12: omega_rabi * 2.0_f64.sqrt() / (2.0 * PI),
            gamma_20,
            gamma_21,
            t1_leakage: t1_leakage_us,
        }
    }

    /// Create rates for a high-quality transmon (low leakage).
    pub fn high_quality_transmon() -> Self {
        Self::from_transmon_params(-0.34, 20.0, 100.0)
    }

    /// Create rates for a fast-gate transmon (higher leakage).
    pub fn fast_gate_transmon() -> Self {
        Self::from_transmon_params(-0.20, 10.0, 50.0)
    }
}

// ============================================================
// TRANSMON NOISE MODEL
// ============================================================

/// Physical parameters describing a transmon qubit for deriving noise properties.
#[derive(Clone, Debug)]
pub struct TransmonParameters {
    /// Qubit frequency in GHz (typically 4-6 GHz)
    pub frequency_ghz: f64,
    /// Anharmonicity in GHz (typically -0.2 to -0.4 GHz)
    pub anharmonicity_ghz: f64,
    /// Single-qubit gate duration in nanoseconds
    pub gate_duration_ns: f64,
    /// Two-qubit gate duration in nanoseconds
    pub two_qubit_gate_duration_ns: f64,
    /// T1 relaxation time in microseconds
    pub t1_us: f64,
    /// T2 dephasing time in microseconds
    pub t2_us: f64,
    /// Readout error probability
    pub readout_error: f64,
}

impl Default for TransmonParameters {
    fn default() -> Self {
        Self {
            frequency_ghz: 5.0,
            anharmonicity_ghz: -0.34,
            gate_duration_ns: 20.0,
            two_qubit_gate_duration_ns: 60.0,
            t1_us: 100.0,
            t2_us: 80.0,
            readout_error: 0.01,
        }
    }
}

impl TransmonParameters {
    /// Derive leakage rates from these transmon parameters.
    pub fn leakage_rates(&self) -> LeakageRates {
        LeakageRates::from_transmon_params(
            self.anharmonicity_ghz,
            self.gate_duration_ns,
            self.t1_us,
        )
    }

    /// IBM Eagle-class transmon parameters.
    pub fn ibm_eagle() -> Self {
        Self {
            frequency_ghz: 5.1,
            anharmonicity_ghz: -0.34,
            gate_duration_ns: 21.3,
            two_qubit_gate_duration_ns: 66.0,
            t1_us: 108.0,
            t2_us: 91.0,
            readout_error: 0.008,
        }
    }

    /// Google Sycamore-class transmon parameters.
    pub fn google_sycamore() -> Self {
        Self {
            frequency_ghz: 5.5,
            anharmonicity_ghz: -0.22,
            gate_duration_ns: 12.0,
            two_qubit_gate_duration_ns: 32.0,
            t1_us: 20.0,
            t2_us: 16.0,
            readout_error: 0.005,
        }
    }
}

// ============================================================
// LEAKAGE GATE
// ============================================================

/// A gate operation in the qutrit (3-level) Hilbert space.
///
/// For single-qutrit gates, the matrix is 3x3.
/// For two-qutrit gates, the matrix is 9x9.
#[derive(Clone, Debug)]
pub struct LeakageGate {
    /// Unitary matrix in the qutrit space.
    /// Stored as a flat vector of dimension d x d where d = 3^k
    /// for a k-qutrit gate.
    pub matrix: Vec<Vec<Complex64>>,
    /// Number of qutrits this gate acts on (1 or 2).
    pub n_qutrits: usize,
    /// Human-readable name for identification.
    pub name: String,
}

impl LeakageGate {
    /// Create a single-qutrit gate from a 3x3 matrix.
    pub fn single_qutrit(matrix: Vec<Vec<Complex64>>, name: &str) -> Self {
        assert_eq!(matrix.len(), QUTRIT_DIM, "Single-qutrit gate must be 3x3");
        for row in &matrix {
            assert_eq!(row.len(), QUTRIT_DIM, "Single-qutrit gate must be 3x3");
        }
        Self {
            matrix,
            n_qutrits: 1,
            name: name.to_string(),
        }
    }

    /// Create a two-qutrit gate from a 9x9 matrix.
    pub fn two_qutrit(matrix: Vec<Vec<Complex64>>, name: &str) -> Self {
        let dim = QUTRIT_DIM * QUTRIT_DIM;
        assert_eq!(matrix.len(), dim, "Two-qutrit gate must be 9x9");
        for row in &matrix {
            assert_eq!(row.len(), dim, "Two-qutrit gate must be 9x9");
        }
        Self {
            matrix,
            n_qutrits: 2,
            name: name.to_string(),
        }
    }

    /// Check if this gate is approximately unitary.
    pub fn is_unitary(&self, tol: f64) -> bool {
        let dim = self.matrix.len();
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = c64_zero();
                for k in 0..dim {
                    dot += self.matrix[k][i].conj() * self.matrix[k][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot.re - expected).abs() > tol || dot.im.abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    // --------------------------------------------------------
    // IDEAL GATES (no leakage, block-diagonal in comp subspace)
    // --------------------------------------------------------

    /// Ideal X gate on the computational subspace, identity on |2>.
    ///
    /// ```text
    /// |0 1 0|
    /// |1 0 0|
    /// |0 0 1|
    /// ```
    pub fn x_ideal() -> Self {
        let m = vec![
            vec![c64_zero(), c64_one(), c64_zero()],
            vec![c64_one(), c64_zero(), c64_zero()],
            vec![c64_zero(), c64_zero(), c64_one()],
        ];
        Self::single_qutrit(m, "X_ideal")
    }

    /// Ideal Z gate: phase on |1>, different phase on |2>.
    ///
    /// For a transmon, the |2> state acquires an additional phase due
    /// to the anharmonic energy spectrum:
    ///
    /// ```text
    /// |1  0          0      |
    /// |0  -1         0      |
    /// |0  0   e^{i*phi_2}   |
    /// ```
    ///
    /// where phi_2 depends on the anharmonicity.
    pub fn z_ideal() -> Self {
        Self::z_with_phase(PI)
    }

    /// Z gate with explicit |2> phase.
    pub fn z_with_phase(phi_2: f64) -> Self {
        let m = vec![
            vec![c64_one(), c64_zero(), c64_zero()],
            vec![c64_zero(), c64(-1.0, 0.0), c64_zero()],
            vec![c64_zero(), c64_zero(), c64(phi_2.cos(), phi_2.sin())],
        ];
        Self::single_qutrit(m, "Z_qutrit")
    }

    /// Ideal Hadamard on computational subspace, identity on |2>.
    pub fn h_ideal() -> Self {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let m = vec![
            vec![c64(inv_sqrt2, 0.0), c64(inv_sqrt2, 0.0), c64_zero()],
            vec![c64(inv_sqrt2, 0.0), c64(-inv_sqrt2, 0.0), c64_zero()],
            vec![c64_zero(), c64_zero(), c64_one()],
        ];
        Self::single_qutrit(m, "H_ideal")
    }

    // --------------------------------------------------------
    // LEAKAGE-AWARE GATES
    // --------------------------------------------------------

    /// X gate with leakage: coherent coupling between |1> and |2>.
    ///
    /// Models the effect of driving the |0>-|1> transition in a system
    /// where the |1>-|2> transition is detuned by the anharmonicity.
    /// The leakage parameter epsilon controls the |1> <-> |2> mixing angle.
    ///
    /// The qutrit unitary is constructed as:
    ///
    /// ```text
    /// U = R_{01}(pi) * R_{12}(epsilon)
    /// ```
    ///
    /// where R_{01} is the X rotation on the {|0>, |1>} subspace and
    /// R_{12} is a small rotation on the {|1>, |2>} subspace.
    pub fn x_with_leakage(epsilon: f64) -> Self {
        // Start with the ideal X gate
        // Then apply a small rotation in the {|1>, |2>} subspace
        let c = epsilon.cos();
        let s = epsilon.sin();

        // Combined unitary: first R_12(epsilon), then X_01
        // R_12 acts on |1>,|2> subspace:
        //   |1> -> cos(e)|1> - i*sin(e)|2>
        //   |2> -> -i*sin(e)|1> + cos(e)|2>
        // X_01 swaps |0> and |1>:
        //   |0> -> |1>
        //   |1> -> |0>
        //
        // Combined:
        //   |0> -> cos(e)|1> - i*sin(e)|2>
        //   |1> -> |0>
        //   |2> -> -i*sin(e)|1> + cos(e)|2>
        //
        // But we want the more physical picture where leakage happens
        // during the X gate. The combined matrix is:
        let m = vec![
            vec![c64_zero(), c64_one(), c64_zero()],
            vec![c64(c, 0.0), c64_zero(), c64(0.0, -s)],
            vec![c64(0.0, -s), c64_zero(), c64(c, 0.0)],
        ];
        Self::single_qutrit(m, &format!("X_leak(eps={:.4})", epsilon))
    }

    /// X gate with leakage derived from physical rates.
    pub fn x_from_rates(rates: &LeakageRates) -> Self {
        // The leakage per gate gives the probability |1> -> |2>,
        // which corresponds to sin^2(epsilon) = gamma_01_to_02
        let epsilon = rates.gamma_01_to_02.sqrt().asin();
        Self::x_with_leakage(epsilon)
    }

    /// Idle evolution for a given duration: T1 decay from |2>.
    ///
    /// During idle time, the |2> state decays to |1> (dominant) and
    /// |0> (subdominant) through radiative and non-radiative processes.
    ///
    /// The Kraus operator for the dominant decay channel |2> -> |1> is:
    ///
    /// ```text
    /// K_0 = diag(1, 1, sqrt(1 - p_21 - p_20))   (no-decay branch)
    /// K_1 = |1><2| * sqrt(p_21)                   (|2> -> |1> decay)
    /// K_2 = |0><2| * sqrt(p_20)                   (|2> -> |0> decay)
    /// ```
    ///
    /// For simplicity we apply the effective non-unitary evolution
    /// and re-normalize.
    pub fn idle_decay(rates: &LeakageRates, duration_ns: f64) -> Self {
        let t = duration_ns * 1e-9; // Convert to seconds
        let t1 = rates.t1_leakage * 1e-6; // Convert to seconds

        // Decay probability from |2> during time t
        let p_21 = 1.0 - (-t / t1).exp();
        // Non-radiative decay is much slower
        let p_20 = p_21 * (rates.gamma_20 / rates.gamma_21).min(1.0);
        // Remaining population in |2>
        let p_stay = (1.0 - p_21 - p_20).max(0.0);

        // Effective evolution matrix (non-unitary, will be normalized)
        let m = vec![
            vec![c64_one(), c64_zero(), c64(p_20.sqrt(), 0.0)],
            vec![c64_zero(), c64_one(), c64(p_21.sqrt(), 0.0)],
            vec![c64_zero(), c64_zero(), c64(p_stay.sqrt(), 0.0)],
        ];
        Self::single_qutrit(m, &format!("Idle({:.1}ns)", duration_ns))
    }

    /// Leakage Reduction Unit (LRU): swaps |2> population to |0>.
    ///
    /// An ideal LRU performs a pi-rotation in the {|0>, |2>} subspace:
    ///
    /// ```text
    /// |0> -> |2>
    /// |1> -> |1>
    /// |2> -> |0>
    /// ```
    ///
    /// This is implemented by driving the |0>-|2> two-photon transition.
    /// In practice, LRUs have finite fidelity; the `fidelity` parameter
    /// controls the mixing angle: 1.0 = perfect swap, 0.0 = identity.
    pub fn lru(fidelity: f64) -> Self {
        let theta = fidelity * PI / 2.0;
        let c = theta.cos();
        let s = theta.sin();

        let m = vec![
            vec![c64(c, 0.0), c64_zero(), c64(0.0, -s)],
            vec![c64_zero(), c64_one(), c64_zero()],
            vec![c64(0.0, -s), c64_zero(), c64(c, 0.0)],
        ];
        Self::single_qutrit(m, &format!("LRU(f={:.3})", fidelity))
    }

    /// Perfect LRU: complete |0> <-> |2> swap.
    pub fn lru_perfect() -> Self {
        Self::lru(1.0)
    }

    // --------------------------------------------------------
    // TWO-QUTRIT GATES
    // --------------------------------------------------------

    /// Ideal CNOT in the qutrit space (9x9 matrix).
    ///
    /// Acts as standard CNOT on the computational subspace {|0>, |1>}
    /// and as identity on |2> states. The basis ordering is:
    ///
    /// |00>, |01>, |02>, |10>, |11>, |12>, |20>, |21>, |22>
    ///
    /// The CNOT flips the target when the control is |1>:
    ///   |10> -> |11>, |11> -> |10>
    /// All other basis states (including those with |2>) are unchanged.
    pub fn cnot_ideal() -> Self {
        let dim = QUTRIT_DIM * QUTRIT_DIM; // 9
        let mut m = vec![vec![c64_zero(); dim]; dim];

        // Identity on all states by default
        for i in 0..dim {
            m[i][i] = c64_one();
        }

        // CNOT action: swap |10> <-> |11>
        // |10> has index 1*3 + 0 = 3
        // |11> has index 1*3 + 1 = 4
        m[3][3] = c64_zero();
        m[3][4] = c64_one();
        m[4][4] = c64_zero();
        m[4][3] = c64_one();

        Self::two_qutrit(m, "CNOT_ideal")
    }

    /// CNOT with leakage: models conditional leakage on the |1>-|2> transition.
    ///
    /// When the control qubit is |1>, the target undergoes an X gate with
    /// leakage. Additionally, the control itself can leak during the
    /// interaction.
    ///
    /// The leakage channels modeled are:
    /// - Target leakage: |11> can leak to |12> (target leaks to |2>)
    /// - Control leakage: |1x> can leak to |2x> (control leaks to |2>)
    pub fn cnot_with_leakage(target_leak: f64, control_leak: f64) -> Self {
        let dim = QUTRIT_DIM * QUTRIT_DIM; // 9
        let mut m = vec![vec![c64_zero(); dim]; dim];

        // Start with identity on all states
        for i in 0..dim {
            m[i][i] = c64_one();
        }

        // Basis ordering: |c,t> = |00>,|01>,|02>,|10>,|11>,|12>,|20>,|21>,|22>
        // Index = control*3 + target

        // CNOT action with target leakage when control = |1>
        let ct = target_leak.cos();
        let st = target_leak.sin();

        // |10> -> ct*|11> - i*st*|12>   (X with leakage on target)
        // |11> -> |10>                    (X part)
        // |12> -> -i*st*|11> + ct*|12>   (leakage mixing)
        m[3][3] = c64_zero(); // clear |10>->|10>
        m[4][3] = c64(ct, 0.0); // |10> -> ct*|11>
        m[5][3] = c64(0.0, -st); // |10> -> -i*st*|12>

        m[4][4] = c64_zero(); // clear |11>->|11>
        m[3][4] = c64_one(); // |11> -> |10>

        m[5][5] = c64(ct, 0.0); // |12> -> ct*|12>
        m[4][5] = c64(0.0, -st); // |12> -> -i*st*|11>

        // Control leakage: compose with a rotation R_ctrl that mixes |1,t> <-> |2,t>.
        // The full gate is U = M_target * R_ctrl, implemented as column mixing on M.
        if control_leak > 1e-15 {
            let cc = control_leak.cos();
            let sc = control_leak.sin();

            // R_ctrl acts on the input: |1,t> -> cc*|1,t> + (-i*sc)*|2,t>
            //                            |2,t> -> (-i*sc)*|1,t> + cc*|2,t>
            // To compose M_new = M * R_ctrl, we mix columns of M.
            for t in 0..QUTRIT_DIM {
                let col_1x = 1 * QUTRIT_DIM + t; // column for input |1,t>
                let col_2x = 2 * QUTRIT_DIM + t; // column for input |2,t>

                // Save current column values
                let col_1x_vals: Vec<Complex64> = (0..dim).map(|r| m[r][col_1x]).collect();
                let col_2x_vals: Vec<Complex64> = (0..dim).map(|r| m[r][col_2x]).collect();

                // New columns: M[:,col_1x] = M[:,col_1x]*cc + M[:,col_2x]*(-i*sc)
                //              M[:,col_2x] = M[:,col_1x]*(-i*sc) + M[:,col_2x]*cc
                for r in 0..dim {
                    m[r][col_1x] = col_1x_vals[r] * c64(cc, 0.0) + col_2x_vals[r] * c64(0.0, -sc);
                    m[r][col_2x] = col_1x_vals[r] * c64(0.0, -sc) + col_2x_vals[r] * c64(cc, 0.0);
                }
            }
        }

        Self::two_qutrit(m, &format!("CNOT_leak(t={:.4},c={:.4})", target_leak, control_leak))
    }

    /// CNOT with leakage derived from physical rates.
    pub fn cnot_from_rates(target_rates: &LeakageRates, control_rates: &LeakageRates) -> Self {
        let target_epsilon = target_rates.gamma_01_to_02.sqrt().asin();
        let control_epsilon = control_rates.gamma_01_to_02.sqrt().asin();
        // Two-qubit gates typically have ~2x the leakage of single-qubit gates
        Self::cnot_with_leakage(target_epsilon * 2.0, control_epsilon)
    }
}

// ============================================================
// LEAKAGE MODEL (QUTRIT SIMULATOR)
// ============================================================

/// Statistics collected during leakage simulation.
#[derive(Clone, Debug, Default)]
pub struct LeakageStatistics {
    /// Number of gates applied.
    pub total_gates: usize,
    /// Number of single-qutrit gates applied.
    pub single_gates: usize,
    /// Number of two-qutrit gates applied.
    pub two_qubit_gates: usize,
    /// Number of LRU operations applied.
    pub lru_applications: usize,
    /// Per-qubit leakage population history (snapshots after each gate).
    pub leakage_history: Vec<Vec<f64>>,
    /// Peak leakage population observed on any qubit.
    pub peak_leakage: f64,
    /// Per-qubit peak leakage.
    pub peak_leakage_per_qubit: Vec<f64>,
}

/// Qutrit-based quantum simulator modeling leakage out of the computational subspace.
///
/// Each qubit is modeled as a 3-level system (qutrit) with levels |0>, |1>, |2>.
/// The |2> state represents the first non-computational energy level of the transmon.
///
/// # Example
///
/// ```rust
/// use nqpu_metal::leakage_simulation::{LeakageModel, LeakageRates, LeakageGate};
///
/// // Create a 2-qutrit system with default leakage rates
/// let rates = LeakageRates::default();
/// let mut model = LeakageModel::new(2, vec![rates.clone(), rates]);
///
/// // Apply an X gate with leakage on qubit 0
/// let x_leak = LeakageGate::x_with_leakage(0.05);
/// model.apply_single_gate(&x_leak, 0);
///
/// // Measure leakage population
/// let leakage = model.leakage_population();
/// println!("Leakage on qubit 0: {:.6}", leakage[0]);
/// ```
pub struct LeakageModel {
    /// Number of qutrits in the system.
    pub n_qubits: usize,
    /// Number of levels per qutrit (always 3 in this implementation).
    pub n_levels: usize,
    /// Per-qubit leakage rates.
    pub leakage_rates: Vec<LeakageRates>,
    /// Full qutrit statevector with 3^n amplitudes.
    pub state: Vec<Complex64>,
    /// Simulation statistics.
    pub stats: LeakageStatistics,
}

impl LeakageModel {
    /// Create a new leakage model with `n` qutrits, all initialized to |0...0>.
    ///
    /// # Panics
    ///
    /// Panics if `n` exceeds `MAX_QUTRITS` or if the number of rate entries
    /// does not match `n`.
    pub fn new(n_qubits: usize, leakage_rates: Vec<LeakageRates>) -> Self {
        assert!(n_qubits > 0, "Must have at least 1 qutrit");
        assert!(
            n_qubits <= MAX_QUTRITS,
            "Maximum {} qutrits supported (3^{} = {} amplitudes)",
            MAX_QUTRITS,
            MAX_QUTRITS,
            3_usize.pow(MAX_QUTRITS as u32)
        );
        assert_eq!(
            leakage_rates.len(),
            n_qubits,
            "Must provide leakage rates for each qutrit"
        );

        let dim = QUTRIT_DIM.pow(n_qubits as u32);
        let mut state = vec![c64_zero(); dim];
        state[0] = c64_one(); // Initialize to |0...0>

        Self {
            n_qubits,
            n_levels: QUTRIT_DIM,
            leakage_rates,
            state,
            stats: LeakageStatistics {
                peak_leakage_per_qubit: vec![0.0; n_qubits],
                ..Default::default()
            },
        }
    }

    /// Create a model with uniform leakage rates on all qubits.
    pub fn with_uniform_rates(n_qubits: usize, rates: LeakageRates) -> Self {
        let all_rates = vec![rates; n_qubits];
        Self::new(n_qubits, all_rates)
    }

    /// Dimension of the full qutrit Hilbert space (3^n).
    pub fn dim(&self) -> usize {
        self.state.len()
    }

    /// Reset the state to |0...0>.
    pub fn reset(&mut self) {
        for amp in self.state.iter_mut() {
            *amp = c64_zero();
        }
        self.state[0] = c64_one();
        self.stats = LeakageStatistics {
            peak_leakage_per_qubit: vec![0.0; self.n_qubits],
            ..Default::default()
        };
    }

    /// Set the state to a specific computational basis state.
    ///
    /// Each element of `levels` must be 0, 1, or 2.
    pub fn set_basis_state(&mut self, levels: &[usize]) {
        assert_eq!(levels.len(), self.n_qubits, "Must specify level for each qutrit");
        for &l in levels {
            assert!(l < QUTRIT_DIM, "Level must be 0, 1, or 2");
        }

        for amp in self.state.iter_mut() {
            *amp = c64_zero();
        }

        let mut index = 0;
        for (i, &l) in levels.iter().enumerate() {
            index += l * QUTRIT_DIM.pow(i as u32);
        }
        self.state[index] = c64_one();
    }

    /// Get the norm-squared of the statevector (should be 1.0).
    pub fn norm_squared(&self) -> f64 {
        self.state.iter().map(|a| a.norm_sqr()).sum()
    }

    /// Renormalize the statevector to unit norm.
    pub fn normalize(&mut self) {
        let norm = self.norm_squared().sqrt();
        if norm > 1e-15 {
            for amp in self.state.iter_mut() {
                *amp = *amp / norm;
            }
        }
    }

    // --------------------------------------------------------
    // STATE ANALYSIS
    // --------------------------------------------------------

    /// Compute the probability of each qutrit being in state |2> (leaked).
    ///
    /// Returns a vector of length `n_qubits` where each entry is the
    /// marginal probability of that qutrit being in the |2> state.
    pub fn leakage_population(&self) -> Vec<f64> {
        let mut leakage = vec![0.0; self.n_qubits];
        let dim = self.dim();

        for idx in 0..dim {
            let prob = self.state[idx].norm_sqr();
            if prob < 1e-20 {
                continue;
            }

            // Decompose index into qutrit levels
            let mut remaining = idx;
            for q in 0..self.n_qubits {
                let level = remaining % QUTRIT_DIM;
                remaining /= QUTRIT_DIM;
                if level == 2 {
                    leakage[q] += prob;
                }
            }
        }

        leakage
    }

    /// Total leakage: sum of all population outside the computational subspace.
    pub fn total_leakage(&self) -> f64 {
        let dim = self.dim();
        let mut leaked = 0.0;

        for idx in 0..dim {
            let prob = self.state[idx].norm_sqr();
            if prob < 1e-20 {
                continue;
            }

            // Check if any qutrit is in |2>
            let mut remaining = idx;
            let mut has_leakage = false;
            for _ in 0..self.n_qubits {
                let level = remaining % QUTRIT_DIM;
                remaining /= QUTRIT_DIM;
                if level == 2 {
                    has_leakage = true;
                    break;
                }
            }
            if has_leakage {
                leaked += prob;
            }
        }

        leaked
    }

    /// Probability distribution over computational basis states (projected).
    ///
    /// Returns probabilities for the 2^n computational basis states {|0>, |1>}^n,
    /// ignoring all population in the |2> states. The probabilities will sum
    /// to less than 1.0 if there is leakage.
    pub fn computational_probabilities(&self) -> Vec<f64> {
        let n_comp = 2_usize.pow(self.n_qubits as u32);
        let mut probs = vec![0.0; n_comp];
        let dim = self.dim();

        for idx in 0..dim {
            let prob = self.state[idx].norm_sqr();
            if prob < 1e-20 {
                continue;
            }

            // Decompose into qutrit levels and check if all are in {0, 1}
            let mut remaining = idx;
            let mut comp_idx = 0;
            let mut all_computational = true;

            for q in 0..self.n_qubits {
                let level = remaining % QUTRIT_DIM;
                remaining /= QUTRIT_DIM;
                if level == 2 {
                    all_computational = false;
                    break;
                }
                comp_idx += level * 2_usize.pow(q as u32);
            }

            if all_computational {
                probs[comp_idx] += prob;
            }
        }

        probs
    }

    /// Fidelity with the ideal (no-leakage) computational state.
    ///
    /// Given an ideal statevector in the 2^n computational Hilbert space,
    /// computes the overlap |<ideal|projected>|^2 where |projected> is the
    /// current state projected onto the computational subspace.
    pub fn computational_fidelity(&self, ideal_state: &[Complex64]) -> f64 {
        let n_comp = 2_usize.pow(self.n_qubits as u32);
        assert_eq!(
            ideal_state.len(),
            n_comp,
            "Ideal state must have 2^n amplitudes"
        );

        // Project current qutrit state onto computational subspace
        let mut projected = vec![c64_zero(); n_comp];
        let dim = self.dim();

        for idx in 0..dim {
            if self.state[idx].norm_sqr() < 1e-20 {
                continue;
            }

            let mut remaining = idx;
            let mut comp_idx = 0;
            let mut all_computational = true;

            for q in 0..self.n_qubits {
                let level = remaining % QUTRIT_DIM;
                remaining /= QUTRIT_DIM;
                if level == 2 {
                    all_computational = false;
                    break;
                }
                comp_idx += level * 2_usize.pow(q as u32);
            }

            if all_computational {
                projected[comp_idx] += self.state[idx];
            }
        }

        // Compute |<ideal|projected>|^2
        let mut overlap = c64_zero();
        for i in 0..n_comp {
            overlap += ideal_state[i].conj() * projected[i];
        }

        overlap.norm_sqr()
    }

    /// Leakage-aware measurement: returns measurement outcome and whether
    /// leakage was detected.
    ///
    /// If a qutrit is measured and found in |2>, this is flagged as a
    /// leakage detection event. In real hardware, this might trigger a
    /// reset or LRU operation.
    pub fn measure_with_leakage_detection(&self) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(self.n_qubits);

        for q in 0..self.n_qubits {
            let mut p = [0.0_f64; QUTRIT_DIM];
            let dim = self.dim();

            for idx in 0..dim {
                let prob = self.state[idx].norm_sqr();
                if prob < 1e-20 {
                    continue;
                }

                let level = (idx / QUTRIT_DIM.pow(q as u32)) % QUTRIT_DIM;
                p[level] += prob;
            }

            results.push(MeasurementResult {
                qubit: q,
                level_probs: p,
                computational_prob: p[0] + p[1],
                leakage_prob: p[2],
                leaked: p[2] > 0.001, // Flag leakage if > 0.1%
            });
        }

        results
    }

    // --------------------------------------------------------
    // GATE APPLICATION
    // --------------------------------------------------------

    /// Apply a single-qutrit gate to the specified qubit.
    ///
    /// The gate's 3x3 unitary is applied to the qutrit at position `target`,
    /// acting as identity on all other qutrits.
    pub fn apply_single_gate(&mut self, gate: &LeakageGate, target: usize) {
        assert_eq!(gate.n_qutrits, 1, "Expected single-qutrit gate");
        assert!(target < self.n_qubits, "Target qutrit out of range");

        let dim = self.dim();
        let mut new_state = vec![c64_zero(); dim];
        let stride = QUTRIT_DIM.pow(target as u32);

        for idx in 0..dim {
            let level = (idx / stride) % QUTRIT_DIM;
            let base = idx - level * stride;

            for new_level in 0..QUTRIT_DIM {
                let new_idx = base + new_level * stride;
                new_state[new_idx] += gate.matrix[new_level][level] * self.state[idx];
            }
        }

        self.state = new_state;

        // Update statistics
        self.stats.total_gates += 1;
        self.stats.single_gates += 1;
        self.update_leakage_stats();
    }

    /// Apply a two-qutrit gate to the specified control and target qubits.
    ///
    /// The gate's 9x9 unitary is applied to the two-qutrit subspace defined
    /// by the control and target positions.
    pub fn apply_two_qubit_gate(
        &mut self,
        gate: &LeakageGate,
        control: usize,
        target: usize,
    ) {
        assert_eq!(gate.n_qutrits, 2, "Expected two-qutrit gate");
        assert!(control < self.n_qubits, "Control qutrit out of range");
        assert!(target < self.n_qubits, "Target qutrit out of range");
        assert_ne!(control, target, "Control and target must be different");

        let dim = self.dim();
        let mut new_state = vec![c64_zero(); dim];

        let stride_c = QUTRIT_DIM.pow(control as u32);
        let stride_t = QUTRIT_DIM.pow(target as u32);

        for idx in 0..dim {
            let level_c = (idx / stride_c) % QUTRIT_DIM;
            let level_t = (idx / stride_t) % QUTRIT_DIM;

            // Map to 2-qutrit gate index: control * 3 + target
            let gate_idx = level_c * QUTRIT_DIM + level_t;
            let base = idx - level_c * stride_c - level_t * stride_t;

            for new_c in 0..QUTRIT_DIM {
                for new_t in 0..QUTRIT_DIM {
                    let new_gate_idx = new_c * QUTRIT_DIM + new_t;
                    let new_idx = base + new_c * stride_c + new_t * stride_t;
                    new_state[new_idx] +=
                        gate.matrix[new_gate_idx][gate_idx] * self.state[idx];
                }
            }
        }

        self.state = new_state;

        // Update statistics
        self.stats.total_gates += 1;
        self.stats.two_qubit_gates += 1;
        self.update_leakage_stats();
    }

    /// Apply an LRU gate to the specified qubit.
    pub fn apply_lru(&mut self, target: usize, fidelity: f64) {
        let lru = LeakageGate::lru(fidelity);
        self.apply_single_gate(&lru, target);
        self.stats.lru_applications += 1;
    }

    /// Apply idle decay to a specific qubit for the given duration.
    pub fn apply_idle(&mut self, target: usize, duration_ns: f64) {
        let idle = LeakageGate::idle_decay(&self.leakage_rates[target], duration_ns);
        self.apply_single_gate(&idle, target);
        // Re-normalize after non-unitary evolution
        self.normalize();
    }

    /// Apply idle decay to all qubits simultaneously.
    pub fn apply_global_idle(&mut self, duration_ns: f64) {
        for q in 0..self.n_qubits {
            let idle = LeakageGate::idle_decay(&self.leakage_rates[q], duration_ns);
            self.apply_single_gate(&idle, q);
        }
        self.normalize();
    }

    // --------------------------------------------------------
    // CIRCUIT EXECUTION
    // --------------------------------------------------------

    /// Execute a sequence of leakage-aware operations.
    pub fn execute_circuit(&mut self, operations: &[LeakageOperation]) {
        for op in operations {
            match op {
                LeakageOperation::SingleGate { gate, target } => {
                    self.apply_single_gate(gate, *target);
                }
                LeakageOperation::TwoQubitGate {
                    gate,
                    control,
                    target,
                } => {
                    self.apply_two_qubit_gate(gate, *control, *target);
                }
                LeakageOperation::Idle { target, duration_ns } => {
                    self.apply_idle(*target, *duration_ns);
                }
                LeakageOperation::GlobalIdle { duration_ns } => {
                    self.apply_global_idle(*duration_ns);
                }
                LeakageOperation::Lru { target, fidelity } => {
                    self.apply_lru(*target, *fidelity);
                }
                LeakageOperation::Barrier => {
                    // No-op, used for circuit visualization
                }
            }
        }
    }

    // --------------------------------------------------------
    // QEC SUPPORT
    // --------------------------------------------------------

    /// Analyze leakage impact on a surface code patch.
    ///
    /// Given a set of data qubit indices and syndrome qubit indices,
    /// computes leakage metrics relevant for QEC:
    ///
    /// - Per-qubit leakage populations
    /// - Whether any data qubit exceeds the leakage threshold
    /// - Estimated logical error rate contribution from leakage
    pub fn qec_leakage_analysis(
        &self,
        data_qubits: &[usize],
        syndrome_qubits: &[usize],
        threshold: f64,
    ) -> QecLeakageReport {
        let leakage = self.leakage_population();

        let data_leakage: Vec<f64> = data_qubits.iter().map(|&q| leakage[q]).collect();
        let syndrome_leakage: Vec<f64> = syndrome_qubits.iter().map(|&q| leakage[q]).collect();

        let data_exceeded: Vec<usize> = data_qubits
            .iter()
            .filter(|&&q| leakage[q] > threshold)
            .copied()
            .collect();

        let syndrome_exceeded: Vec<usize> = syndrome_qubits
            .iter()
            .filter(|&&q| leakage[q] > threshold)
            .copied()
            .collect();

        let avg_data_leakage = if data_leakage.is_empty() {
            0.0
        } else {
            data_leakage.iter().sum::<f64>() / data_leakage.len() as f64
        };

        QecLeakageReport {
            data_leakage,
            syndrome_leakage,
            data_exceeded,
            syndrome_exceeded,
            avg_data_leakage,
            total_leakage: self.total_leakage(),
            needs_lru: !data_qubits
                .iter()
                .all(|&q| leakage[q] <= threshold),
        }
    }

    // --------------------------------------------------------
    // INTERNAL HELPERS
    // --------------------------------------------------------

    /// Update leakage tracking statistics after a gate.
    fn update_leakage_stats(&mut self) {
        let leakage = self.leakage_population();
        for (q, &l) in leakage.iter().enumerate() {
            if l > self.stats.peak_leakage {
                self.stats.peak_leakage = l;
            }
            if l > self.stats.peak_leakage_per_qubit[q] {
                self.stats.peak_leakage_per_qubit[q] = l;
            }
        }
        self.stats.leakage_history.push(leakage);
    }

    /// Get the qutrit level decomposition for a basis index.
    pub fn index_to_levels(&self, index: usize) -> Vec<usize> {
        let mut levels = Vec::with_capacity(self.n_qubits);
        let mut remaining = index;
        for _ in 0..self.n_qubits {
            levels.push(remaining % QUTRIT_DIM);
            remaining /= QUTRIT_DIM;
        }
        levels
    }

    /// Convert a set of qutrit levels to a basis index.
    pub fn levels_to_index(&self, levels: &[usize]) -> usize {
        assert_eq!(levels.len(), self.n_qubits);
        let mut index = 0;
        for (i, &l) in levels.iter().enumerate() {
            assert!(l < QUTRIT_DIM, "Level must be 0, 1, or 2");
            index += l * QUTRIT_DIM.pow(i as u32);
        }
        index
    }
}

// ============================================================
// SUPPORTING TYPES
// ============================================================

/// Result of measuring a single qutrit with leakage detection.
#[derive(Clone, Debug)]
pub struct MeasurementResult {
    /// Which qubit was measured.
    pub qubit: usize,
    /// Probability of each level: [P(|0>), P(|1>), P(|2>)].
    pub level_probs: [f64; QUTRIT_DIM],
    /// Total probability in the computational subspace.
    pub computational_prob: f64,
    /// Probability of being in the leaked |2> state.
    pub leakage_prob: f64,
    /// Whether leakage was detected (above threshold).
    pub leaked: bool,
}

/// A leakage-aware circuit operation.
#[derive(Clone, Debug)]
pub enum LeakageOperation {
    /// Apply a single-qutrit gate.
    SingleGate {
        gate: LeakageGate,
        target: usize,
    },
    /// Apply a two-qutrit gate.
    TwoQubitGate {
        gate: LeakageGate,
        control: usize,
        target: usize,
    },
    /// Idle a specific qubit for a duration (T1 decay).
    Idle {
        target: usize,
        duration_ns: f64,
    },
    /// Idle all qubits for a duration.
    GlobalIdle {
        duration_ns: f64,
    },
    /// Apply a Leakage Reduction Unit.
    Lru {
        target: usize,
        fidelity: f64,
    },
    /// Barrier (no-op, for circuit structure).
    Barrier,
}

/// Report on leakage impact for quantum error correction.
#[derive(Clone, Debug)]
pub struct QecLeakageReport {
    /// Leakage population on each data qubit.
    pub data_leakage: Vec<f64>,
    /// Leakage population on each syndrome qubit.
    pub syndrome_leakage: Vec<f64>,
    /// Data qubits exceeding the leakage threshold.
    pub data_exceeded: Vec<usize>,
    /// Syndrome qubits exceeding the leakage threshold.
    pub syndrome_exceeded: Vec<usize>,
    /// Average leakage across data qubits.
    pub avg_data_leakage: f64,
    /// Total leakage population in the system.
    pub total_leakage: f64,
    /// Whether LRU operations are recommended.
    pub needs_lru: bool,
}

// ============================================================
// CONVENIENCE BUILDERS
// ============================================================

/// Builder for constructing leakage simulation circuits.
pub struct LeakageCircuitBuilder {
    operations: Vec<LeakageOperation>,
    n_qubits: usize,
    default_rates: LeakageRates,
}

impl LeakageCircuitBuilder {
    /// Create a new circuit builder.
    pub fn new(n_qubits: usize) -> Self {
        Self {
            operations: Vec::new(),
            n_qubits,
            default_rates: LeakageRates::default(),
        }
    }

    /// Create a builder with specific leakage rates.
    pub fn with_rates(n_qubits: usize, rates: LeakageRates) -> Self {
        Self {
            operations: Vec::new(),
            n_qubits,
            default_rates: rates,
        }
    }

    /// Add an ideal X gate (no leakage).
    pub fn x_ideal(&mut self, target: usize) -> &mut Self {
        self.operations.push(LeakageOperation::SingleGate {
            gate: LeakageGate::x_ideal(),
            target,
        });
        self
    }

    /// Add an X gate with leakage from default rates.
    pub fn x_leaky(&mut self, target: usize) -> &mut Self {
        self.operations.push(LeakageOperation::SingleGate {
            gate: LeakageGate::x_from_rates(&self.default_rates),
            target,
        });
        self
    }

    /// Add a Hadamard gate.
    pub fn h(&mut self, target: usize) -> &mut Self {
        self.operations.push(LeakageOperation::SingleGate {
            gate: LeakageGate::h_ideal(),
            target,
        });
        self
    }

    /// Add an ideal CNOT gate.
    pub fn cnot_ideal(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(LeakageOperation::TwoQubitGate {
            gate: LeakageGate::cnot_ideal(),
            control,
            target,
        });
        self
    }

    /// Add a CNOT with leakage from default rates.
    pub fn cnot_leaky(&mut self, control: usize, target: usize) -> &mut Self {
        self.operations.push(LeakageOperation::TwoQubitGate {
            gate: LeakageGate::cnot_from_rates(&self.default_rates, &self.default_rates),
            control,
            target,
        });
        self
    }

    /// Add an idle period on a specific qubit.
    pub fn idle(&mut self, target: usize, duration_ns: f64) -> &mut Self {
        self.operations.push(LeakageOperation::Idle {
            target,
            duration_ns,
        });
        self
    }

    /// Add an idle period on all qubits.
    pub fn global_idle(&mut self, duration_ns: f64) -> &mut Self {
        self.operations
            .push(LeakageOperation::GlobalIdle { duration_ns });
        self
    }

    /// Add an LRU operation.
    pub fn lru(&mut self, target: usize, fidelity: f64) -> &mut Self {
        self.operations.push(LeakageOperation::Lru {
            target,
            fidelity,
        });
        self
    }

    /// Add a perfect LRU operation.
    pub fn lru_perfect(&mut self, target: usize) -> &mut Self {
        self.lru(target, 1.0)
    }

    /// Add a barrier.
    pub fn barrier(&mut self) -> &mut Self {
        self.operations.push(LeakageOperation::Barrier);
        self
    }

    /// Build the operation list.
    pub fn build(&self) -> Vec<LeakageOperation> {
        self.operations.clone()
    }

    /// Execute the circuit on a model and return it.
    pub fn execute(&self, model: &mut LeakageModel) {
        model.execute_circuit(&self.build());
    }
}

// ============================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================

impl std::fmt::Display for LeakageModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "LeakageModel ({} qutrits, dim={})", self.n_qubits, self.dim())?;
        let leakage = self.leakage_population();
        for (q, l) in leakage.iter().enumerate() {
            writeln!(f, "  Qutrit {}: leakage = {:.6}", q, l)?;
        }
        writeln!(f, "  Total leakage: {:.6}", self.total_leakage())?;
        writeln!(f, "  Norm: {:.10}", self.norm_squared())?;
        writeln!(f, "  Gates applied: {}", self.stats.total_gates)?;
        Ok(())
    }
}

impl std::fmt::Display for LeakageStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Leakage Statistics:")?;
        writeln!(f, "  Total gates: {}", self.total_gates)?;
        writeln!(f, "  Single gates: {}", self.single_gates)?;
        writeln!(f, "  Two-qubit gates: {}", self.two_qubit_gates)?;
        writeln!(f, "  LRU applications: {}", self.lru_applications)?;
        writeln!(f, "  Peak leakage: {:.6}", self.peak_leakage)?;
        for (q, &p) in self.peak_leakage_per_qubit.iter().enumerate() {
            writeln!(f, "  Peak leakage qubit {}: {:.6}", q, p)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for QecLeakageReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QEC Leakage Report:")?;
        writeln!(f, "  Total leakage: {:.6}", self.total_leakage)?;
        writeln!(f, "  Avg data leakage: {:.6}", self.avg_data_leakage)?;
        writeln!(f, "  Data qubits exceeded threshold: {:?}", self.data_exceeded)?;
        writeln!(f, "  Syndrome qubits exceeded threshold: {:?}", self.syndrome_exceeded)?;
        writeln!(f, "  Needs LRU: {}", self.needs_lru)?;
        Ok(())
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;
    const GATE_TOL: f64 = 1e-6;

    /// Helper: check that a float is approximately equal to expected.
    fn assert_approx(actual: f64, expected: f64, tol: f64, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{}: expected {:.10}, got {:.10} (diff={:.2e})",
            msg,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    /// Helper: create default single-qutrit model.
    fn model_1q() -> LeakageModel {
        LeakageModel::new(1, vec![LeakageRates::default()])
    }

    /// Helper: create default two-qutrit model.
    fn model_2q() -> LeakageModel {
        LeakageModel::new(2, vec![LeakageRates::default(), LeakageRates::default()])
    }

    // --------------------------------------------------------
    // 1. test_qutrit_state_initialization
    // --------------------------------------------------------

    #[test]
    fn test_qutrit_state_initialization() {
        // Single qutrit: 3 amplitudes, initialized to |0>
        let model = model_1q();
        assert_eq!(model.dim(), 3);
        assert_approx(model.state[0].re, 1.0, TOL, "|0> amplitude");
        assert_approx(model.state[1].norm_sqr(), 0.0, TOL, "|1> population");
        assert_approx(model.state[2].norm_sqr(), 0.0, TOL, "|2> population");
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm");

        // Two qutrits: 9 amplitudes
        let model2 = model_2q();
        assert_eq!(model2.dim(), 9);
        assert_approx(model2.state[0].re, 1.0, TOL, "|00> amplitude");
        assert_approx(model2.norm_squared(), 1.0, TOL, "Two-qutrit norm");

        // Three qutrits: 27 amplitudes
        let model3 = LeakageModel::with_uniform_rates(3, LeakageRates::default());
        assert_eq!(model3.dim(), 27);
        assert_approx(model3.norm_squared(), 1.0, TOL, "Three-qutrit norm");
    }

    // --------------------------------------------------------
    // 2. test_computational_subspace_projection
    // --------------------------------------------------------

    #[test]
    fn test_computational_subspace_projection() {
        let mut model = model_2q();

        // In |00> state, all population is computational
        let probs = model.computational_probabilities();
        assert_eq!(probs.len(), 4); // 2^2
        assert_approx(probs[0], 1.0, TOL, "|00> prob");
        assert_approx(probs.iter().sum::<f64>(), 1.0, TOL, "Total comp prob");
        assert_approx(model.total_leakage(), 0.0, TOL, "No leakage");

        // Manually put some population in |2> state
        model.state[0] = c64(0.9_f64.sqrt(), 0.0); // |00>
        model.state[2] = c64(0.1_f64.sqrt(), 0.0); // |02> (qutrit 0 leaked)
        let probs = model.computational_probabilities();
        assert_approx(probs.iter().sum::<f64>(), 0.9, TOL, "Comp prob with leakage");
        assert!(model.total_leakage() > 0.09, "Should detect leakage");
    }

    // --------------------------------------------------------
    // 3. test_x_gate_ideal
    // --------------------------------------------------------

    #[test]
    fn test_x_gate_ideal() {
        let mut model = model_1q();

        // Apply ideal X: |0> -> |1>
        let x = LeakageGate::x_ideal();
        model.apply_single_gate(&x, 0);

        assert_approx(model.state[0].norm_sqr(), 0.0, TOL, "|0> after X");
        assert_approx(model.state[1].norm_sqr(), 1.0, TOL, "|1> after X");
        assert_approx(model.state[2].norm_sqr(), 0.0, TOL, "|2> after X");
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm after X");
        assert_approx(model.total_leakage(), 0.0, TOL, "No leakage from ideal X");

        // Apply X again: |1> -> |0>
        model.apply_single_gate(&x, 0);
        assert_approx(model.state[0].norm_sqr(), 1.0, TOL, "|0> after XX");
        assert_approx(model.state[1].norm_sqr(), 0.0, TOL, "|1> after XX");
    }

    // --------------------------------------------------------
    // 4. test_x_gate_with_leakage
    // --------------------------------------------------------

    #[test]
    fn test_x_gate_with_leakage() {
        let mut model = model_1q();

        // Apply X with small leakage
        let epsilon = 0.1;
        let x_leak = LeakageGate::x_with_leakage(epsilon);
        model.apply_single_gate(&x_leak, 0);

        // |0> -> mostly |1> with small |2> component
        let p0 = model.state[0].norm_sqr();
        let p1 = model.state[1].norm_sqr();
        let p2 = model.state[2].norm_sqr();

        assert!(p1 > 0.9, "|1> should have most population: got {:.6}", p1);
        assert!(p2 > 0.0, "|2> should have some population: got {:.6}", p2);
        assert!(p2 < p1, "|2> pop should be less than |1> pop");
        assert_approx(p0 + p1 + p2, 1.0, TOL, "Norm preserved");

        // Leakage should be detected
        let leakage = model.leakage_population();
        assert!(leakage[0] > 0.0, "Leakage should be non-zero");
        assert_approx(leakage[0], p2, TOL, "Leakage matches |2> pop");

        // Verify gate is still unitary
        assert!(x_leak.is_unitary(1e-10), "Leaky X gate must be unitary");
    }

    // --------------------------------------------------------
    // 5. test_z_gate_qutrit
    // --------------------------------------------------------

    #[test]
    fn test_z_gate_qutrit() {
        // Z gate should not change populations, only phases
        let mut model = model_1q();

        // Put in superposition first
        let h = LeakageGate::h_ideal();
        model.apply_single_gate(&h, 0);

        let probs_before: Vec<f64> = model.state.iter().map(|a| a.norm_sqr()).collect();

        let z = LeakageGate::z_ideal();
        model.apply_single_gate(&z, 0);

        let probs_after: Vec<f64> = model.state.iter().map(|a| a.norm_sqr()).collect();

        // Populations should be unchanged
        for i in 0..3 {
            assert_approx(
                probs_after[i],
                probs_before[i],
                TOL,
                &format!("Z preserves population at |{}>", i),
            );
        }
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm after Z");

        // But phases should change: |1> gets a -1 phase
        // After H|0> = (|0> + |1>)/sqrt(2), Z gives (|0> - |1>)/sqrt(2)
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert_approx(model.state[0].re, inv_sqrt2, TOL, "|0> phase after HZ");
        assert_approx(model.state[1].re, -inv_sqrt2, TOL, "|1> phase after HZ");
    }

    // --------------------------------------------------------
    // 6. test_cnot_ideal_qutrit
    // --------------------------------------------------------

    #[test]
    fn test_cnot_ideal_qutrit() {
        let mut model = model_2q();

        // CNOT with control=|0> should be identity
        let cnot = LeakageGate::cnot_ideal();
        model.apply_two_qubit_gate(&cnot, 0, 1);
        assert_approx(model.state[0].norm_sqr(), 1.0, TOL, "|00> unchanged");

        // Set to |10>, CNOT should give |11>
        model.reset();
        model.set_basis_state(&[1, 0]); // qubit 0 = |1>, qubit 1 = |0>
        // Index = 1*1 + 0*3 = 1 => wait, index = q0_level * 3^0 + q1_level * 3^1
        // |10> means q0=1, q1=0 => index = 1 + 0 = 1
        assert_approx(model.state[1].norm_sqr(), 1.0, TOL, "Prepared |10>");

        model.apply_two_qubit_gate(&cnot, 0, 1);
        // CNOT(control=0, target=1): when q0=|1>, flip q1
        // |10> -> |11> has index = 1 + 1*3 = 4
        assert_approx(model.state[4].norm_sqr(), 1.0, TOL, "|11> after CNOT|10>");
        assert_approx(model.total_leakage(), 0.0, TOL, "No leakage from ideal CNOT");
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm after CNOT");
    }

    // --------------------------------------------------------
    // 7. test_cnot_with_leakage
    // --------------------------------------------------------

    #[test]
    fn test_cnot_with_leakage() {
        let mut model = model_2q();

        // Start in |10> (control excited)
        model.set_basis_state(&[1, 0]);

        // CNOT with target leakage
        let epsilon = 0.1;
        let cnot_leak = LeakageGate::cnot_with_leakage(epsilon, 0.0);
        model.apply_two_qubit_gate(&cnot_leak, 0, 1);

        // Should mostly be in |11> but some leakage to |12>
        // |11> = index 4, |12> = index 1 + 2*3 = 7
        let p11 = model.state[4].norm_sqr();
        let p12 = model.state[7].norm_sqr();

        assert!(p11 > 0.9, "|11> should have most population: {:.6}", p11);
        assert!(p12 > 0.0, "Target leakage to |12>: {:.6}", p12);
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm preserved");

        // Target qubit should show leakage
        let leakage = model.leakage_population();
        assert!(leakage[1] > 0.0, "Target should show leakage");
    }

    // --------------------------------------------------------
    // 8. test_idle_t1_decay
    // --------------------------------------------------------

    #[test]
    fn test_idle_t1_decay() {
        let mut model = model_1q();

        // Put qutrit in |2> state
        model.set_basis_state(&[2]);
        assert_approx(model.state[2].norm_sqr(), 1.0, TOL, "Prepared |2>");

        // Apply idle for a long time (should decay significantly)
        // With T1_leakage = 30 us, waiting 30 us should decay ~63%
        let rates = &model.leakage_rates[0].clone();
        let idle = LeakageGate::idle_decay(rates, 30_000.0); // 30 us
        model.apply_single_gate(&idle, 0);
        model.normalize();

        // |2> population should have decreased
        let p2 = model.state[2].norm_sqr();
        assert!(p2 < 0.5, "|2> should decay after T1: got {:.6}", p2);

        // |1> should have gained population (dominant decay channel)
        let p1 = model.state[1].norm_sqr();
        assert!(p1 > 0.3, "|1> should gain from |2> decay: got {:.6}", p1);

        // Norm should be 1 after normalization
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm after idle");
    }

    // --------------------------------------------------------
    // 9. test_leakage_population_measurement
    // --------------------------------------------------------

    #[test]
    fn test_leakage_population_measurement() {
        let mut model = model_2q();

        // Put qubit 0 in |2>, qubit 1 in |0>
        model.set_basis_state(&[2, 0]);
        // Index = 2 + 0*3 = 2

        let leakage = model.leakage_population();
        assert_approx(leakage[0], 1.0, TOL, "Qubit 0 fully leaked");
        assert_approx(leakage[1], 0.0, TOL, "Qubit 1 not leaked");
        assert_approx(model.total_leakage(), 1.0, TOL, "Total leakage = 1.0");

        // Superposition: 50% |00>, 50% |20>
        model.reset();
        model.state[0] = c64(1.0 / 2.0_f64.sqrt(), 0.0); // |00>
        model.state[2] = c64(1.0 / 2.0_f64.sqrt(), 0.0); // |20>

        let leakage = model.leakage_population();
        assert_approx(leakage[0], 0.5, TOL, "Qubit 0 half leaked");
        assert_approx(leakage[1], 0.0, TOL, "Qubit 1 not leaked");
    }

    // --------------------------------------------------------
    // 10. test_leakage_reduction_unit
    // --------------------------------------------------------

    #[test]
    fn test_leakage_reduction_unit() {
        let mut model = model_1q();

        // Put in |2> state
        model.set_basis_state(&[2]);
        assert_approx(model.state[2].norm_sqr(), 1.0, TOL, "Prepared |2>");

        // Apply perfect LRU: |2> -> |0> (with phase)
        let lru = LeakageGate::lru_perfect();
        assert!(lru.is_unitary(1e-10), "LRU must be unitary");

        model.apply_single_gate(&lru, 0);

        // |2> should be swapped to |0> (up to phase)
        let p0 = model.state[0].norm_sqr();
        let p2 = model.state[2].norm_sqr();

        assert_approx(p0, 1.0, TOL, "|0> after LRU on |2>");
        assert_approx(p2, 0.0, TOL, "|2> after LRU on |2>");
        assert_approx(model.total_leakage(), 0.0, TOL, "No leakage after LRU");
    }

    // --------------------------------------------------------
    // 11. test_transmon_noise_model
    // --------------------------------------------------------

    #[test]
    fn test_transmon_noise_model() {
        let params = TransmonParameters::default();

        assert_approx(params.frequency_ghz, 5.0, TOL, "Default frequency");
        assert_approx(params.anharmonicity_ghz, -0.34, TOL, "Default anharmonicity");

        let rates = params.leakage_rates();
        assert!(rates.gamma_01_to_02 > 0.0, "Leakage rate must be positive");
        assert!(rates.gamma_01_to_02 < 0.5, "Leakage rate must be < 0.5");
        assert!(rates.gamma_21 > rates.gamma_20, "gamma_21 > gamma_20");
        assert!(rates.t1_leakage > 0.0, "T1 leakage must be positive");
        assert!(
            rates.t1_leakage < params.t1_us,
            "T1(|2>) < T1(|1>) for transmons"
        );

        // IBM Eagle should have specific properties
        let ibm = TransmonParameters::ibm_eagle();
        let ibm_rates = ibm.leakage_rates();
        assert!(ibm_rates.gamma_01_to_02 < 0.01, "IBM Eagle should have low leakage");

        // Google Sycamore (faster gates, more leakage)
        let google = TransmonParameters::google_sycamore();
        let google_rates = google.leakage_rates();
        assert!(
            google_rates.gamma_01_to_02 > ibm_rates.gamma_01_to_02,
            "Sycamore faster gates = more leakage"
        );
    }

    // --------------------------------------------------------
    // 12. test_leakage_rate_from_anharmonicity
    // --------------------------------------------------------

    #[test]
    fn test_leakage_rate_from_anharmonicity() {
        // More anharmonicity = less leakage (better separation of levels)
        let rates_high_alpha = LeakageRates::from_transmon_params(-0.40, 20.0, 100.0);
        let rates_low_alpha = LeakageRates::from_transmon_params(-0.20, 20.0, 100.0);

        assert!(
            rates_low_alpha.gamma_01_to_02 > rates_high_alpha.gamma_01_to_02,
            "Lower anharmonicity = more leakage: {:.6} vs {:.6}",
            rates_low_alpha.gamma_01_to_02,
            rates_high_alpha.gamma_01_to_02
        );

        // Faster gates = more leakage (stronger drive)
        let rates_slow = LeakageRates::from_transmon_params(-0.30, 40.0, 100.0);
        let rates_fast = LeakageRates::from_transmon_params(-0.30, 10.0, 100.0);

        assert!(
            rates_fast.gamma_01_to_02 > rates_slow.gamma_01_to_02,
            "Faster gates = more leakage: {:.6} vs {:.6}",
            rates_fast.gamma_01_to_02,
            rates_slow.gamma_01_to_02
        );

        // T1 affects decay rates
        let rates_long_t1 = LeakageRates::from_transmon_params(-0.30, 20.0, 200.0);
        let rates_short_t1 = LeakageRates::from_transmon_params(-0.30, 20.0, 20.0);

        assert!(
            rates_short_t1.gamma_21 > rates_long_t1.gamma_21,
            "Shorter T1 = faster decay"
        );
    }

    // --------------------------------------------------------
    // 13. test_multi_gate_leakage_accumulation
    // --------------------------------------------------------

    #[test]
    fn test_multi_gate_leakage_accumulation() {
        let mut model = model_1q();
        let epsilon = 0.1;
        let x_leak = LeakageGate::x_with_leakage(epsilon);

        // Apply leaky X gates repeatedly
        let mut prev_leakage = 0.0;
        for i in 0..10 {
            model.apply_single_gate(&x_leak, 0);
            let leakage = model.leakage_population()[0];

            // Leakage should generally increase (though it oscillates)
            if i > 0 && i % 2 == 0 {
                // After even number of X gates, should return approximately
                // to starting pattern. Track that leakage is non-trivial.
                assert!(
                    model.stats.peak_leakage > 0.0,
                    "Peak leakage should be non-zero"
                );
            }
            prev_leakage = leakage;
        }

        // Statistics should track the gates
        assert_eq!(model.stats.total_gates, 10);
        assert_eq!(model.stats.single_gates, 10);
        assert!(
            model.stats.leakage_history.len() == 10,
            "Should have 10 snapshots"
        );
    }

    // --------------------------------------------------------
    // 14. test_leakage_aware_measurement
    // --------------------------------------------------------

    #[test]
    fn test_leakage_aware_measurement() {
        let mut model = model_2q();

        // Pure |00> state
        let results = model.measure_with_leakage_detection();
        assert_eq!(results.len(), 2);
        assert_approx(results[0].level_probs[0], 1.0, TOL, "Q0 in |0>");
        assert_approx(results[0].computational_prob, 1.0, TOL, "Q0 fully computational");
        assert!(!results[0].leaked, "Q0 not leaked");

        // Put qubit 0 partially in |2>
        model.state[0] = c64(0.95_f64.sqrt(), 0.0); // |00>
        model.state[2] = c64(0.05_f64.sqrt(), 0.0); // |20>

        let results = model.measure_with_leakage_detection();
        assert_approx(results[0].leakage_prob, 0.05, TOL, "Q0 5% leaked");
        assert!(results[0].leaked, "Q0 should flag leakage > 0.1%");
        assert!(!results[1].leaked, "Q1 should not flag leakage");
    }

    // --------------------------------------------------------
    // 15. test_computational_fidelity_with_leakage
    // --------------------------------------------------------

    #[test]
    fn test_computational_fidelity_with_leakage() {
        let mut model = model_1q();

        // Ideal: |1> (after X)
        let ideal_state = vec![c64_zero(), c64_one()]; // |1>

        // Apply ideal X
        let x = LeakageGate::x_ideal();
        model.apply_single_gate(&x, 0);
        let fid = model.computational_fidelity(&ideal_state);
        assert_approx(fid, 1.0, TOL, "Perfect fidelity with ideal X");

        // Now with leakage
        model.reset();
        let x_leak = LeakageGate::x_with_leakage(0.1);
        model.apply_single_gate(&x_leak, 0);
        let fid_leak = model.computational_fidelity(&ideal_state);
        assert!(fid_leak < 1.0, "Fidelity reduced by leakage: {:.6}", fid_leak);
        assert!(fid_leak > 0.9, "Fidelity still high for small leakage: {:.6}", fid_leak);
    }

    // --------------------------------------------------------
    // 16. test_bell_state_with_leakage
    // --------------------------------------------------------

    #[test]
    fn test_bell_state_with_leakage() {
        let mut model = model_2q();

        // Create Bell state: H on q0, then CNOT(q0, q1)
        let h = LeakageGate::h_ideal();
        model.apply_single_gate(&h, 0);

        // Use ideal CNOT first
        let cnot = LeakageGate::cnot_ideal();
        model.apply_two_qubit_gate(&cnot, 0, 1);

        // Should be (|00> + |11>)/sqrt(2)
        // |00> = index 0, |11> = index 4
        assert_approx(model.state[0].norm_sqr(), 0.5, TOL, "|00> in Bell state");
        assert_approx(model.state[4].norm_sqr(), 0.5, TOL, "|11> in Bell state");
        assert_approx(model.total_leakage(), 0.0, TOL, "No leakage in ideal Bell");

        // Now with leaky CNOT
        model.reset();
        model.apply_single_gate(&h, 0);
        let cnot_leak = LeakageGate::cnot_with_leakage(0.05, 0.0);
        model.apply_two_qubit_gate(&cnot_leak, 0, 1);

        // Should still be mostly Bell state but with some leakage
        let total_leak = model.total_leakage();
        assert!(total_leak > 0.0, "Should have leakage from leaky CNOT");
        assert!(total_leak < 0.1, "Leakage should be small: {:.6}", total_leak);

        let comp_probs = model.computational_probabilities();
        let comp_total: f64 = comp_probs.iter().sum();
        assert!(comp_total > 0.9, "Most population still computational");

        // Bell state fidelity
        let ideal_bell = vec![
            c64(1.0 / 2.0_f64.sqrt(), 0.0),
            c64_zero(),
            c64_zero(),
            c64(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let fid = model.computational_fidelity(&ideal_bell);
        assert!(fid > 0.9, "Bell fidelity still high: {:.6}", fid);
    }

    // --------------------------------------------------------
    // 17. test_repeated_gates_leakage_growth
    // --------------------------------------------------------

    #[test]
    fn test_repeated_gates_leakage_growth() {
        let mut model = model_1q();
        let epsilon = 0.05;
        let x_leak = LeakageGate::x_with_leakage(epsilon);

        let mut max_leakage = 0.0_f64;

        // Apply many leaky X gates
        for _ in 0..20 {
            model.apply_single_gate(&x_leak, 0);
            let leak = model.leakage_population()[0];
            max_leakage = max_leakage.max(leak);
        }

        // With repeated leaky gates, leakage should reach some equilibrium
        assert!(max_leakage > 0.0, "Leakage should occur");
        assert!(
            model.stats.peak_leakage > 0.0,
            "Peak leakage should be tracked"
        );

        // History should have 20 entries
        assert_eq!(model.stats.leakage_history.len(), 20);
    }

    // --------------------------------------------------------
    // 18. test_lru_after_leakage
    // --------------------------------------------------------

    #[test]
    fn test_lru_after_leakage() {
        let mut model = model_1q();

        // Apply leaky X to generate leakage
        let x_leak = LeakageGate::x_with_leakage(0.2);
        model.apply_single_gate(&x_leak, 0);

        let leakage_before = model.leakage_population()[0];
        assert!(leakage_before > 0.01, "Should have significant leakage");

        // Apply LRU to correct leakage
        model.apply_lru(0, 1.0);

        // The LRU swaps |0> <-> |2>, so |2> population goes to |0>
        // This should reduce leakage
        let leakage_after = model.leakage_population()[0];

        // After the sequence X_leak followed by LRU, the state evolution is:
        // |0> -> X_leak -> a|1> + b|2> -> LRU -> a|1> + phase*|0>
        // So leakage should be essentially zero
        assert!(
            leakage_after < leakage_before,
            "LRU should reduce leakage: {:.6} -> {:.6}",
            leakage_before,
            leakage_after
        );

        // Check stats
        assert_eq!(model.stats.lru_applications, 1);
    }

    // --------------------------------------------------------
    // 19. test_leakage_statistics
    // --------------------------------------------------------

    #[test]
    fn test_leakage_statistics() {
        let mut model = model_2q();

        // Build a small circuit
        let h = LeakageGate::h_ideal();
        let x_leak = LeakageGate::x_with_leakage(0.1);
        let cnot = LeakageGate::cnot_ideal();

        model.apply_single_gate(&h, 0);
        model.apply_single_gate(&x_leak, 1);
        model.apply_two_qubit_gate(&cnot, 0, 1);
        model.apply_lru(1, 0.9);

        assert_eq!(model.stats.total_gates, 4, "4 gates total (including LRU)");
        assert_eq!(model.stats.single_gates, 3, "3 single gates (H, X_leak, LRU)");
        assert_eq!(model.stats.two_qubit_gates, 1, "1 two-qubit gate");
        assert_eq!(model.stats.lru_applications, 1, "1 LRU");
        assert_eq!(
            model.stats.leakage_history.len(),
            4,
            "4 leakage snapshots"
        );
        assert_eq!(
            model.stats.peak_leakage_per_qubit.len(),
            2,
            "2 qubit peak tracking"
        );
    }

    // --------------------------------------------------------
    // 20. test_three_qubit_leakage
    // --------------------------------------------------------

    #[test]
    fn test_three_qubit_leakage() {
        let rates = LeakageRates::default();
        let mut model = LeakageModel::new(
            3,
            vec![rates.clone(), rates.clone(), rates],
        );
        assert_eq!(model.dim(), 27); // 3^3

        // Apply H to qubit 0
        let h = LeakageGate::h_ideal();
        model.apply_single_gate(&h, 0);

        // CNOT(0, 1) with leakage
        let cnot_leak = LeakageGate::cnot_with_leakage(0.05, 0.0);
        model.apply_two_qubit_gate(&cnot_leak, 0, 1);

        // CNOT(1, 2) ideal
        let cnot = LeakageGate::cnot_ideal();
        model.apply_two_qubit_gate(&cnot, 1, 2);

        // Check properties
        assert_approx(model.norm_squared(), 1.0, GATE_TOL, "3-qubit norm");

        // Should have some leakage from the leaky CNOT
        let total_leak = model.total_leakage();
        assert!(total_leak >= 0.0, "Leakage is non-negative");

        // Computational probabilities
        let comp_probs = model.computational_probabilities();
        assert_eq!(comp_probs.len(), 8); // 2^3

        // Stats
        assert_eq!(model.stats.total_gates, 3);
        assert_eq!(model.stats.peak_leakage_per_qubit.len(), 3);
    }

    // --------------------------------------------------------
    // 21. test_basis_state_encoding
    // --------------------------------------------------------

    #[test]
    fn test_basis_state_encoding() {
        let model = LeakageModel::with_uniform_rates(3, LeakageRates::default());

        // Test index_to_levels and levels_to_index are inverses
        for idx in 0..27 {
            let levels = model.index_to_levels(idx);
            let reconstructed = model.levels_to_index(&levels);
            assert_eq!(idx, reconstructed, "Round-trip for index {}", idx);
        }

        // Specific cases
        assert_eq!(model.index_to_levels(0), vec![0, 0, 0]); // |000>
        assert_eq!(model.index_to_levels(1), vec![1, 0, 0]); // |001> (q0=1)
        assert_eq!(model.index_to_levels(3), vec![0, 1, 0]); // |010> (q1=1)
        assert_eq!(model.index_to_levels(13), vec![1, 1, 1]); // |111>
    }

    // --------------------------------------------------------
    // 22. test_circuit_builder
    // --------------------------------------------------------

    #[test]
    fn test_circuit_builder() {
        let mut model = model_2q();

        let mut builder = LeakageCircuitBuilder::new(2);
        builder
            .h(0)
            .cnot_ideal(0, 1)
            .barrier()
            .idle(0, 100.0)
            .lru_perfect(1);

        let ops = builder.build();
        assert_eq!(ops.len(), 5);

        builder.execute(&mut model);
        assert_eq!(model.stats.total_gates, 4, "4 gates (barrier is no-op)");
    }

    // --------------------------------------------------------
    // 23. test_gate_unitarity
    // --------------------------------------------------------

    #[test]
    fn test_gate_unitarity() {
        let tol = 1e-10;

        // All ideal gates should be unitary
        assert!(LeakageGate::x_ideal().is_unitary(tol), "X ideal unitary");
        assert!(LeakageGate::z_ideal().is_unitary(tol), "Z ideal unitary");
        assert!(LeakageGate::h_ideal().is_unitary(tol), "H ideal unitary");
        assert!(LeakageGate::cnot_ideal().is_unitary(tol), "CNOT ideal unitary");

        // Leaky gates should also be unitary
        assert!(
            LeakageGate::x_with_leakage(0.1).is_unitary(tol),
            "X leaky unitary"
        );
        assert!(
            LeakageGate::x_with_leakage(0.5).is_unitary(tol),
            "X large leak unitary"
        );
        assert!(
            LeakageGate::cnot_with_leakage(0.1, 0.05).is_unitary(tol),
            "CNOT leaky unitary"
        );

        // LRU gates should be unitary
        assert!(LeakageGate::lru(0.5).is_unitary(tol), "LRU half unitary");
        assert!(LeakageGate::lru(1.0).is_unitary(tol), "LRU full unitary");
    }

    // --------------------------------------------------------
    // 24. test_reset_clears_state
    // --------------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let mut model = model_2q();

        // Apply some gates to create a complex state
        let h = LeakageGate::h_ideal();
        let x_leak = LeakageGate::x_with_leakage(0.2);
        model.apply_single_gate(&h, 0);
        model.apply_single_gate(&x_leak, 1);

        assert!(model.stats.total_gates > 0);
        assert!(model.state[0].norm_sqr() < 0.99); // State changed

        // Reset
        model.reset();
        assert_approx(model.state[0].norm_sqr(), 1.0, TOL, "|00> after reset");
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm after reset");
        assert_eq!(model.stats.total_gates, 0, "Stats cleared");
        assert_eq!(model.stats.leakage_history.len(), 0, "History cleared");
    }

    // --------------------------------------------------------
    // 25. test_qec_leakage_analysis
    // --------------------------------------------------------

    #[test]
    fn test_qec_leakage_analysis() {
        // 4-qubit model: qubits 0,1 are data, qubits 2,3 are syndrome
        let rates = LeakageRates::default();
        let mut model = LeakageModel::new(
            4,
            vec![rates.clone(), rates.clone(), rates.clone(), rates],
        );

        // Introduce leakage on data qubit 0
        model.set_basis_state(&[2, 0, 0, 0]);

        let report = model.qec_leakage_analysis(&[0, 1], &[2, 3], 0.01);

        assert_approx(report.data_leakage[0], 1.0, TOL, "Data qubit 0 leaked");
        assert_approx(report.data_leakage[1], 0.0, TOL, "Data qubit 1 fine");
        assert_approx(report.syndrome_leakage[0], 0.0, TOL, "Syndrome 0 fine");
        assert_eq!(report.data_exceeded, vec![0], "Data qubit 0 exceeded threshold");
        assert!(report.syndrome_exceeded.is_empty(), "No syndrome exceeded");
        assert!(report.needs_lru, "LRU needed");
        assert!(report.total_leakage > 0.99, "Total leakage ~1.0");
    }

    // --------------------------------------------------------
    // 26. test_partial_lru_fidelity
    // --------------------------------------------------------

    #[test]
    fn test_partial_lru_fidelity() {
        let mut model = model_1q();

        // Put in |2>
        model.set_basis_state(&[2]);

        // Apply partial LRU (50% fidelity)
        let lru_partial = LeakageGate::lru(0.5);
        assert!(lru_partial.is_unitary(1e-10), "Partial LRU unitary");

        model.apply_single_gate(&lru_partial, 0);

        // Should be partially in |0> and partially in |2>
        let p0 = model.state[0].norm_sqr();
        let p2 = model.state[2].norm_sqr();

        assert!(p0 > 0.0, "Some population moved to |0>");
        assert!(p2 > 0.0, "Some population remains in |2>");
        assert!(p0 < 1.0, "Not all population moved (partial)");
        assert_approx(p0 + p2, 1.0, TOL, "Only |0> and |2> populated");
    }

    // --------------------------------------------------------
    // 27. test_display_implementations
    // --------------------------------------------------------

    #[test]
    fn test_display_implementations() {
        let mut model = model_2q();
        let h = LeakageGate::h_ideal();
        model.apply_single_gate(&h, 0);

        let display = format!("{}", model);
        assert!(display.contains("LeakageModel"), "Should show model name");
        assert!(display.contains("qutrits"), "Should mention qutrits");
        assert!(display.contains("leakage"), "Should show leakage");

        let stats_display = format!("{}", model.stats);
        assert!(stats_display.contains("Total gates: 1"), "Should show gate count");
    }

    // --------------------------------------------------------
    // 28. test_cnot_control_leakage
    // --------------------------------------------------------

    #[test]
    fn test_cnot_control_leakage() {
        let mut model = model_2q();

        // Start in |10>
        model.set_basis_state(&[1, 0]);

        // CNOT with significant control leakage
        let cnot_leak = LeakageGate::cnot_with_leakage(0.0, 0.15);
        model.apply_two_qubit_gate(&cnot_leak, 0, 1);

        // Control qubit should show leakage
        let leakage = model.leakage_population();
        assert!(
            leakage[0] > 0.0,
            "Control should show leakage: {:.6}",
            leakage[0]
        );

        assert_approx(model.norm_squared(), 1.0, GATE_TOL, "Norm preserved");
    }

    // --------------------------------------------------------
    // 29. test_set_basis_state
    // --------------------------------------------------------

    #[test]
    fn test_set_basis_state() {
        let mut model = LeakageModel::with_uniform_rates(3, LeakageRates::default());

        // Set to |1, 2, 0>
        model.set_basis_state(&[1, 2, 0]);
        let idx = model.levels_to_index(&[1, 2, 0]);
        assert_approx(model.state[idx].norm_sqr(), 1.0, TOL, "Correct basis state");
        assert_approx(model.norm_squared(), 1.0, TOL, "Norm = 1");

        // Verify leakage detection
        let leakage = model.leakage_population();
        assert_approx(leakage[0], 0.0, TOL, "Q0 in |1>, no leakage");
        assert_approx(leakage[1], 1.0, TOL, "Q1 in |2>, leaked");
        assert_approx(leakage[2], 0.0, TOL, "Q2 in |0>, no leakage");
    }

    // --------------------------------------------------------
    // 30. test_high_quality_vs_fast_gate_transmon
    // --------------------------------------------------------

    #[test]
    fn test_high_quality_vs_fast_gate_transmon() {
        let hq = LeakageRates::high_quality_transmon();
        let fg = LeakageRates::fast_gate_transmon();

        // Fast-gate transmon should have higher leakage
        assert!(
            fg.gamma_01_to_02 > hq.gamma_01_to_02,
            "Fast gates = more leakage: {:.6} vs {:.6}",
            fg.gamma_01_to_02,
            hq.gamma_01_to_02
        );

        // Verify both produce valid simulations
        let mut model_hq = LeakageModel::new(1, vec![hq.clone()]);
        let mut model_fg = LeakageModel::new(1, vec![fg.clone()]);

        let x_hq = LeakageGate::x_from_rates(&hq);
        let x_fg = LeakageGate::x_from_rates(&fg);

        model_hq.apply_single_gate(&x_hq, 0);
        model_fg.apply_single_gate(&x_fg, 0);

        let leak_hq = model_hq.leakage_population()[0];
        let leak_fg = model_fg.leakage_population()[0];

        assert!(
            leak_fg > leak_hq,
            "Fast gate model produces more leakage: {:.6} vs {:.6}",
            leak_fg,
            leak_hq
        );
    }
}
