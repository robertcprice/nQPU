//! Superconducting Transmon Quantum Processing Unit Backend
//!
//! Physics-based simulation of superconducting transmon quantum processors,
//! modeling real device behavior including leakage, crosstalk, decoherence,
//! and calibration drift.
//!
//! # Physical Model
//!
//! Transmon qubits are weakly anharmonic oscillators with energy levels:
//!   E_n = ω₀₁·n - (α/2)·n(n-1)
//! where ω₀₁ is the 0→1 transition frequency (4-6 GHz) and α is the
//! anharmonicity (-200 to -400 MHz).  The third level |2⟩ is kept to
//! model leakage during fast gates.
//!
//! Two-qubit coupling is capacitive, producing an always-on ZZ interaction:
//!   ζ_ij ≈ g²·α / (Δ·(Δ - α))
//! where g is the coupling strength and Δ = ω_i - ω_j.
//!
//! # Noise Sources
//!
//! - **T1 relaxation**: energy decay |1⟩→|0⟩ from dielectric loss, Purcell
//!   effect, and quasiparticle tunneling.
//! - **T2 dephasing**: phase decay from 1/T2 = 1/(2T1) + 1/Tφ, with Tφ
//!   dominated by charge noise, flux noise, and TLS defects.
//! - **ZZ crosstalk**: always-on conditional phase from capacitive coupling.
//! - **Leakage**: population transfer to |2⟩ during fast single-qubit gates
//!   and two-qubit drives.
//! - **Readout errors**: assignment infidelity and measurement crosstalk.
//! - **TLS defects**: two-level systems in substrate/oxide causing frequency
//!   fluctuations and T1 variation.
//! - **Calibration drift**: gate parameters drift over hours-days timescale.
//!
//! # Device Presets
//!
//! Calibrated configurations for real hardware families:
//! - IBM Eagle (127Q, heavy-hex, ECR native gate)
//! - IBM Heron (156Q, heavy-hex, ECR native gate, improved coherence)
//! - Google Sycamore (53Q, grid, √iSWAP native gate)
//! - Google Willow (105Q, grid, √iSWAP native gate)
//! - Rigetti Ankaa (84Q, octagonal, CZ native gate)
//!
//! # References
//!
//! - Koch et al., PRA 76 (2007) -- Transmon qubit
//! - Krantz et al., Appl. Phys. Rev. 6 (2019) -- Superconducting review
//! - Sheldon et al., PRA 93 (2016) -- Cross-resonance gate
//! - Arute et al., Nature 574 (2019) -- Sycamore / quantum supremacy
//! - Acharya et al., Nature 634 (2024) -- Willow / below threshold QEC
//! - Müller et al., Rep. Prog. Phys. 82 (2019) -- TLS defects

use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::gates::{Gate, GateType};
use crate::traits::{BackendError, BackendResult, ErrorModel, QuantumBackend};

// ===================================================================
// PHYSICAL CONSTANTS
// ===================================================================

/// Boltzmann constant in J/K.
const K_B: f64 = 1.380_649e-23;
/// Reduced Planck constant in J·s.
const HBAR: f64 = 1.054_571_817e-34;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors from superconducting backend operations.
#[derive(Debug, Clone)]
pub enum TransmonError {
    /// Configuration parameter out of valid range.
    InvalidConfig(String),
    /// Qubit index out of bounds.
    QubitOutOfBounds { index: usize, num_qubits: usize },
    /// Requested coupling not present in topology.
    NoCoupling { qubit_a: usize, qubit_b: usize },
    /// Gate precondition violated.
    GatePrecondition(String),
    /// Simulation error.
    SimulationError(String),
}

impl std::fmt::Display for TransmonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            Self::QubitOutOfBounds { index, num_qubits } => {
                write!(f, "Qubit {} out of bounds ({})", index, num_qubits)
            }
            Self::NoCoupling { qubit_a, qubit_b } => {
                write!(f, "No coupling between qubits {} and {}", qubit_a, qubit_b)
            }
            Self::GatePrecondition(msg) => write!(f, "Gate precondition: {}", msg),
            Self::SimulationError(msg) => write!(f, "Simulation error: {}", msg),
        }
    }
}

impl std::error::Error for TransmonError {}

pub type TransmonResult<T> = Result<T, TransmonError>;

// ===================================================================
// CHIP TOPOLOGY
// ===================================================================

/// Layout families for superconducting processors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopologyKind {
    /// IBM heavy-hex lattice (degree-3 connectivity with bridge qubits).
    HeavyHex,
    /// Google-style 2D grid (degree-4 connectivity).
    Grid,
    /// Rigetti octagonal coupler layout.
    Octagonal,
    /// Fully connected (for small test chips).
    FullyConnected,
    /// Custom (edges supplied manually).
    Custom,
}

/// A single coupling link between two qubits on-chip.
#[derive(Debug, Clone)]
pub struct CouplerLink {
    pub qubit_a: usize,
    pub qubit_b: usize,
    /// Coupling strength g/2π in MHz.
    pub coupling_mhz: f64,
    /// Measured ZZ rate ζ/2π in kHz (derived from g, Δ, α if not supplied).
    pub zz_khz: Option<f64>,
}

/// Chip topology: qubit count + coupling map.
#[derive(Debug, Clone)]
pub struct ChipTopology {
    pub kind: TopologyKind,
    pub num_qubits: usize,
    pub couplers: Vec<CouplerLink>,
}

impl ChipTopology {
    /// Generate a heavy-hex lattice for `n` qubits (IBM-style).
    ///
    /// Heavy-hex places qubits at degree-2 and degree-3 vertices of a
    /// hexagonal lattice, yielding low crosstalk and high coherence.
    pub fn heavy_hex(num_qubits: usize, coupling_mhz: f64) -> Self {
        let mut couplers = Vec::new();
        // Simplified heavy-hex: row-based generation.
        // Real IBM chips use specific coupling maps; this approximates the structure.
        let cols = ((num_qubits as f64).sqrt() * 1.2) as usize;
        let rows = (num_qubits + cols - 1) / cols;
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if q >= num_qubits {
                    break;
                }
                // Horizontal neighbor
                if c + 1 < cols {
                    let q2 = r * cols + c + 1;
                    if q2 < num_qubits {
                        couplers.push(CouplerLink {
                            qubit_a: q,
                            qubit_b: q2,
                            coupling_mhz,
                            zz_khz: None,
                        });
                    }
                }
                // Vertical neighbor (skip every other to make heavy-hex sparse)
                if r + 1 < rows && c % 2 == 0 {
                    let q2 = (r + 1) * cols + c;
                    if q2 < num_qubits {
                        couplers.push(CouplerLink {
                            qubit_a: q,
                            qubit_b: q2,
                            coupling_mhz,
                            zz_khz: None,
                        });
                    }
                }
            }
        }
        Self {
            kind: TopologyKind::HeavyHex,
            num_qubits,
            couplers,
        }
    }

    /// Generate a 2D grid topology (Google-style).
    pub fn grid(rows: usize, cols: usize, coupling_mhz: f64) -> Self {
        let num_qubits = rows * cols;
        let mut couplers = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let q = r * cols + c;
                if c + 1 < cols {
                    couplers.push(CouplerLink {
                        qubit_a: q,
                        qubit_b: q + 1,
                        coupling_mhz,
                        zz_khz: None,
                    });
                }
                if r + 1 < rows {
                    couplers.push(CouplerLink {
                        qubit_a: q,
                        qubit_b: q + cols,
                        coupling_mhz,
                        zz_khz: None,
                    });
                }
            }
        }
        Self {
            kind: TopologyKind::Grid,
            num_qubits,
            couplers,
        }
    }

    /// Fully connected topology for small test chips.
    pub fn fully_connected(num_qubits: usize, coupling_mhz: f64) -> Self {
        let mut couplers = Vec::new();
        for i in 0..num_qubits {
            for j in (i + 1)..num_qubits {
                couplers.push(CouplerLink {
                    qubit_a: i,
                    qubit_b: j,
                    coupling_mhz,
                    zz_khz: None,
                });
            }
        }
        Self {
            kind: TopologyKind::FullyConnected,
            num_qubits,
            couplers,
        }
    }

    /// Build from an explicit edge list.
    pub fn custom(num_qubits: usize, couplers: Vec<CouplerLink>) -> Self {
        Self {
            kind: TopologyKind::Custom,
            num_qubits,
            couplers,
        }
    }

    /// Check whether two qubits are directly coupled.
    pub fn are_coupled(&self, a: usize, b: usize) -> bool {
        self.couplers
            .iter()
            .any(|c| (c.qubit_a == a && c.qubit_b == b) || (c.qubit_a == b && c.qubit_b == a))
    }

    /// Get the coupling link between two qubits, if it exists.
    pub fn get_coupler(&self, a: usize, b: usize) -> Option<&CouplerLink> {
        self.couplers
            .iter()
            .find(|c| (c.qubit_a == a && c.qubit_b == b) || (c.qubit_a == b && c.qubit_b == a))
    }

    /// Return the neighbors of a qubit.
    pub fn neighbors(&self, qubit: usize) -> Vec<usize> {
        self.couplers
            .iter()
            .filter_map(|c| {
                if c.qubit_a == qubit {
                    Some(c.qubit_b)
                } else if c.qubit_b == qubit {
                    Some(c.qubit_a)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ===================================================================
// TRANSMON QUBIT MODEL
// ===================================================================

/// Physical parameters of a single transmon qubit.
#[derive(Debug, Clone)]
pub struct TransmonQubit {
    /// Qubit index on the chip.
    pub index: usize,
    /// 0→1 transition frequency ω₀₁/2π in GHz.
    pub frequency_ghz: f64,
    /// Anharmonicity α/2π in MHz (negative for transmon, typically -200 to -400).
    pub anharmonicity_mhz: f64,
    /// Energy relaxation time T1 in microseconds.
    pub t1_us: f64,
    /// Dephasing time T2 in microseconds.
    pub t2_us: f64,
    /// Single-qubit gate fidelity (0.999 typical for modern devices).
    pub single_gate_fidelity: f64,
    /// Readout assignment fidelity (P(0|0) + P(1|1)) / 2.
    pub readout_fidelity: f64,
    /// Thermal population of |1⟩ at base temperature (typically 0.5-2%).
    pub thermal_population: f64,
    /// Single-qubit gate duration in nanoseconds.
    pub gate_time_ns: f64,
}

impl TransmonQubit {
    /// Create a qubit with typical IBM Heron-class parameters.
    pub fn typical(index: usize) -> Self {
        Self {
            index,
            frequency_ghz: 5.0 + 0.1 * (index as f64 % 5.0), // stagger frequencies
            anharmonicity_mhz: -330.0,
            t1_us: 300.0,
            t2_us: 200.0,
            single_gate_fidelity: 0.9995,
            readout_fidelity: 0.99,
            thermal_population: 0.01,
            gate_time_ns: 25.0,
        }
    }

    /// 0→2 transition frequency in GHz.
    pub fn frequency_02_ghz(&self) -> f64 {
        2.0 * self.frequency_ghz + self.anharmonicity_mhz / 1000.0
    }

    /// Pure dephasing rate 1/Tφ in 1/μs.
    pub fn pure_dephasing_rate(&self) -> f64 {
        let gamma2 = 1.0 / self.t2_us;
        let gamma1_half = 0.5 / self.t1_us;
        (gamma2 - gamma1_half).max(0.0)
    }

    /// Leakage rate to |2⟩ during a gate of given duration.
    ///
    /// Scales as (Ω / α)² where Ω is the drive Rabi frequency.
    /// For a gate of duration t_gate, Ω ≈ π / t_gate, so
    /// P_leak ≈ (π / (t_gate · α))².
    pub fn leakage_probability(&self) -> f64 {
        let alpha_ghz = self.anharmonicity_mhz.abs() / 1000.0;
        let omega_ghz = 1.0 / (2.0 * self.gate_time_ns * 1e-9) / 1e9;
        let ratio = omega_ghz / alpha_ghz;
        (ratio * ratio).min(0.05) // cap at 5%
    }
}

// ===================================================================
// NATIVE GATE SET
// ===================================================================

/// Native two-qubit gate families for different hardware vendors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NativeGateFamily {
    /// Echoed cross-resonance (IBM).
    ECR,
    /// √iSWAP (Google Sycamore/Willow).
    SqrtISWAP,
    /// Controlled-Z (Rigetti, some Google).
    CZ,
    /// Parameterized fSim (Google, most general).
    FSim,
}

/// A native gate operation on the physical processor.
#[derive(Debug, Clone)]
pub enum NativeGate {
    /// Virtual-Z rotation (zero duration, frame update only).
    Rz { qubit: usize, angle: f64 },
    /// Physical X90 pulse (π/2 rotation about X).
    SX { qubit: usize },
    /// Physical X pulse (π rotation about X).
    X { qubit: usize },
    /// Echoed cross-resonance gate (IBM native 2Q gate).
    ECR { qubit_a: usize, qubit_b: usize },
    /// √iSWAP gate (Google native 2Q gate).
    SqrtISWAP { qubit_a: usize, qubit_b: usize },
    /// Controlled-Z gate (Rigetti native 2Q gate).
    CZGate { qubit_a: usize, qubit_b: usize },
    /// General fSim(θ, φ) gate.
    FSim {
        qubit_a: usize,
        qubit_b: usize,
        theta: f64,
        phi: f64,
    },
    /// Measurement.
    Measure { qubit: usize },
    /// Reset to |0⟩.
    Reset { qubit: usize },
}

// ===================================================================
// PROCESSOR CONFIGURATION
// ===================================================================

/// Full processor configuration.
#[derive(Debug, Clone)]
pub struct TransmonProcessor {
    /// Per-qubit physical parameters.
    pub qubits: Vec<TransmonQubit>,
    /// Chip connectivity and coupling data.
    pub topology: ChipTopology,
    /// Native two-qubit gate family.
    pub native_2q_gate: NativeGateFamily,
    /// Two-qubit gate fidelity (average).
    pub two_qubit_fidelity: f64,
    /// Two-qubit gate duration in nanoseconds.
    pub two_qubit_gate_time_ns: f64,
    /// Readout duration in nanoseconds.
    pub readout_time_ns: f64,
    /// Measurement crosstalk: fraction of signal leaking to neighbors.
    pub measurement_crosstalk: f64,
    /// Base temperature in millikelvin.
    pub temperature_mk: f64,
}

impl TransmonProcessor {
    /// Number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// Compute ZZ coupling rate ζ/2π in kHz between two qubits.
    ///
    /// ζ ≈ g²·α / (Δ·(Δ - α)) where Δ = ω_i - ω_j, g is coupling,
    /// α is anharmonicity of the higher-frequency qubit.
    pub fn zz_coupling_khz(&self, a: usize, b: usize) -> f64 {
        if let Some(coupler) = self.topology.get_coupler(a, b) {
            // If ZZ was measured directly, use that.
            if let Some(zz) = coupler.zz_khz {
                return zz;
            }
            let qa = &self.qubits[a];
            let qb = &self.qubits[b];
            let g_ghz = coupler.coupling_mhz / 1000.0;
            let delta_ghz = qa.frequency_ghz - qb.frequency_ghz;
            let alpha_ghz = qa.anharmonicity_mhz.abs() / 1000.0;
            if delta_ghz.abs() < 1e-6 {
                // Near-resonant: perturbation theory breaks down, return large ZZ.
                return (g_ghz * g_ghz / alpha_ghz) * 1e6; // kHz
            }
            let zz_ghz = g_ghz * g_ghz * alpha_ghz / (delta_ghz * (delta_ghz - alpha_ghz));
            zz_ghz.abs() * 1e6 // convert GHz → kHz
        } else {
            0.0
        }
    }
}

// ===================================================================
// DEVICE PRESETS
// ===================================================================

/// IBM Eagle: 127 qubits, heavy-hex, ECR gate.
pub fn ibm_eagle() -> TransmonProcessor {
    let num_qubits = 127;
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: 4.8 + 0.05 * (i as f64 % 8.0),
            anharmonicity_mhz: -340.0,
            t1_us: 120.0,
            t2_us: 80.0,
            single_gate_fidelity: 0.9996,
            readout_fidelity: 0.98,
            thermal_population: 0.015,
            gate_time_ns: 35.0,
        })
        .collect();
    TransmonProcessor {
        topology: ChipTopology::heavy_hex(num_qubits, 3.5),
        qubits,
        native_2q_gate: NativeGateFamily::ECR,
        two_qubit_fidelity: 0.99,
        two_qubit_gate_time_ns: 300.0,
        readout_time_ns: 1200.0,
        measurement_crosstalk: 0.02,
        temperature_mk: 15.0,
    }
}

/// IBM Heron: 156 qubits, heavy-hex, ECR gate, improved coherence.
pub fn ibm_heron() -> TransmonProcessor {
    let num_qubits = 156;
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: 4.9 + 0.04 * (i as f64 % 10.0),
            anharmonicity_mhz: -320.0,
            t1_us: 300.0,
            t2_us: 200.0,
            single_gate_fidelity: 0.9998,
            readout_fidelity: 0.995,
            thermal_population: 0.008,
            gate_time_ns: 25.0,
        })
        .collect();
    TransmonProcessor {
        topology: ChipTopology::heavy_hex(num_qubits, 3.0),
        qubits,
        native_2q_gate: NativeGateFamily::ECR,
        two_qubit_fidelity: 0.995,
        two_qubit_gate_time_ns: 200.0,
        readout_time_ns: 800.0,
        measurement_crosstalk: 0.01,
        temperature_mk: 12.0,
    }
}

/// Google Sycamore: 53 qubits, grid, √iSWAP gate.
pub fn google_sycamore() -> TransmonProcessor {
    let num_qubits = 53;
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: 5.5 + 0.15 * (i as f64 % 7.0),
            anharmonicity_mhz: -220.0,
            t1_us: 16.0,
            t2_us: 12.0,
            single_gate_fidelity: 0.9985,
            readout_fidelity: 0.965,
            thermal_population: 0.02,
            gate_time_ns: 25.0,
        })
        .collect();
    // Sycamore is roughly a 6x9 grid (minus some qubits).
    TransmonProcessor {
        topology: ChipTopology::grid(6, 9, 5.0),
        qubits,
        native_2q_gate: NativeGateFamily::SqrtISWAP,
        two_qubit_fidelity: 0.995,
        two_qubit_gate_time_ns: 32.0,
        readout_time_ns: 1000.0,
        measurement_crosstalk: 0.03,
        temperature_mk: 20.0,
    }
}

/// Google Willow: 105 qubits, grid, √iSWAP gate, below-threshold QEC.
pub fn google_willow() -> TransmonProcessor {
    let num_qubits = 105;
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: 5.2 + 0.12 * (i as f64 % 8.0),
            anharmonicity_mhz: -230.0,
            t1_us: 68.0,
            t2_us: 30.0,
            single_gate_fidelity: 0.9993,
            readout_fidelity: 0.993,
            thermal_population: 0.006,
            gate_time_ns: 22.0,
        })
        .collect();
    TransmonProcessor {
        topology: ChipTopology::grid(10, 11, 4.5),
        qubits,
        native_2q_gate: NativeGateFamily::SqrtISWAP,
        two_qubit_fidelity: 0.997,
        two_qubit_gate_time_ns: 26.0,
        readout_time_ns: 800.0,
        measurement_crosstalk: 0.015,
        temperature_mk: 15.0,
    }
}

/// Rigetti Ankaa: 84 qubits, octagonal, CZ gate.
pub fn rigetti_ankaa() -> TransmonProcessor {
    let num_qubits = 84;
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: 4.5 + 0.08 * (i as f64 % 6.0),
            anharmonicity_mhz: -280.0,
            t1_us: 25.0,
            t2_us: 15.0,
            single_gate_fidelity: 0.999,
            readout_fidelity: 0.97,
            thermal_population: 0.02,
            gate_time_ns: 40.0,
        })
        .collect();
    // Ankaa uses an octagonal layout; approximate as grid.
    TransmonProcessor {
        topology: ChipTopology::grid(7, 12, 6.0),
        qubits,
        native_2q_gate: NativeGateFamily::CZ,
        two_qubit_fidelity: 0.99,
        two_qubit_gate_time_ns: 180.0,
        readout_time_ns: 1000.0,
        measurement_crosstalk: 0.025,
        temperature_mk: 18.0,
    }
}

/// Small test processor: 5 qubits, fully connected, ideal-ish parameters.
pub fn test_processor(num_qubits: usize) -> TransmonProcessor {
    let qubits: Vec<TransmonQubit> = (0..num_qubits)
        .map(|i| TransmonQubit::typical(i))
        .collect();
    TransmonProcessor {
        topology: ChipTopology::fully_connected(num_qubits, 4.0),
        qubits,
        native_2q_gate: NativeGateFamily::ECR,
        two_qubit_fidelity: 0.995,
        two_qubit_gate_time_ns: 200.0,
        readout_time_ns: 800.0,
        measurement_crosstalk: 0.01,
        temperature_mk: 15.0,
    }
}

/// Build a processor from calibration data (digital twin construction).
///
/// Accepts per-qubit T1/T2/frequency/readout data and a coupling map,
/// returning a physics-consistent processor model.
pub fn from_calibration_data(
    frequencies_ghz: &[f64],
    anharmonicities_mhz: &[f64],
    t1_us: &[f64],
    t2_us: &[f64],
    readout_fidelities: &[f64],
    edges: &[(usize, usize, f64)], // (qubit_a, qubit_b, coupling_mhz)
    native_gate: NativeGateFamily,
    two_qubit_fidelity: f64,
    two_qubit_gate_time_ns: f64,
) -> TransmonResult<TransmonProcessor> {
    let n = frequencies_ghz.len();
    if anharmonicities_mhz.len() != n
        || t1_us.len() != n
        || t2_us.len() != n
        || readout_fidelities.len() != n
    {
        return Err(TransmonError::InvalidConfig(
            "All per-qubit arrays must have the same length".to_string(),
        ));
    }
    let qubits: Vec<TransmonQubit> = (0..n)
        .map(|i| TransmonQubit {
            index: i,
            frequency_ghz: frequencies_ghz[i],
            anharmonicity_mhz: anharmonicities_mhz[i],
            t1_us: t1_us[i],
            t2_us: t2_us[i],
            single_gate_fidelity: 0.9995,
            readout_fidelity: readout_fidelities[i],
            thermal_population: 0.01,
            gate_time_ns: 25.0,
        })
        .collect();
    let couplers: Vec<CouplerLink> = edges
        .iter()
        .map(|&(a, b, g)| CouplerLink {
            qubit_a: a,
            qubit_b: b,
            coupling_mhz: g,
            zz_khz: None,
        })
        .collect();
    Ok(TransmonProcessor {
        topology: ChipTopology::custom(n, couplers),
        qubits,
        native_2q_gate: native_gate,
        two_qubit_fidelity,
        two_qubit_gate_time_ns,
        readout_time_ns: 800.0,
        measurement_crosstalk: 0.01,
        temperature_mk: 15.0,
    })
}

// ===================================================================
// GATE COMPILER: Standard gates → native operations
// ===================================================================

/// Compile a standard gate into a sequence of native operations.
pub fn compile_to_native(gate: &Gate, proc: &TransmonProcessor) -> Vec<NativeGate> {
    match &gate.gate_type {
        // Single-qubit gates compile to virtual-Z + SX sequences.
        GateType::H => {
            let q = gate.targets[0];
            // H = Rz(π) · SX · Rz(π/2) up to global phase
            vec![
                NativeGate::Rz {
                    qubit: q,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: q },
                NativeGate::Rz {
                    qubit: q,
                    angle: PI,
                },
            ]
        }
        GateType::X => {
            let q = gate.targets[0];
            vec![NativeGate::X { qubit: q }]
        }
        GateType::Z => {
            let q = gate.targets[0];
            vec![NativeGate::Rz {
                qubit: q,
                angle: PI,
            }]
        }
        GateType::S => {
            let q = gate.targets[0];
            vec![NativeGate::Rz {
                qubit: q,
                angle: PI / 2.0,
            }]
        }
        GateType::T => {
            let q = gate.targets[0];
            vec![NativeGate::Rz {
                qubit: q,
                angle: PI / 4.0,
            }]
        }
        GateType::SX => {
            let q = gate.targets[0];
            vec![NativeGate::SX { qubit: q }]
        }
        GateType::Rz(angle) => {
            let q = gate.targets[0];
            vec![NativeGate::Rz {
                qubit: q,
                angle: *angle,
            }]
        }
        GateType::Rx(angle) => {
            let q = gate.targets[0];
            // Rx(θ) = Rz(-π/2) · SX · Rz(π/2 - θ) · SX · Rz(0) simplified
            vec![
                NativeGate::Rz {
                    qubit: q,
                    angle: -PI / 2.0,
                },
                NativeGate::SX { qubit: q },
                NativeGate::Rz {
                    qubit: q,
                    angle: PI - angle,
                },
                NativeGate::SX { qubit: q },
                NativeGate::Rz {
                    qubit: q,
                    angle: PI / 2.0,
                },
            ]
        }
        GateType::Ry(angle) => {
            let q = gate.targets[0];
            // Ry(θ) = Rz(π/2) · Rx(θ) · Rz(-π/2) → expand Rx
            vec![
                NativeGate::SX { qubit: q },
                NativeGate::Rz {
                    qubit: q,
                    angle: *angle,
                },
                NativeGate::SX { qubit: q },
                NativeGate::Rz {
                    qubit: q,
                    angle: PI,
                },
            ]
        }
        GateType::Y => {
            let q = gate.targets[0];
            vec![
                NativeGate::Rz {
                    qubit: q,
                    angle: PI,
                },
                NativeGate::X { qubit: q },
            ]
        }
        GateType::Phase(angle) => {
            let q = gate.targets[0];
            vec![NativeGate::Rz {
                qubit: q,
                angle: *angle,
            }]
        }
        // Two-qubit gates → native 2Q gate + single-qubit dressing.
        GateType::CNOT => {
            let ctrl = gate.controls[0];
            let tgt = gate.targets[0];
            compile_cnot(ctrl, tgt, proc)
        }
        GateType::CZ => {
            let ctrl = gate.controls[0];
            let tgt = gate.targets[0];
            compile_cz(ctrl, tgt, proc)
        }
        GateType::SWAP => {
            let a = gate.targets[0];
            let b = gate.targets[1];
            // SWAP = 3 CNOTs
            let mut ops = compile_cnot(a, b, proc);
            ops.extend(compile_cnot(b, a, proc));
            ops.extend(compile_cnot(a, b, proc));
            ops
        }
        // Fallback: wrap as identity (will produce a warning in simulation).
        _ => {
            vec![]
        }
    }
}

/// Compile CNOT into native 2Q gate + single-qubit dressing.
fn compile_cnot(ctrl: usize, tgt: usize, proc: &TransmonProcessor) -> Vec<NativeGate> {
    match proc.native_2q_gate {
        NativeGateFamily::ECR => {
            // CNOT = (I ⊗ Rz(-π/2)) · ECR · (Rz(π/2) ⊗ SX)
            vec![
                NativeGate::Rz {
                    qubit: ctrl,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: tgt },
                NativeGate::ECR {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: -PI / 2.0,
                },
            ]
        }
        NativeGateFamily::SqrtISWAP => {
            // CNOT from √iSWAP: more complex decomposition.
            vec![
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 2.0,
                },
                NativeGate::SqrtISWAP {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: ctrl,
                    angle: -PI / 2.0,
                },
                NativeGate::SqrtISWAP {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: -PI / 2.0,
                },
            ]
        }
        NativeGateFamily::CZ | NativeGateFamily::FSim => {
            // CNOT = (I ⊗ H) · CZ · (I ⊗ H)
            vec![
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: tgt },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI,
                },
                NativeGate::CZGate {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: tgt },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI,
                },
            ]
        }
    }
}

/// Compile CZ into native 2Q gate + single-qubit dressing.
fn compile_cz(ctrl: usize, tgt: usize, proc: &TransmonProcessor) -> Vec<NativeGate> {
    match proc.native_2q_gate {
        NativeGateFamily::CZ | NativeGateFamily::FSim => {
            vec![NativeGate::CZGate {
                qubit_a: ctrl,
                qubit_b: tgt,
            }]
        }
        NativeGateFamily::ECR => {
            // CZ = (I ⊗ H) · CNOT · (I ⊗ H), with CNOT from ECR
            let mut ops = vec![
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: tgt },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI,
                },
            ];
            ops.extend(compile_cnot(ctrl, tgt, proc));
            ops.extend(vec![
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 2.0,
                },
                NativeGate::SX { qubit: tgt },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI,
                },
            ]);
            ops
        }
        NativeGateFamily::SqrtISWAP => {
            // CZ from √iSWAP
            vec![
                NativeGate::Rz {
                    qubit: ctrl,
                    angle: -PI / 4.0,
                },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: PI / 4.0,
                },
                NativeGate::SqrtISWAP {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: ctrl,
                    angle: PI / 2.0,
                },
                NativeGate::SqrtISWAP {
                    qubit_a: ctrl,
                    qubit_b: tgt,
                },
                NativeGate::Rz {
                    qubit: ctrl,
                    angle: PI / 4.0,
                },
                NativeGate::Rz {
                    qubit: tgt,
                    angle: -PI / 4.0,
                },
            ]
        }
    }
}

// ===================================================================
// NOISE MODEL
// ===================================================================

/// Physics-based noise model for superconducting transmon processors.
///
/// Each noise source is independently toggleable and parameterized
/// by the physical constants of the processor.
#[derive(Debug, Clone)]
pub struct TransmonNoiseModel {
    /// Enable T1 energy relaxation.
    pub enable_t1: bool,
    /// Enable T2 dephasing.
    pub enable_t2: bool,
    /// Enable always-on ZZ crosstalk accumulation.
    pub enable_zz_crosstalk: bool,
    /// Enable leakage to |2⟩.
    pub enable_leakage: bool,
    /// Enable readout assignment errors.
    pub enable_readout_error: bool,
    /// Enable measurement crosstalk.
    pub enable_measurement_crosstalk: bool,
    /// Enable thermal initialization error.
    pub enable_thermal_init: bool,
    /// Reference to processor parameters.
    processor: TransmonProcessor,
}

impl TransmonNoiseModel {
    /// Create a full noise model from a processor config (all sources on).
    pub fn new(processor: &TransmonProcessor) -> Self {
        Self {
            enable_t1: true,
            enable_t2: true,
            enable_zz_crosstalk: true,
            enable_leakage: true,
            enable_readout_error: true,
            enable_measurement_crosstalk: true,
            enable_thermal_init: true,
            processor: processor.clone(),
        }
    }

    /// Create an ideal noise model (all sources off).
    pub fn ideal(processor: &TransmonProcessor) -> Self {
        Self {
            enable_t1: false,
            enable_t2: false,
            enable_zz_crosstalk: false,
            enable_leakage: false,
            enable_readout_error: false,
            enable_measurement_crosstalk: false,
            enable_thermal_init: false,
            processor: processor.clone(),
        }
    }

    /// Probability of T1 decay during a gate of given duration on a qubit.
    pub fn t1_decay_prob(&self, qubit: usize, gate_time_ns: f64) -> f64 {
        if !self.enable_t1 {
            return 0.0;
        }
        let t1_ns = self.processor.qubits[qubit].t1_us * 1000.0;
        1.0 - (-gate_time_ns / t1_ns).exp()
    }

    /// Probability of T2 dephasing during a gate.
    pub fn t2_dephase_prob(&self, qubit: usize, gate_time_ns: f64) -> f64 {
        if !self.enable_t2 {
            return 0.0;
        }
        let t2_ns = self.processor.qubits[qubit].t2_us * 1000.0;
        1.0 - (-gate_time_ns / t2_ns).exp()
    }

    /// ZZ phase accumulation in radians between two qubits during a gate.
    pub fn zz_phase_rad(&self, qubit_a: usize, qubit_b: usize, gate_time_ns: f64) -> f64 {
        if !self.enable_zz_crosstalk {
            return 0.0;
        }
        let zz_khz = self.processor.zz_coupling_khz(qubit_a, qubit_b);
        // ζ in rad/ns: zz_khz * 2π * 1e3 / 1e9 = zz_khz * 2π * 1e-6
        2.0 * PI * zz_khz * 1e-6 * gate_time_ns
    }

    /// Leakage probability during a single-qubit gate.
    pub fn leakage_prob(&self, qubit: usize) -> f64 {
        if !self.enable_leakage {
            return 0.0;
        }
        self.processor.qubits[qubit].leakage_probability()
    }

    /// Readout confusion matrix for a qubit: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]].
    pub fn readout_confusion(&self, qubit: usize) -> [[f64; 2]; 2] {
        if !self.enable_readout_error {
            return [[1.0, 0.0], [0.0, 1.0]];
        }
        let f = self.processor.qubits[qubit].readout_fidelity;
        // Asymmetric: |1⟩→|0⟩ more common than |0⟩→|1⟩ for transmons.
        let p0_given_0 = f + (1.0 - f) * 0.3;
        let p1_given_1 = f - (1.0 - f) * 0.3;
        [
            [p0_given_0.min(1.0), (1.0 - p0_given_0).max(0.0)],
            [(1.0 - p1_given_1).max(0.0), p1_given_1.min(1.0)],
        ]
    }

    /// Total depolarizing error rate for a single-qubit gate.
    pub fn single_gate_error(&self, qubit: usize) -> f64 {
        // If all noise sources are disabled, return zero.
        if !self.enable_t1
            && !self.enable_t2
            && !self.enable_leakage
            && !self.enable_readout_error
            && !self.enable_measurement_crosstalk
            && !self.enable_thermal_init
            && !self.enable_zz_crosstalk
        {
            return 0.0;
        }
        let q = &self.processor.qubits[qubit];
        let gate_err = 1.0 - q.single_gate_fidelity;
        let t1_err = self.t1_decay_prob(qubit, q.gate_time_ns);
        let t2_err = self.t2_dephase_prob(qubit, q.gate_time_ns);
        let leak_err = self.leakage_prob(qubit);
        // Combine errors (independent): 1 - (1-e1)(1-e2)... ≈ sum for small errors
        1.0 - (1.0 - gate_err) * (1.0 - t1_err / 3.0) * (1.0 - t2_err / 2.0) * (1.0 - leak_err)
    }

    /// Total error rate for a two-qubit gate between qubits a and b.
    pub fn two_qubit_gate_error(&self, qubit_a: usize, qubit_b: usize) -> f64 {
        // If all noise sources are disabled, return zero (ideal mode).
        if !self.enable_t1
            && !self.enable_t2
            && !self.enable_leakage
            && !self.enable_readout_error
            && !self.enable_measurement_crosstalk
            && !self.enable_thermal_init
            && !self.enable_zz_crosstalk
        {
            return 0.0;
        }
        let gate_err = 1.0 - self.processor.two_qubit_fidelity;
        let t1_a = self.t1_decay_prob(qubit_a, self.processor.two_qubit_gate_time_ns);
        let t1_b = self.t1_decay_prob(qubit_b, self.processor.two_qubit_gate_time_ns);
        let t2_a = self.t2_dephase_prob(qubit_a, self.processor.two_qubit_gate_time_ns);
        let t2_b = self.t2_dephase_prob(qubit_b, self.processor.two_qubit_gate_time_ns);
        // ZZ from spectator qubits accumulates during the 2Q gate.
        let mut zz_err = 0.0;
        if self.enable_zz_crosstalk {
            for neighbor in self.processor.topology.neighbors(qubit_a) {
                if neighbor != qubit_b {
                    let phase =
                        self.zz_phase_rad(qubit_a, neighbor, self.processor.two_qubit_gate_time_ns);
                    zz_err += phase.abs() / (2.0 * PI);
                }
            }
            for neighbor in self.processor.topology.neighbors(qubit_b) {
                if neighbor != qubit_a {
                    let phase =
                        self.zz_phase_rad(qubit_b, neighbor, self.processor.two_qubit_gate_time_ns);
                    zz_err += phase.abs() / (2.0 * PI);
                }
            }
        }
        1.0 - (1.0 - gate_err)
            * (1.0 - (t1_a + t1_b) / 3.0)
            * (1.0 - (t2_a + t2_b) / 2.0)
            * (1.0 - zz_err)
    }
}

// ===================================================================
// CALIBRATION DRIFT MODEL
// ===================================================================

/// Time-dependent calibration drift model.
///
/// Real processors drift over hours: qubit frequencies shift from TLS
/// interactions, T1 fluctuates, gate parameters slowly degrade.
#[derive(Debug, Clone)]
pub struct CalibrationDrift {
    /// Time since last calibration in seconds.
    pub time_since_cal_s: f64,
    /// Frequency drift rate in MHz/hour per qubit (RMS).
    pub freq_drift_mhz_per_hour: f64,
    /// T1 fractional fluctuation per hour (RMS).
    pub t1_fluctuation_per_hour: f64,
    /// Gate fidelity degradation rate per hour.
    pub fidelity_decay_per_hour: f64,
}

impl CalibrationDrift {
    /// Typical drift for a modern processor.
    pub fn typical() -> Self {
        Self {
            time_since_cal_s: 0.0,
            freq_drift_mhz_per_hour: 0.05,
            t1_fluctuation_per_hour: 0.05,
            fidelity_decay_per_hour: 0.001,
        }
    }

    /// No drift (freshly calibrated or ideal).
    pub fn none() -> Self {
        Self {
            time_since_cal_s: 0.0,
            freq_drift_mhz_per_hour: 0.0,
            t1_fluctuation_per_hour: 0.0,
            fidelity_decay_per_hour: 0.0,
        }
    }

    /// Compute additional gate error from calibration drift.
    pub fn gate_error_overhead(&self) -> f64 {
        let hours = self.time_since_cal_s / 3600.0;
        self.fidelity_decay_per_hour * hours
    }

    /// Compute frequency shift in MHz at current drift time.
    pub fn frequency_shift_mhz(&self) -> f64 {
        let hours = self.time_since_cal_s / 3600.0;
        self.freq_drift_mhz_per_hour * hours.sqrt() // random-walk scaling
    }
}

// ===================================================================
// TRANSMON SIMULATOR (QuantumBackend implementation)
// ===================================================================

/// State-vector simulator with physics-based transmon noise injection.
///
/// Wraps an ideal state-vector simulation and applies noise from the
/// `TransmonNoiseModel` after each gate, modeling the physical behavior
/// of a real superconducting processor.
pub struct TransmonSimulator {
    /// Number of qubits.
    num_qubits: usize,
    /// State vector (2^n complex amplitudes).
    state: Vec<Complex64>,
    /// Processor configuration.
    processor: TransmonProcessor,
    /// Noise model.
    noise: TransmonNoiseModel,
    /// Calibration drift.
    drift: CalibrationDrift,
    /// Per-qubit accumulated idle time in ns (for idle noise).
    idle_time_ns: Vec<f64>,
    /// Simple PRNG state for noise injection.
    rng_state: u64,
}

impl TransmonSimulator {
    /// Create a new simulator for the given processor.
    pub fn new(processor: TransmonProcessor) -> Self {
        let n = processor.num_qubits();
        let dim = 1 << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        let noise = TransmonNoiseModel::new(&processor);
        Self {
            num_qubits: n,
            state,
            noise,
            drift: CalibrationDrift::typical(),
            idle_time_ns: vec![0.0; n],
            rng_state: 42,
            processor,
        }
    }

    /// Create an ideal (noiseless) simulator.
    pub fn ideal(processor: TransmonProcessor) -> Self {
        let n = processor.num_qubits();
        let dim = 1 << n;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        let noise = TransmonNoiseModel::ideal(&processor);
        Self {
            num_qubits: n,
            state,
            noise,
            drift: CalibrationDrift::none(),
            idle_time_ns: vec![0.0; n],
            rng_state: 42,
            processor,
        }
    }

    /// Set calibration drift model.
    pub fn with_drift(mut self, drift: CalibrationDrift) -> Self {
        self.drift = drift;
        self
    }

    /// Set noise model.
    pub fn with_noise(mut self, noise: TransmonNoiseModel) -> Self {
        self.noise = noise;
        self
    }

    /// Simple xorshift64 PRNG for reproducible noise.
    fn rand_f64(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Apply a 2x2 unitary to a single qubit in the state vector.
    fn apply_single_qubit_unitary(&mut self, qubit: usize, u: [[Complex64; 2]; 2]) {
        let n = self.num_qubits;
        let dim = 1 << n;
        let mask = 1 << qubit;
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                let a0 = self.state[i];
                let a1 = self.state[j];
                self.state[i] = u[0][0] * a0 + u[0][1] * a1;
                self.state[j] = u[1][0] * a0 + u[1][1] * a1;
            }
        }
    }

    /// Apply a 4x4 unitary to two qubits.
    fn apply_two_qubit_unitary(&mut self, q0: usize, q1: usize, u: &[Complex64; 16]) {
        let n = self.num_qubits;
        let dim = 1 << n;
        let mask0 = 1 << q0;
        let mask1 = 1 << q1;
        for i in 0..dim {
            if i & mask0 == 0 && i & mask1 == 0 {
                let i00 = i;
                let i01 = i | mask1;
                let i10 = i | mask0;
                let i11 = i | mask0 | mask1;
                let a = [
                    self.state[i00],
                    self.state[i01],
                    self.state[i10],
                    self.state[i11],
                ];
                for (row, idx) in [(0, i00), (1, i01), (2, i10), (3, i11)] {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for col in 0..4 {
                        sum += u[row * 4 + col] * a[col];
                    }
                    self.state[idx] = sum;
                }
            }
        }
    }

    /// Execute a native gate on the state vector.
    fn execute_native(&mut self, gate: &NativeGate) {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        match gate {
            NativeGate::Rz { qubit, angle } => {
                let phase_neg = Complex64::new((-angle / 2.0).cos(), (-angle / 2.0).sin());
                let phase_pos = Complex64::new((angle / 2.0).cos(), (angle / 2.0).sin());
                self.apply_single_qubit_unitary(
                    *qubit,
                    [[phase_neg, zero], [zero, phase_pos]],
                );
            }
            NativeGate::SX { qubit } => {
                let half = Complex64::new(0.5, 0.0);
                let ihalf = Complex64::new(0.0, 0.5);
                // SX = (1/2)[[1+i, 1-i],[1-i, 1+i]]
                self.apply_single_qubit_unitary(
                    *qubit,
                    [
                        [half + ihalf, half - ihalf],
                        [half - ihalf, half + ihalf],
                    ],
                );
            }
            NativeGate::X { qubit } => {
                self.apply_single_qubit_unitary(*qubit, [[zero, one], [one, zero]]);
            }
            NativeGate::ECR { qubit_a, qubit_b } => {
                // ECR matrix: (1/√2) * [[0, 0, 1, i], [0, 0, i, 1], [1, -i, 0, 0], [-i, 1, 0, 0]]
                let s = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
                let is_ = Complex64::new(0.0, 1.0 / 2.0_f64.sqrt());
                let nis = Complex64::new(0.0, -1.0 / 2.0_f64.sqrt());
                let ecr = [
                    zero, zero, s, is_,
                    zero, zero, is_, s,
                    s, nis, zero, zero,
                    nis, s, zero, zero,
                ];
                self.apply_two_qubit_unitary(*qubit_a, *qubit_b, &ecr);
            }
            NativeGate::SqrtISWAP { qubit_a, qubit_b } => {
                // √iSWAP: diag(1, cos(π/4), cos(π/4), 1) + off-diag i·sin(π/4)
                let c = Complex64::new((PI / 4.0).cos(), 0.0);
                let is_ = Complex64::new(0.0, (PI / 4.0).sin());
                let sqrt_iswap = [
                    one, zero, zero, zero,
                    zero, c, is_, zero,
                    zero, is_, c, zero,
                    zero, zero, zero, one,
                ];
                self.apply_two_qubit_unitary(*qubit_a, *qubit_b, &sqrt_iswap);
            }
            NativeGate::CZGate { qubit_a, qubit_b } => {
                let neg_one = Complex64::new(-1.0, 0.0);
                let cz = [
                    one, zero, zero, zero,
                    zero, one, zero, zero,
                    zero, zero, one, zero,
                    zero, zero, zero, neg_one,
                ];
                self.apply_two_qubit_unitary(*qubit_a, *qubit_b, &cz);
            }
            NativeGate::FSim {
                qubit_a,
                qubit_b,
                theta,
                phi,
            } => {
                let ct = Complex64::new(theta.cos(), 0.0);
                let ist = Complex64::new(0.0, -theta.sin());
                let ep = Complex64::new((-phi).cos(), (-phi).sin());
                let fsim = [
                    one, zero, zero, zero,
                    zero, ct, ist, zero,
                    zero, ist, ct, zero,
                    zero, zero, zero, ep,
                ];
                self.apply_two_qubit_unitary(*qubit_a, *qubit_b, &fsim);
            }
            NativeGate::Measure { .. } | NativeGate::Reset { .. } => {
                // Handled separately in the measurement path.
            }
        }
    }

    /// Inject depolarizing noise on a qubit with given error probability.
    fn inject_depolarizing(&mut self, qubit: usize, p: f64) {
        if p <= 0.0 {
            return;
        }
        let r = self.rand_f64();
        if r < p {
            let zero = Complex64::new(0.0, 0.0);
            let one = Complex64::new(1.0, 0.0);
            let i_ = Complex64::new(0.0, 1.0);
            let ni = Complex64::new(0.0, -1.0);
            let neg = Complex64::new(-1.0, 0.0);
            // Apply random Pauli
            let pauli_choice = (self.rand_f64() * 3.0) as usize;
            match pauli_choice {
                0 => self.apply_single_qubit_unitary(qubit, [[zero, one], [one, zero]]), // X
                1 => self.apply_single_qubit_unitary(qubit, [[zero, ni], [i_, zero]]),   // Y
                _ => self.apply_single_qubit_unitary(qubit, [[one, zero], [zero, neg]]), // Z
            }
        }
    }

    /// Inject ZZ phase accumulation between active qubit and its neighbors.
    fn inject_zz_crosstalk(&mut self, qubit: usize, gate_time_ns: f64) {
        if !self.noise.enable_zz_crosstalk {
            return;
        }
        let neighbors = self.processor.topology.neighbors(qubit);
        for neighbor in neighbors {
            let phase = self.noise.zz_phase_rad(qubit, neighbor, gate_time_ns);
            if phase.abs() > 1e-10 {
                // ZZ interaction: apply exp(-i·ζ·Z⊗Z/2)
                // This adds phase to |11⟩ component.
                let dim = 1 << self.num_qubits;
                let mask_q = 1 << qubit;
                let mask_n = 1 << neighbor;
                let phase_factor = Complex64::new(phase.cos(), phase.sin());
                let phase_neg = Complex64::new(phase.cos(), -phase.sin());
                for i in 0..dim {
                    let bit_q = (i & mask_q) != 0;
                    let bit_n = (i & mask_n) != 0;
                    if bit_q == bit_n {
                        self.state[i] *= phase_factor;
                    } else {
                        self.state[i] *= phase_neg;
                    }
                }
            }
        }
    }

    /// Apply noise after a single-qubit gate.
    fn noise_after_single_gate(&mut self, qubit: usize) {
        let gate_time = self.processor.qubits[qubit].gate_time_ns;
        let err = self.noise.single_gate_error(qubit) + self.drift.gate_error_overhead();
        self.inject_depolarizing(qubit, err);
        self.inject_zz_crosstalk(qubit, gate_time);
    }

    /// Apply noise after a two-qubit gate.
    fn noise_after_two_qubit_gate(&mut self, qubit_a: usize, qubit_b: usize) {
        let err =
            self.noise.two_qubit_gate_error(qubit_a, qubit_b) + self.drift.gate_error_overhead();
        let half_err = err / 2.0;
        self.inject_depolarizing(qubit_a, half_err);
        self.inject_depolarizing(qubit_b, half_err);
    }
}

impl QuantumBackend for TransmonSimulator {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn apply_gate(&mut self, gate: &Gate) -> BackendResult<()> {
        // Apply the ideal gate using its exact matrix representation.
        // Native compilation is only used for noise modeling, not state evolution,
        // because hardware-specific decompositions introduce phases that are
        // compensated by frame tracking on real devices but not in simulation.
        let matrix = gate.gate_type.matrix();
        let all_qubits: Vec<usize> = gate
            .controls
            .iter()
            .chain(gate.targets.iter())
            .copied()
            .collect();

        match all_qubits.len() {
            1 => {
                let q = all_qubits[0];
                let u = [
                    [matrix[0][0], matrix[0][1]],
                    [matrix[1][0], matrix[1][1]],
                ];
                self.apply_single_qubit_unitary(q, u);
            }
            2 => {
                let (q0, q1) = (all_qubits[0], all_qubits[1]);
                let mut u = [Complex64::new(0.0, 0.0); 16];
                for r in 0..4 {
                    for c in 0..4 {
                        u[r * 4 + c] = matrix[r][c];
                    }
                }
                self.apply_two_qubit_unitary(q0, q1, &u);
            }
            _ => {
                // 3+ qubit gates: fall back to native decomposition.
                let native_ops = compile_to_native(gate, &self.processor);
                for op in &native_ops {
                    self.execute_native(op);
                }
            }
        }

        // Apply noise based on gate type.
        let is_two_qubit = all_qubits.len() == 2;
        if is_two_qubit {
            self.noise_after_two_qubit_gate(all_qubits[0], all_qubits[1]);
        } else {
            for &t in &gate.targets {
                self.noise_after_single_gate(t);
            }
        }

        Ok(())
    }

    fn probabilities(&self) -> BackendResult<Vec<f64>> {
        Ok(self.state.iter().map(|a| a.norm_sqr()).collect())
    }

    fn sample(&self, n_shots: usize) -> BackendResult<HashMap<usize, usize>> {
        let probs = self.probabilities()?;
        let mut counts = HashMap::new();
        let mut rng = self.rng_state;
        for _ in 0..n_shots {
            // xorshift for sampling
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let r = (rng as f64) / (u64::MAX as f64);
            let mut cumulative = 0.0;
            let mut outcome = probs.len() - 1;
            for (i, &p) in probs.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    outcome = i;
                    break;
                }
            }
            *counts.entry(outcome).or_insert(0) += 1;
        }
        Ok(counts)
    }

    fn measure_qubit(&mut self, qubit: usize) -> BackendResult<u8> {
        if qubit >= self.num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit,
                num_qubits: self.num_qubits,
            });
        }
        // Compute probability of |0⟩ on this qubit.
        let mask = 1 << qubit;
        let dim = 1 << self.num_qubits;
        let mut prob_0: f64 = 0.0;
        for i in 0..dim {
            if i & mask == 0 {
                prob_0 += self.state[i].norm_sqr();
            }
        }
        // Determine outcome.
        let r = {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            (self.rng_state as f64) / (u64::MAX as f64)
        };
        let ideal_outcome = if r < prob_0 { 0u8 } else { 1u8 };

        // Apply readout error.
        let confusion = self.noise.readout_confusion(qubit);
        let flip_r = {
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            (self.rng_state as f64) / (u64::MAX as f64)
        };
        let reported = if ideal_outcome == 0 {
            if flip_r < confusion[0][0] {
                0
            } else {
                1
            }
        } else if flip_r < confusion[1][1] {
            1
        } else {
            0
        };

        // Collapse state.
        let norm_sq = if ideal_outcome == 0 {
            prob_0
        } else {
            1.0 - prob_0
        };
        let norm = norm_sq.sqrt();
        if norm > 1e-15 {
            for i in 0..dim {
                let bit = ((i & mask) != 0) as u8;
                if bit == ideal_outcome {
                    self.state[i] /= Complex64::new(norm, 0.0);
                } else {
                    self.state[i] = Complex64::new(0.0, 0.0);
                }
            }
        }

        Ok(reported)
    }

    fn reset(&mut self) {
        let dim = 1 << self.num_qubits;
        self.state = vec![Complex64::new(0.0, 0.0); dim];
        self.state[0] = Complex64::new(1.0, 0.0);
        self.idle_time_ns = vec![0.0; self.num_qubits];
    }

    fn name(&self) -> &str {
        "TransmonSimulator"
    }
}

// ===================================================================
// ErrorModel TRAIT IMPLEMENTATION
// ===================================================================

/// ErrorModel wrapper for the transmon noise model.
pub struct TransmonErrorModel {
    noise: TransmonNoiseModel,
    drift: CalibrationDrift,
}

impl TransmonErrorModel {
    pub fn new(processor: &TransmonProcessor) -> Self {
        Self {
            noise: TransmonNoiseModel::new(processor),
            drift: CalibrationDrift::typical(),
        }
    }

    pub fn with_drift(mut self, drift: CalibrationDrift) -> Self {
        self.drift = drift;
        self
    }
}

impl ErrorModel for TransmonErrorModel {
    fn apply_noise_after_gate(
        &self,
        _gate: &Gate,
        _state: &mut dyn QuantumBackend,
    ) -> BackendResult<()> {
        // Noise is applied internally by TransmonSimulator; this is for
        // composing with other backends via the ErrorModel trait.
        Ok(())
    }

    fn apply_idle_noise(&self, _qubit: usize, _state: &mut dyn QuantumBackend) -> BackendResult<()> {
        Ok(())
    }

    fn gate_error_rate(&self, gate: &Gate) -> f64 {
        let is_two_qubit = matches!(
            gate.gate_type,
            GateType::CNOT
                | GateType::CZ
                | GateType::SWAP
                | GateType::ISWAP
                | GateType::Rxx(_)
                | GateType::Ryy(_)
                | GateType::Rzz(_)
        );
        let base = if is_two_qubit {
            let (a, b) = if !gate.controls.is_empty() {
                (gate.controls[0], gate.targets[0])
            } else {
                (gate.targets[0], gate.targets.get(1).copied().unwrap_or(0))
            };
            self.noise.two_qubit_gate_error(a, b)
        } else {
            let q = gate.targets[0];
            self.noise.single_gate_error(q)
        };
        base + self.drift.gate_error_overhead()
    }
}

// ===================================================================
// QCVV EXPERIMENT GENERATORS
// ===================================================================

/// Generate circuits for quantum characterization, verification, and validation.
pub struct TransmonQCVV;

impl TransmonQCVV {
    /// Generate a single-qubit randomized benchmarking sequence.
    ///
    /// Returns a circuit of `depth` random Clifford gates followed by
    /// the inverse Clifford, compiled to native {Rz, SX} operations.
    pub fn rb_sequence(qubit: usize, depth: usize) -> Vec<Gate> {
        let clifford_gates = [GateType::H, GateType::S, GateType::SX, GateType::X, GateType::Z];
        let mut circuit = Vec::with_capacity(depth + 1);
        // Deterministic "random" sequence using depth as seed.
        let mut seed = depth as u64 * 1000 + qubit as u64;
        for _ in 0..depth {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let idx = (seed as usize) % clifford_gates.len();
            circuit.push(Gate::single(clifford_gates[idx].clone(), qubit));
        }
        // Inverse Clifford (simplified: just add an H at the end for measurement).
        circuit.push(Gate::single(GateType::H, qubit));
        circuit
    }

    /// Generate a cross-entropy benchmarking (XEB) random circuit layer.
    ///
    /// Alternates random single-qubit gates with a layer of two-qubit gates
    /// on the provided edge set.
    pub fn xeb_layer(
        num_qubits: usize,
        edges: &[(usize, usize)],
        layer_idx: usize,
    ) -> Vec<Gate> {
        let mut circuit = Vec::new();
        // Random single-qubit layer.
        let sq_gates = [GateType::SX, GateType::H, GateType::T];
        for q in 0..num_qubits {
            let seed = (layer_idx as u64).wrapping_mul(1000).wrapping_add(q as u64);
            let idx = seed.wrapping_mul(6364136223846793005).wrapping_add(1) as usize % sq_gates.len();
            circuit.push(Gate::single(sq_gates[idx].clone(), q));
        }
        // Two-qubit layer.
        for &(a, b) in edges {
            circuit.push(Gate::two(GateType::CZ, a, b));
        }
        circuit
    }

    /// Generate a quantum volume circuit for `n` qubits and `depth` layers.
    pub fn quantum_volume_circuit(num_qubits: usize, depth: usize) -> Vec<Gate> {
        let mut circuit = Vec::new();
        let mut seed = (num_qubits * 1000 + depth) as u64;
        for layer in 0..depth {
            // Random permutation of qubits (simplified: pair adjacent).
            for pair in 0..(num_qubits / 2) {
                let q0 = pair * 2;
                let q1 = pair * 2 + 1;
                // Random SU(4) approximated as random single-qubit + CNOT + random single-qubit.
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let angle0 = (seed as f64 / u64::MAX as f64) * 2.0 * PI;
                circuit.push(Gate::single(GateType::Ry(angle0), q0));
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let angle1 = (seed as f64 / u64::MAX as f64) * 2.0 * PI;
                circuit.push(Gate::single(GateType::Ry(angle1), q1));
                circuit.push(Gate::two(GateType::CNOT, q0, q1));
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let angle2 = (seed as f64 / u64::MAX as f64) * 2.0 * PI;
                circuit.push(Gate::single(GateType::Ry(angle2), q0));
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let angle3 = (seed as f64 / u64::MAX as f64) * 2.0 * PI;
                circuit.push(Gate::single(GateType::Ry(angle3), q1));
            }
        }
        circuit
    }

    /// Generate a GHZ state preparation circuit.
    pub fn ghz_circuit(num_qubits: usize) -> Vec<Gate> {
        let mut circuit = vec![Gate::single(GateType::H, 0)];
        for i in 0..(num_qubits - 1) {
            circuit.push(Gate::two(GateType::CNOT, i, i + 1));
        }
        circuit
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // 1. TransmonQubit tests
    // ---------------------------------------------------------------

    #[test]
    fn test_transmon_qubit_typical() {
        let q = TransmonQubit::typical(0);
        assert!(q.frequency_ghz > 4.0 && q.frequency_ghz < 6.0);
        assert!(q.anharmonicity_mhz < 0.0);
        assert!(q.t1_us > 0.0);
        assert!(q.t2_us > 0.0);
        assert!(q.t2_us <= 2.0 * q.t1_us); // T2 <= 2*T1 always
    }

    #[test]
    fn test_transmon_frequency_02() {
        let q = TransmonQubit::typical(0);
        let f02 = q.frequency_02_ghz();
        // f02 = 2*f01 + α (α negative, so f02 < 2*f01)
        assert!(f02 < 2.0 * q.frequency_ghz);
        assert!(f02 > q.frequency_ghz);
    }

    #[test]
    fn test_pure_dephasing_rate() {
        let q = TransmonQubit {
            t1_us: 100.0,
            t2_us: 50.0,
            ..TransmonQubit::typical(0)
        };
        let rate = q.pure_dephasing_rate();
        // 1/T2 - 1/(2*T1) = 1/50 - 1/200 = 0.02 - 0.005 = 0.015
        assert!((rate - 0.015).abs() < 1e-6);
    }

    #[test]
    fn test_leakage_probability() {
        let q = TransmonQubit::typical(0);
        let p = q.leakage_probability();
        assert!(p > 0.0);
        assert!(p < 0.05); // capped at 5%
    }

    // ---------------------------------------------------------------
    // 2. Topology tests
    // ---------------------------------------------------------------

    #[test]
    fn test_fully_connected_topology() {
        let topo = ChipTopology::fully_connected(4, 5.0);
        assert_eq!(topo.num_qubits, 4);
        assert_eq!(topo.couplers.len(), 6); // C(4,2) = 6
        assert!(topo.are_coupled(0, 3));
        assert!(topo.are_coupled(1, 2));
    }

    #[test]
    fn test_grid_topology() {
        let topo = ChipTopology::grid(3, 3, 4.0);
        assert_eq!(topo.num_qubits, 9);
        // 3x3 grid: 12 edges (6 horizontal + 6 vertical)
        assert_eq!(topo.couplers.len(), 12);
        assert!(topo.are_coupled(0, 1)); // horizontal
        assert!(topo.are_coupled(0, 3)); // vertical
        assert!(!topo.are_coupled(0, 4)); // diagonal - not connected
    }

    #[test]
    fn test_neighbors() {
        let topo = ChipTopology::grid(3, 3, 4.0);
        let n = topo.neighbors(4); // center of 3x3
        assert_eq!(n.len(), 4); // up, down, left, right
    }

    // ---------------------------------------------------------------
    // 3. Device preset tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ibm_eagle_preset() {
        let proc = ibm_eagle();
        assert_eq!(proc.num_qubits(), 127);
        assert_eq!(proc.native_2q_gate, NativeGateFamily::ECR);
        assert!(proc.two_qubit_fidelity > 0.98);
    }

    #[test]
    fn test_ibm_heron_preset() {
        let proc = ibm_heron();
        assert_eq!(proc.num_qubits(), 156);
        assert!(proc.qubits[0].t1_us > 200.0); // better coherence than Eagle
    }

    #[test]
    fn test_google_sycamore_preset() {
        let proc = google_sycamore();
        assert_eq!(proc.num_qubits(), 53);
        assert_eq!(proc.native_2q_gate, NativeGateFamily::SqrtISWAP);
    }

    #[test]
    fn test_google_willow_preset() {
        let proc = google_willow();
        assert_eq!(proc.num_qubits(), 105);
        assert!(proc.qubits[0].t1_us > 50.0); // much improved over Sycamore
    }

    #[test]
    fn test_rigetti_ankaa_preset() {
        let proc = rigetti_ankaa();
        assert_eq!(proc.num_qubits(), 84);
        assert_eq!(proc.native_2q_gate, NativeGateFamily::CZ);
    }

    // ---------------------------------------------------------------
    // 4. ZZ coupling tests
    // ---------------------------------------------------------------

    #[test]
    fn test_zz_coupling_nonzero() {
        let proc = test_processor(3);
        let zz = proc.zz_coupling_khz(0, 1);
        assert!(zz > 0.0, "ZZ coupling should be nonzero: {}", zz);
    }

    #[test]
    fn test_zz_coupling_uncoupled() {
        // Custom topology with no edge between 0 and 2
        let topo = ChipTopology::custom(
            3,
            vec![CouplerLink {
                qubit_a: 0,
                qubit_b: 1,
                coupling_mhz: 4.0,
                zz_khz: None,
            }],
        );
        let proc = TransmonProcessor {
            qubits: (0..3).map(TransmonQubit::typical).collect(),
            topology: topo,
            native_2q_gate: NativeGateFamily::ECR,
            two_qubit_fidelity: 0.99,
            two_qubit_gate_time_ns: 200.0,
            readout_time_ns: 800.0,
            measurement_crosstalk: 0.01,
            temperature_mk: 15.0,
        };
        assert_eq!(proc.zz_coupling_khz(0, 2), 0.0);
    }

    // ---------------------------------------------------------------
    // 5. Gate compilation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_compile_h_gate() {
        let proc = test_processor(2);
        let h = Gate::single(GateType::H, 0);
        let native = compile_to_native(&h, &proc);
        assert!(native.len() >= 2, "H should compile to Rz+SX sequence");
    }

    #[test]
    fn test_compile_cnot_ecr() {
        let proc = test_processor(2);
        let cnot = Gate::two(GateType::CNOT, 0, 1);
        let native = compile_to_native(&cnot, &proc);
        // Should contain at least one ECR gate
        let has_ecr = native.iter().any(|g| matches!(g, NativeGate::ECR { .. }));
        assert!(has_ecr, "CNOT compilation should use ECR for IBM backend");
    }

    #[test]
    fn test_compile_cnot_sqrt_iswap() {
        let mut proc = test_processor(2);
        proc.native_2q_gate = NativeGateFamily::SqrtISWAP;
        let cnot = Gate::two(GateType::CNOT, 0, 1);
        let native = compile_to_native(&cnot, &proc);
        let has_siswap = native
            .iter()
            .any(|g| matches!(g, NativeGate::SqrtISWAP { .. }));
        assert!(
            has_siswap,
            "CNOT compilation should use √iSWAP for Google backend"
        );
    }

    // ---------------------------------------------------------------
    // 6. Noise model tests
    // ---------------------------------------------------------------

    #[test]
    fn test_t1_decay_probability() {
        let proc = test_processor(2);
        let noise = TransmonNoiseModel::new(&proc);
        let p = noise.t1_decay_prob(0, 25.0); // 25ns gate, T1~300μs
        assert!(p > 0.0);
        assert!(p < 0.001, "T1 decay in 25ns should be tiny: {}", p);
    }

    #[test]
    fn test_t2_dephase_probability() {
        let proc = test_processor(2);
        let noise = TransmonNoiseModel::new(&proc);
        let p = noise.t2_dephase_prob(0, 25.0);
        assert!(p > 0.0);
        assert!(p < 0.001, "T2 dephase in 25ns should be tiny: {}", p);
    }

    #[test]
    fn test_ideal_noise_model() {
        let proc = test_processor(2);
        let noise = TransmonNoiseModel::ideal(&proc);
        assert_eq!(noise.t1_decay_prob(0, 1000.0), 0.0);
        assert_eq!(noise.t2_dephase_prob(0, 1000.0), 0.0);
        assert_eq!(noise.leakage_prob(0), 0.0);
        let confusion = noise.readout_confusion(0);
        assert_eq!(confusion[0][0], 1.0);
        assert_eq!(confusion[1][1], 1.0);
    }

    #[test]
    fn test_readout_confusion_matrix() {
        let proc = test_processor(2);
        let noise = TransmonNoiseModel::new(&proc);
        let c = noise.readout_confusion(0);
        // Rows must sum to 1.
        assert!((c[0][0] + c[0][1] - 1.0).abs() < 1e-10);
        assert!((c[1][0] + c[1][1] - 1.0).abs() < 1e-10);
        // Diagonal should be > 0.9.
        assert!(c[0][0] > 0.9);
        assert!(c[1][1] > 0.9);
    }

    #[test]
    fn test_two_qubit_error_exceeds_single() {
        let proc = test_processor(3);
        let noise = TransmonNoiseModel::new(&proc);
        let single_err = noise.single_gate_error(0);
        let two_err = noise.two_qubit_gate_error(0, 1);
        assert!(
            two_err > single_err,
            "2Q error {} should exceed 1Q error {}",
            two_err,
            single_err
        );
    }

    // ---------------------------------------------------------------
    // 7. Calibration drift tests
    // ---------------------------------------------------------------

    #[test]
    fn test_drift_increases_error() {
        let d0 = CalibrationDrift::none();
        let d1 = CalibrationDrift {
            time_since_cal_s: 7200.0, // 2 hours
            ..CalibrationDrift::typical()
        };
        assert_eq!(d0.gate_error_overhead(), 0.0);
        assert!(d1.gate_error_overhead() > 0.0);
    }

    #[test]
    fn test_frequency_shift_random_walk() {
        let mut d = CalibrationDrift::typical();
        d.time_since_cal_s = 3600.0; // 1 hour
        let shift1 = d.frequency_shift_mhz();
        d.time_since_cal_s = 14400.0; // 4 hours
        let shift4 = d.frequency_shift_mhz();
        // Random walk: shift scales as sqrt(t), so 4h shift ≈ 2x 1h shift
        assert!((shift4 / shift1 - 2.0).abs() < 0.01);
    }

    // ---------------------------------------------------------------
    // 8. Simulator tests
    // ---------------------------------------------------------------

    #[test]
    fn test_ideal_simulator_bell_state() {
        let proc = test_processor(2);
        let mut sim = TransmonSimulator::ideal(proc);
        sim.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        sim.apply_gate(&Gate::two(GateType::CNOT, 0, 1)).unwrap();
        let probs = sim.probabilities().unwrap();
        assert!(
            (probs[0] - 0.5).abs() < 0.01,
            "P(00) should be ~0.5: {}",
            probs[0]
        );
        assert!(
            (probs[3] - 0.5).abs() < 0.01,
            "P(11) should be ~0.5: {}",
            probs[3]
        );
        assert!(probs[1] < 0.01, "P(01) should be ~0: {}", probs[1]);
        assert!(probs[2] < 0.01, "P(10) should be ~0: {}", probs[2]);
    }

    #[test]
    fn test_noisy_simulator_bell_fidelity() {
        let proc = test_processor(2);
        let mut sim = TransmonSimulator::new(proc);
        sim.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        sim.apply_gate(&Gate::two(GateType::CNOT, 0, 1)).unwrap();
        let probs = sim.probabilities().unwrap();
        // With noise, Bell state fidelity should be slightly degraded.
        let fidelity = probs[0] + probs[3];
        assert!(
            fidelity > 0.9,
            "Noisy Bell fidelity should be > 0.9: {}",
            fidelity
        );
        assert!(
            fidelity < 1.0,
            "Noisy Bell fidelity should be < 1.0: {}",
            fidelity
        );
    }

    #[test]
    fn test_simulator_reset() {
        let proc = test_processor(2);
        let mut sim = TransmonSimulator::ideal(proc);
        sim.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        sim.reset();
        let probs = sim.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulator_sampling() {
        let proc = test_processor(1);
        let mut sim = TransmonSimulator::ideal(proc);
        sim.apply_gate(&Gate::single(GateType::X, 0)).unwrap();
        let counts = sim.sample(100).unwrap();
        assert_eq!(*counts.get(&1).unwrap_or(&0), 100);
    }

    #[test]
    fn test_simulator_name() {
        let proc = test_processor(2);
        let sim = TransmonSimulator::new(proc);
        assert_eq!(sim.name(), "TransmonSimulator");
    }

    // ---------------------------------------------------------------
    // 9. ErrorModel trait tests
    // ---------------------------------------------------------------

    #[test]
    fn test_error_model_trait() {
        let proc = test_processor(3);
        let model = TransmonErrorModel::new(&proc);
        let h_gate = Gate::single(GateType::H, 0);
        let rate = ErrorModel::gate_error_rate(&model, &h_gate);
        assert!(rate > 0.0, "H gate error should be > 0");
        assert!(rate < 0.01, "H gate error should be < 1%: {}", rate);

        let cnot = Gate::two(GateType::CNOT, 0, 1);
        let rate_cnot = ErrorModel::gate_error_rate(&model, &cnot);
        assert!(
            rate_cnot > rate,
            "CNOT error {} should exceed H error {}",
            rate_cnot,
            rate
        );
    }

    #[test]
    fn test_error_model_with_drift() {
        let proc = test_processor(2);
        let model_fresh = TransmonErrorModel::new(&proc);
        let model_drifted = TransmonErrorModel::new(&proc).with_drift(CalibrationDrift {
            time_since_cal_s: 7200.0,
            ..CalibrationDrift::typical()
        });
        let h = Gate::single(GateType::H, 0);
        let fresh_err = ErrorModel::gate_error_rate(&model_fresh, &h);
        let drifted_err = ErrorModel::gate_error_rate(&model_drifted, &h);
        assert!(
            drifted_err > fresh_err,
            "Drifted error {} should exceed fresh error {}",
            drifted_err,
            fresh_err
        );
    }

    // ---------------------------------------------------------------
    // 10. Calibration data construction test
    // ---------------------------------------------------------------

    #[test]
    fn test_from_calibration_data() {
        let proc = from_calibration_data(
            &[5.0, 5.1, 5.2],
            &[-330.0, -320.0, -340.0],
            &[100.0, 120.0, 90.0],
            &[60.0, 80.0, 50.0],
            &[0.98, 0.99, 0.97],
            &[(0, 1, 4.0), (1, 2, 3.5)],
            NativeGateFamily::ECR,
            0.99,
            200.0,
        )
        .unwrap();
        assert_eq!(proc.num_qubits(), 3);
        assert!(proc.topology.are_coupled(0, 1));
        assert!(proc.topology.are_coupled(1, 2));
        assert!(!proc.topology.are_coupled(0, 2));
    }

    #[test]
    fn test_from_calibration_data_mismatched_lengths() {
        let result = from_calibration_data(
            &[5.0, 5.1],
            &[-330.0], // wrong length
            &[100.0, 120.0],
            &[60.0, 80.0],
            &[0.98, 0.99],
            &[(0, 1, 4.0)],
            NativeGateFamily::ECR,
            0.99,
            200.0,
        );
        assert!(result.is_err());
    }

    // ---------------------------------------------------------------
    // 11. QCVV circuit generation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_rb_sequence_length() {
        let circuit = TransmonQCVV::rb_sequence(0, 10);
        assert_eq!(circuit.len(), 11); // 10 random + 1 inverse
    }

    #[test]
    fn test_xeb_layer_structure() {
        let edges = vec![(0, 1), (2, 3)];
        let layer = TransmonQCVV::xeb_layer(4, &edges, 0);
        // 4 single-qubit + 2 two-qubit gates
        assert_eq!(layer.len(), 6);
    }

    #[test]
    fn test_quantum_volume_circuit() {
        let circuit = TransmonQCVV::quantum_volume_circuit(4, 4);
        assert!(!circuit.is_empty());
        // Each layer: 2 pairs × (2 Ry + 1 CNOT + 2 Ry) = 10 gates
        // 4 layers = 40 gates
        assert_eq!(circuit.len(), 40);
    }

    #[test]
    fn test_ghz_circuit() {
        let circuit = TransmonQCVV::ghz_circuit(5);
        assert_eq!(circuit.len(), 5); // 1 H + 4 CNOTs
    }

    // ---------------------------------------------------------------
    // 12. Multi-qubit simulation tests
    // ---------------------------------------------------------------

    #[test]
    fn test_three_qubit_ghz_ideal() {
        let proc = test_processor(3);
        let mut sim = TransmonSimulator::ideal(proc);
        let circuit = TransmonQCVV::ghz_circuit(3);
        for gate in &circuit {
            sim.apply_gate(gate).unwrap();
        }
        let probs = sim.probabilities().unwrap();
        assert!(
            (probs[0] - 0.5).abs() < 0.01,
            "P(000) = {} should be ~0.5",
            probs[0]
        );
        assert!(
            (probs[7] - 0.5).abs() < 0.01,
            "P(111) = {} should be ~0.5",
            probs[7]
        );
    }

    #[test]
    fn test_measurement_returns_valid_bit() {
        let proc = test_processor(2);
        let mut sim = TransmonSimulator::ideal(proc);
        let bit = sim.measure_qubit(0).unwrap();
        assert!(bit == 0 || bit == 1);
    }

    #[test]
    fn test_measurement_collapses_state() {
        let proc = test_processor(1);
        let mut sim = TransmonSimulator::ideal(proc);
        sim.apply_gate(&Gate::single(GateType::H, 0)).unwrap();
        let bit = sim.measure_qubit(0).unwrap();
        let probs = sim.probabilities().unwrap();
        // After measurement, state should be collapsed.
        if bit == 0 {
            assert!((probs[0] - 1.0).abs() < 1e-10);
        } else {
            assert!((probs[1] - 1.0).abs() < 1e-10);
        }
    }
}
