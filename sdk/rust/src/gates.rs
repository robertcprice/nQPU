//! Quantum Gate Definitions
//!
//! This module provides a unified interface for quantum gates
//! with support for optimization, and execution.

/// Quantum gate types
#[derive(Clone, Debug, PartialEq)]
pub enum GateType {
    /// Single-qubit gates
    H, // Hadamard
    X,       // Pauli-X (NOT)
    Y,       // Pauli-Y
    Z,       // Pauli-Z
    S,       // Phase gate (π/2)
    T,       // π/8 gate
    Rx(f64), // Rotation around X
    Ry(f64), // Rotation around Y
    Rz(f64), // Rotation around Z
    U {
        theta: f64,
        phi: f64,
        lambda: f64,
    }, // Universal single-qubit gate

    /// Two-qubit gates
    CNOT, // Controlled-NOT
    CZ,   // Controlled-Z
    SWAP, // Swap two qubits

    /// Three-qubit gates
    Toffoli, // Controlled-controlled-NOT

    /// Parameterized gates
    CRx(f64), // Controlled rotation around X
    CRy(f64), // Controlled rotation around Y
    CRz(f64), // Controlled rotation around Z
    CR(f64),  // Controlled phase rotation

    /// Additional standard gates
    SX, // √X gate (half-X)
    Phase(f64), // Phase gate P(θ) = diag(1, e^iθ)
    ISWAP,      // iSWAP gate
    CCZ,        // Controlled-controlled-Z

    /// Two-qubit rotation gates
    Rxx(f64), // XX rotation: exp(-i * theta/2 * XX)
    Ryy(f64), // YY rotation: exp(-i * theta/2 * YY)
    Rzz(f64), // ZZ rotation: exp(-i * theta/2 * ZZ)

    /// Three-qubit gates (additional)
    CSWAP, // Controlled-SWAP (Fredkin gate)

    /// Generic controlled-U gate
    CU {
        theta: f64,
        phi: f64,
        lambda: f64,
        gamma: f64,
    },

    /// Custom unitary (for advanced use)
    Custom(Vec<Vec<C64>>),
}

// Use the crate's C64 type
use crate::C64;

impl GateType {
    /// Get the matrix representation of this gate
    pub fn matrix(&self) -> Vec<Vec<C64>> {
        match self {
            GateType::H => vec![
                vec![
                    C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                ],
                vec![
                    C64::new(1.0 / 2.0_f64.sqrt(), 0.0),
                    C64::new(-1.0 / 2.0_f64.sqrt(), 0.0),
                ],
            ],
            GateType::X => vec![
                vec![C64::new(0.0, 0.0), C64::new(1.0, 0.0)],
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
            ],
            GateType::Y => vec![
                vec![C64::new(0.0, 0.0), C64::new(0.0, -1.0)],
                vec![C64::new(0.0, 1.0), C64::new(0.0, 0.0)],
            ],
            GateType::Z => vec![
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                vec![C64::new(0.0, 0.0), C64::new(-1.0, 0.0)],
            ],
            GateType::S => vec![
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                vec![C64::new(0.0, 0.0), C64::new(0.0, 1.0)],
            ],
            GateType::T => vec![
                vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                vec![
                    C64::new(0.0, 0.0),
                    C64::new(
                        (std::f64::consts::PI / 4.0).cos(),
                        (std::f64::consts::PI / 4.0).sin(),
                    ),
                ],
            ],
            GateType::Rx(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![C64::new(cos, 0.0), C64::new(0.0, -sin)],
                    vec![C64::new(0.0, -sin), C64::new(cos, 0.0)],
                ]
            }
            GateType::Ry(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![C64::new(cos, 0.0), C64::new(-sin, 0.0)],
                    vec![C64::new(sin, 0.0), C64::new(cos, 0.0)],
                ]
            }
            GateType::Rz(theta) => {
                vec![
                    vec![
                        C64::new((-theta / 2.0).cos(), (-theta / 2.0).sin()),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new((theta / 2.0).cos(), (theta / 2.0).sin()),
                    ],
                ]
            }
            GateType::U { theta, phi, lambda } => {
                let cos_theta_2 = (theta / 2.0).cos();
                let sin_theta_2 = (theta / 2.0).sin();
                vec![
                    vec![
                        C64::new(cos_theta_2, 0.0),
                        C64::new(-sin_theta_2 * lambda.cos(), -sin_theta_2 * lambda.sin()),
                    ],
                    vec![
                        C64::new(sin_theta_2 * phi.cos(), sin_theta_2 * phi.sin()),
                        C64::new(
                            cos_theta_2 * (phi + lambda).cos(),
                            cos_theta_2 * (phi + lambda).sin(),
                        ),
                    ],
                ]
            }
            GateType::CNOT => {
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                ]
            }
            GateType::CZ => {
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(-1.0, 0.0),
                    ],
                ]
            }
            GateType::SWAP => {
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                    ],
                ]
            }
            GateType::Toffoli => {
                // 8x8 matrix for Toffoli (CCNOT)
                let mut matrix = vec![vec![C64::new(0.0, 0.0); 8]; 8];
                for i in 0..6 {
                    matrix[i][i] = C64::new(1.0, 0.0);
                }
                matrix[6][7] = C64::new(1.0, 0.0);
                matrix[7][6] = C64::new(1.0, 0.0);
                matrix
            }
            GateType::SX => {
                // √X = (1+i)/2 * [[1, -i], [-i, 1]] = [[½(1+i), ½(1-i)], [½(1-i), ½(1+i)]]
                vec![
                    vec![C64::new(0.5, 0.5), C64::new(0.5, -0.5)],
                    vec![C64::new(0.5, -0.5), C64::new(0.5, 0.5)],
                ]
            }
            GateType::Phase(theta) => {
                // P(θ) = diag(1, e^iθ)
                vec![
                    vec![C64::new(1.0, 0.0), C64::new(0.0, 0.0)],
                    vec![C64::new(0.0, 0.0), C64::new(theta.cos(), theta.sin())],
                ]
            }
            GateType::ISWAP => {
                // iSWAP: |01⟩ → i|10⟩, |10⟩ → i|01⟩
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 1.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 1.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                    ],
                ]
            }
            GateType::CCZ => {
                // CCZ: 8x8 identity with |111⟩ → -|111⟩
                let mut matrix = vec![vec![C64::new(0.0, 0.0); 8]; 8];
                for i in 0..8 {
                    matrix[i][i] = if i == 7 {
                        C64::new(-1.0, 0.0)
                    } else {
                        C64::new(1.0, 0.0)
                    };
                }
                matrix
            }
            GateType::Custom(m) => m.clone(),
            // Controlled rotations (4x4 matrices)
            GateType::CRx(angle) => {
                let cos = (angle / 2.0).cos();
                let sin = (angle / 2.0).sin();
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                        C64::new(0.0, -sin),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, -sin),
                        C64::new(cos, 0.0),
                    ],
                ]
            }
            GateType::CRy(angle) => {
                let cos = (angle / 2.0).cos();
                let sin = (angle / 2.0).sin();
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                        C64::new(-sin, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(sin, 0.0),
                        C64::new(cos, 0.0),
                    ],
                ]
            }
            GateType::CRz(angle) => {
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new((-angle / 2.0).cos(), (-angle / 2.0).sin()),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new((angle / 2.0).cos(), (angle / 2.0).sin()),
                    ],
                ]
            }
            GateType::CR(angle) => {
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(angle.cos(), angle.sin()),
                    ],
                ]
            }

            // -- Two-qubit XX rotation: exp(-i * theta/2 * XX) --
            GateType::Rxx(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![
                        C64::new(cos, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, -sin),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                        C64::new(0.0, -sin),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, -sin),
                        C64::new(cos, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, -sin),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                    ],
                ]
            }

            // -- Two-qubit YY rotation: exp(-i * theta/2 * YY) --
            GateType::Ryy(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![
                        C64::new(cos, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, sin),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                        C64::new(0.0, -sin),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, -sin),
                        C64::new(cos, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, sin),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, 0.0),
                    ],
                ]
            }

            // -- Two-qubit ZZ rotation: exp(-i * theta/2 * ZZ) --
            GateType::Rzz(theta) => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![
                        C64::new(cos, -sin),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(cos, sin),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, sin),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(cos, -sin),
                    ],
                ]
            }

            // -- Controlled-SWAP (Fredkin gate) -- 8x8 matrix
            GateType::CSWAP => {
                let mut matrix = vec![vec![C64::new(0.0, 0.0); 8]; 8];
                // |000> -> |000>
                matrix[0][0] = C64::new(1.0, 0.0);
                // |001> -> |001>
                matrix[1][1] = C64::new(1.0, 0.0);
                // |010> -> |010>
                matrix[2][2] = C64::new(1.0, 0.0);
                // |011> -> |011>
                matrix[3][3] = C64::new(1.0, 0.0);
                // |100> -> |100>
                matrix[4][4] = C64::new(1.0, 0.0);
                // |101> -> |110> (swap targets when control=1)
                matrix[5][6] = C64::new(1.0, 0.0);
                // |110> -> |101> (swap targets when control=1)
                matrix[6][5] = C64::new(1.0, 0.0);
                // |111> -> |111>
                matrix[7][7] = C64::new(1.0, 0.0);
                matrix
            }

            // -- Generic Controlled-U gate -- CU(theta, phi, lambda, gamma)
            // 4x4 matrix:
            //   |0><0| x I + e^(i*gamma) |1><1| x U(theta, phi, lambda)
            GateType::CU { theta, phi, lambda, gamma } => {
                let cos = (theta / 2.0).cos();
                let sin = (theta / 2.0).sin();
                vec![
                    vec![
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(1.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(gamma.cos() * cos, gamma.sin() * cos),
                        C64::new(
                            -(gamma + lambda).cos() * sin,
                            -(gamma + lambda).sin() * sin,
                        ),
                    ],
                    vec![
                        C64::new(0.0, 0.0),
                        C64::new(0.0, 0.0),
                        C64::new(
                            (gamma + phi).cos() * sin,
                            (gamma + phi).sin() * sin,
                        ),
                        C64::new(
                            (gamma + phi + lambda).cos() * cos,
                            (gamma + phi + lambda).sin() * cos,
                        ),
                    ],
                ]
            }
        }
    }

    /// Check if this gate is its own inverse
    pub fn is_self_inverse(&self) -> bool {
        matches!(
            self,
            GateType::H
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::CNOT
                | GateType::CZ
                | GateType::SWAP
                | GateType::Toffoli
                | GateType::CCZ
                | GateType::CSWAP
        )
    }

    /// Get the inverse of this gate
    pub fn inverse(&self) -> GateType {
        match self {
            GateType::H => GateType::H,
            GateType::X => GateType::X,
            GateType::Y => GateType::Y,
            GateType::Z => GateType::Z,
            GateType::S => GateType::Rz(-std::f64::consts::PI / 2.0), // S† = Rz(-π/2)
            GateType::T => GateType::Rz(-std::f64::consts::PI / 4.0), // T† = Rz(-π/4)
            GateType::Rx(theta) => GateType::Rx(-theta),
            GateType::Ry(theta) => GateType::Ry(-theta),
            GateType::Rz(theta) => GateType::Rz(-theta),
            GateType::U { theta, phi, lambda } => GateType::U {
                theta: -theta,
                phi: -lambda,
                lambda: -phi,
            },
            GateType::CNOT => GateType::CNOT,
            GateType::CZ => GateType::CZ,
            GateType::SWAP => GateType::SWAP,
            GateType::Toffoli => GateType::Toffoli,
            GateType::CSWAP => GateType::CSWAP,
            GateType::Rxx(theta) => GateType::Rxx(-theta),
            GateType::Ryy(theta) => GateType::Ryy(-theta),
            GateType::Rzz(theta) => GateType::Rzz(-theta),
            _ => self.clone(),
        }
    }
}

/// A quantum gate with target and control qubits
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub gate_type: GateType,
    pub targets: Vec<usize>,
    pub controls: Vec<usize>,
    pub params: Option<Vec<f64>>,
}

impl Gate {
    /// Create a new gate
    pub fn new(gate_type: GateType, targets: Vec<usize>, controls: Vec<usize>) -> Self {
        Gate {
            gate_type,
            targets,
            controls,
            params: None,
        }
    }

    /// Create a parameterized gate
    pub fn with_params(
        gate_type: GateType,
        targets: Vec<usize>,
        controls: Vec<usize>,
        params: Vec<f64>,
    ) -> Self {
        Gate {
            gate_type,
            targets,
            controls,
            params: Some(params),
        }
    }

    /// Convenience constructor for single-qubit gates
    pub fn single(gate_type: GateType, target: usize) -> Self {
        Gate::new(gate_type, vec![target], vec![])
    }

    /// Convenience constructor for two-qubit gates
    pub fn two(gate_type: GateType, control: usize, target: usize) -> Self {
        Gate::new(gate_type, vec![target], vec![control])
    }

    /// Get the number of qubits this gate operates on
    pub fn num_qubits(&self) -> usize {
        self.targets.len() + self.controls.len()
    }

    /// Check if this is a single-qubit gate
    pub fn is_single_qubit(&self) -> bool {
        self.targets.len() == 1 && self.controls.is_empty()
    }

    /// Check if this is a two-qubit gate
    pub fn is_two_qubit(&self) -> bool {
        self.num_qubits() == 2
    }
}

/// Common gate constructors
impl Gate {
    pub fn h(target: usize) -> Self {
        Gate::single(GateType::H, target)
    }
    pub fn x(target: usize) -> Self {
        Gate::single(GateType::X, target)
    }
    pub fn y(target: usize) -> Self {
        Gate::single(GateType::Y, target)
    }
    pub fn z(target: usize) -> Self {
        Gate::single(GateType::Z, target)
    }
    pub fn s(target: usize) -> Self {
        Gate::single(GateType::S, target)
    }
    pub fn t(target: usize) -> Self {
        Gate::single(GateType::T, target)
    }
    pub fn rx(target: usize, angle: f64) -> Self {
        Gate::single(GateType::Rx(angle), target)
    }
    pub fn ry(target: usize, angle: f64) -> Self {
        Gate::single(GateType::Ry(angle), target)
    }
    pub fn rz(target: usize, angle: f64) -> Self {
        Gate::single(GateType::Rz(angle), target)
    }
    pub fn cnot(control: usize, target: usize) -> Self {
        Gate::two(GateType::CNOT, control, target)
    }
    pub fn cz(control: usize, target: usize) -> Self {
        Gate::two(GateType::CZ, control, target)
    }
    pub fn swap(a: usize, b: usize) -> Self {
        Gate::new(GateType::SWAP, vec![a, b], vec![])
    }
    pub fn toffoli(control1: usize, control2: usize, target: usize) -> Self {
        Gate::new(GateType::Toffoli, vec![target], vec![control1, control2])
    }
    pub fn sx(target: usize) -> Self {
        Gate::single(GateType::SX, target)
    }
    pub fn phase(target: usize, angle: f64) -> Self {
        Gate::single(GateType::Phase(angle), target)
    }
    pub fn iswap(a: usize, b: usize) -> Self {
        Gate::new(GateType::ISWAP, vec![a, b], vec![])
    }
    pub fn ccz(control1: usize, control2: usize, target: usize) -> Self {
        Gate::new(GateType::CCZ, vec![target], vec![control1, control2])
    }
    pub fn rxx(a: usize, b: usize, angle: f64) -> Self {
        Gate::new(GateType::Rxx(angle), vec![a, b], vec![])
    }
    pub fn ryy(a: usize, b: usize, angle: f64) -> Self {
        Gate::new(GateType::Ryy(angle), vec![a, b], vec![])
    }
    pub fn rzz(a: usize, b: usize, angle: f64) -> Self {
        Gate::new(GateType::Rzz(angle), vec![a, b], vec![])
    }
    pub fn cswap(control: usize, target1: usize, target2: usize) -> Self {
        Gate::new(GateType::CSWAP, vec![target1, target2], vec![control])
    }
    pub fn cu(control: usize, target: usize, theta: f64, phi: f64, lambda: f64, gamma: f64) -> Self {
        Gate::new(
            GateType::CU { theta, phi, lambda, gamma },
            vec![target],
            vec![control],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_creation() {
        let h = Gate::h(0);
        assert_eq!(h.targets, vec![0usize]);
        assert_eq!(h.controls, Vec::<usize>::new());

        let cnot = Gate::cnot(0, 1);
        assert_eq!(cnot.targets, vec![1usize]);
        assert_eq!(cnot.controls, vec![0usize]);
    }

    #[test]
    fn test_self_inverse() {
        assert!(GateType::H.is_self_inverse());
        assert!(GateType::X.is_self_inverse());
        assert!(!GateType::Rx(0.5).is_self_inverse());
    }
}
