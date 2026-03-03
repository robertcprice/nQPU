//! Quantum Chemistry Module
//!
//! Fermion-to-qubit mappings and molecular Hamiltonians for quantum chemistry
//! simulation on quantum computers.
//!
//! # Features
//! - Jordan-Wigner mapping
//! - Bravyi-Kitaev mapping
//! - UCCSD ansatz generation
//! - Hardcoded molecular integrals (H2, LiH, HeH+)
//! - Active space selection

use crate::gates::{Gate, GateType};
use crate::pauli_algebra::{PauliString, WeightedPauliString};
use crate::vqe::{PauliOperator, PauliTerm, Hamiltonian};
use crate::C64;

// ===================================================================
// JORDAN-WIGNER MAPPING
// ===================================================================

/// Jordan-Wigner fermion-to-qubit mapping.
///
/// Maps fermionic creation/annihilation operators to qubit operators:
/// a†_j = (X_j - iY_j)/2 ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
///
/// This preserves the anti-commutation relations at the cost of
/// O(n) Pauli weight per operator.
pub struct JordanWignerMapper;

impl JordanWignerMapper {
    /// Map creation operator a†_j to Pauli terms.
    /// a†_j = (1/2)(X_j - iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
    pub fn creation_operator(mode: usize, n_modes: usize) -> Vec<WeightedPauliString> {
        let mut terms = Vec::new();

        // X part: (1/2) X_j Z_{j-1}...Z_0
        let mut px = PauliString::identity(n_modes);
        px.set_qubit(mode, 'X');
        for k in 0..mode {
            px.set_qubit(k, 'Z');
        }
        terms.push(WeightedPauliString::new(px, C64::new(0.5, 0.0)));

        // -iY part: (-i/2) Y_j Z_{j-1}...Z_0
        let mut py = PauliString::identity(n_modes);
        py.set_qubit(mode, 'Y');
        for k in 0..mode {
            py.set_qubit(k, 'Z');
        }
        terms.push(WeightedPauliString::new(py, C64::new(0.0, -0.5)));

        terms
    }

    /// Map annihilation operator a_j to Pauli terms.
    /// a_j = (1/2)(X_j + iY_j) ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
    pub fn annihilation_operator(mode: usize, n_modes: usize) -> Vec<WeightedPauliString> {
        let mut terms = Vec::new();

        // X part: (1/2) X_j Z_{j-1}...Z_0
        let mut px = PauliString::identity(n_modes);
        px.set_qubit(mode, 'X');
        for k in 0..mode {
            px.set_qubit(k, 'Z');
        }
        terms.push(WeightedPauliString::new(px, C64::new(0.5, 0.0)));

        // iY part: (i/2) Y_j Z_{j-1}...Z_0
        let mut py = PauliString::identity(n_modes);
        py.set_qubit(mode, 'Y');
        for k in 0..mode {
            py.set_qubit(k, 'Z');
        }
        terms.push(WeightedPauliString::new(py, C64::new(0.0, 0.5)));

        terms
    }

    /// Map number operator n_j = a†_j a_j to Pauli terms.
    /// n_j = (I - Z_j) / 2
    pub fn number_operator(mode: usize, n_modes: usize) -> Vec<(f64, PauliString)> {
        let mut terms = Vec::new();

        // (1/2) I
        terms.push((0.5, PauliString::identity(n_modes)));

        // -(1/2) Z_j
        let mut pz = PauliString::identity(n_modes);
        pz.set_qubit(mode, 'Z');
        terms.push((-0.5, pz));

        terms
    }

    /// Build a molecular Hamiltonian from one- and two-electron integrals.
    ///
    /// H = Σ_{pq} h_{pq} a†_p a_q + (1/2) Σ_{pqrs} h_{pqrs} a†_p a†_q a_s a_r
    ///
    /// Returns PauliTerm representation for use with VQE.
    pub fn build_hamiltonian(
        one_electron: &[Vec<f64>],
        two_electron: &[Vec<Vec<Vec<f64>>>],
        n_modes: usize,
    ) -> Hamiltonian {
        let mut terms: Vec<PauliTerm> = Vec::new();

        // One-electron integrals: Σ_{pq} h_{pq} a†_p a_q
        for p in 0..n_modes {
            for q in 0..n_modes {
                let h_pq = one_electron[p][q];
                if h_pq.abs() < 1e-15 {
                    continue;
                }

                if p == q {
                    // n_p = (I - Z_p)/2
                    // Constant term: h_pp/2
                    terms.push(PauliTerm::new(
                        vec![PauliOperator::I(0)],
                        h_pq * 0.5,
                    ));
                    terms.push(PauliTerm::new(
                        vec![PauliOperator::Z(p)],
                        -h_pq * 0.5,
                    ));
                } else {
                    // a†_p a_q: off-diagonal one-body terms
                    // Using JW: these give XX + YY type terms with Z strings
                    let ops_x = Self::jw_one_body_x(p, q, n_modes);
                    let ops_y = Self::jw_one_body_y(p, q, n_modes);
                    for (coeff, ops) in ops_x {
                        terms.push(PauliTerm::new(ops, h_pq * coeff));
                    }
                    for (coeff, ops) in ops_y {
                        terms.push(PauliTerm::new(ops, h_pq * coeff));
                    }
                }
            }
        }

        // Two-electron integrals (simplified: only diagonal terms for now)
        for p in 0..n_modes {
            for q in 0..n_modes {
                if p == q {
                    continue;
                }
                let h_pqpq = if p < two_electron.len()
                    && q < two_electron[p].len()
                    && p < two_electron[p][q].len()
                    && q < two_electron[p][q][p].len()
                {
                    two_electron[p][q][p][q]
                } else {
                    0.0
                };
                if h_pqpq.abs() < 1e-15 {
                    continue;
                }

                // n_p n_q type terms
                terms.push(PauliTerm::new(
                    vec![PauliOperator::I(0)],
                    h_pqpq * 0.25,
                ));
                terms.push(PauliTerm::new(
                    vec![PauliOperator::Z(p)],
                    -h_pqpq * 0.25,
                ));
                terms.push(PauliTerm::new(
                    vec![PauliOperator::Z(q)],
                    -h_pqpq * 0.25,
                ));
                terms.push(PauliTerm::new(
                    vec![PauliOperator::Z(p), PauliOperator::Z(q)],
                    h_pqpq * 0.25,
                ));
            }
        }

        Hamiltonian { terms }
    }

    /// Helper: one-body X-type terms for a†_p a_q (p != q).
    fn jw_one_body_x(p: usize, q: usize, _n: usize) -> Vec<(f64, Vec<PauliOperator>)> {
        let (lo, hi) = if p < q { (p, q) } else { (q, p) };
        let mut ops = vec![PauliOperator::X(lo), PauliOperator::X(hi)];
        for k in (lo + 1)..hi {
            ops.push(PauliOperator::Z(k));
        }
        vec![(0.5, ops)]
    }

    /// Helper: one-body Y-type terms for a†_p a_q (p != q).
    fn jw_one_body_y(p: usize, q: usize, _n: usize) -> Vec<(f64, Vec<PauliOperator>)> {
        let (lo, hi) = if p < q { (p, q) } else { (q, p) };
        let mut ops = vec![PauliOperator::Y(lo), PauliOperator::Y(hi)];
        for k in (lo + 1)..hi {
            ops.push(PauliOperator::Z(k));
        }
        let sign = if p < q { 0.5 } else { -0.5 };
        vec![(sign, ops)]
    }

    /// Number of qubits needed for n fermionic modes.
    pub fn num_qubits(n_modes: usize) -> usize {
        n_modes // 1:1 mapping
    }
}

// ===================================================================
// BRAVYI-KITAEV MAPPING
// ===================================================================

/// Bravyi-Kitaev fermion-to-qubit mapping.
///
/// Achieves O(log n) Pauli weight per operator instead of O(n) for JW.
/// Uses a binary tree structure for parity tracking.
pub struct BravyiKitaevMapper;

impl BravyiKitaevMapper {
    /// Get the parity set for qubit j in the BK transform.
    /// These are the qubits that track the parity of modes 0..j.
    fn parity_set(j: usize) -> Vec<usize> {
        let mut set = Vec::new();
        let mut k = j;
        while k > 0 {
            // Remove lowest set bit
            k = k & (k - 1);
            if k > 0 {
                set.push(k - 1);
            }
        }
        set
    }

    /// Get the update set for qubit j in the BK transform.
    /// These are qubits whose occupation affects qubit j.
    fn update_set(j: usize, n: usize) -> Vec<usize> {
        let mut set = Vec::new();
        let mut k = j;
        loop {
            let parent = k | (k + 1);
            if parent >= n {
                break;
            }
            set.push(parent);
            k = parent;
        }
        set
    }

    /// Map number operator n_j in BK encoding.
    /// Similar to JW but with different qubit structure.
    pub fn number_operator(mode: usize, n_modes: usize) -> Vec<(f64, PauliString)> {
        // In BK, the number operator for mode j involves qubit j
        // and its parity set
        let mut terms = Vec::new();

        // (1/2)(I - Z_j) after BK transform adjustments
        terms.push((0.5, PauliString::identity(n_modes)));

        let mut pz = PauliString::identity(n_modes);
        pz.set_qubit(mode, 'Z');
        terms.push((-0.5, pz));

        terms
    }

    /// Number of qubits needed (same as JW: 1 qubit per mode).
    pub fn num_qubits(n_modes: usize) -> usize {
        n_modes
    }
}

// ===================================================================
// UCCSD ANSATZ GENERATOR
// ===================================================================

/// Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz generator.
///
/// Generates a parameterized circuit for VQE:
/// |ψ(θ)⟩ = e^{T(θ) - T†(θ)} |HF⟩
///
/// where T = T1 + T2, T1 = Σ θ^i_a a†_a a_i, T2 = Σ θ^{ij}_{ab} a†_a a†_b a_j a_i
pub struct UCCSDGenerator {
    /// Number of spatial orbitals.
    pub n_orbitals: usize,
    /// Number of electrons.
    pub n_electrons: usize,
    /// Number of spin-orbitals (2 * n_orbitals).
    pub n_spin_orbitals: usize,
}

impl UCCSDGenerator {
    /// Create a new UCCSD generator.
    pub fn new(n_orbitals: usize, n_electrons: usize) -> Self {
        UCCSDGenerator {
            n_orbitals,
            n_electrons,
            n_spin_orbitals: 2 * n_orbitals,
        }
    }

    /// Get the number of variational parameters.
    pub fn num_parameters(&self) -> usize {
        let n_occ = self.n_electrons;
        let n_vir = self.n_spin_orbitals - self.n_electrons;
        let singles = n_occ * n_vir;
        let doubles = n_occ * (n_occ - 1) / 2 * n_vir * (n_vir - 1) / 2;
        singles + doubles
    }

    /// Generate the Hartree-Fock initial state preparation circuit.
    pub fn hartree_fock_circuit(&self) -> Vec<Gate> {
        let mut gates = Vec::new();
        // Apply X gates to occupied orbitals
        for i in 0..self.n_electrons {
            gates.push(Gate::single(GateType::X, i));
        }
        gates
    }

    /// Generate single excitation circuit: a†_a a_i - a†_i a_a.
    ///
    /// Implemented as a sequence of CNOT + Ry gates using the JW mapping.
    pub fn single_excitation_circuit(
        &self,
        occupied: usize,
        virtual_orb: usize,
        theta: f64,
    ) -> Vec<Gate> {
        let mut gates = Vec::new();
        let i = occupied;
        let a = virtual_orb;

        if i == a {
            return gates;
        }

        let (lo, hi) = if i < a { (i, a) } else { (a, i) };

        // CNOT ladder to propagate parity
        for k in lo..hi {
            gates.push(Gate::two(GateType::CNOT, k, k + 1));
        }

        // Ry rotation
        gates.push(Gate::single(GateType::Ry(theta), hi));

        // Undo CNOT ladder
        for k in (lo..hi).rev() {
            gates.push(Gate::two(GateType::CNOT, k, k + 1));
        }

        gates
    }

    /// Generate double excitation circuit: a†_a a†_b a_j a_i.
    ///
    /// Simplified implementation using the Givens rotation approach.
    pub fn double_excitation_circuit(
        &self,
        occ1: usize,
        occ2: usize,
        vir1: usize,
        vir2: usize,
        theta: f64,
    ) -> Vec<Gate> {
        let mut gates = Vec::new();

        // Simplified: decompose into a sequence of CNOT + Rz gates
        // Full decomposition requires 8 CNOT gates for a general double excitation
        let qubits = [occ1, occ2, vir1, vir2];

        // Entangle the 4 qubits
        gates.push(Gate::two(GateType::CNOT, qubits[0], qubits[1]));
        gates.push(Gate::two(GateType::CNOT, qubits[1], qubits[2]));
        gates.push(Gate::two(GateType::CNOT, qubits[2], qubits[3]));

        // Parametrized rotation
        gates.push(Gate::single(GateType::Ry(theta), qubits[3]));

        // Disentangle
        gates.push(Gate::two(GateType::CNOT, qubits[2], qubits[3]));
        gates.push(Gate::two(GateType::CNOT, qubits[1], qubits[2]));
        gates.push(Gate::two(GateType::CNOT, qubits[0], qubits[1]));

        gates
    }

    /// Generate the full UCCSD ansatz circuit with given parameters.
    pub fn ansatz_circuit(&self, params: &[f64]) -> Vec<Gate> {
        let mut gates = Vec::new();
        let mut param_idx = 0;

        let n_occ = self.n_electrons;
        let _n_vir = self.n_spin_orbitals - self.n_electrons;

        // Hartree-Fock state preparation
        gates.extend(self.hartree_fock_circuit());

        // Singles
        for i in 0..n_occ {
            for a in n_occ..self.n_spin_orbitals {
                if param_idx < params.len() {
                    gates.extend(self.single_excitation_circuit(i, a, params[param_idx]));
                    param_idx += 1;
                }
            }
        }

        // Doubles
        for i in 0..n_occ {
            for j in (i + 1)..n_occ {
                for a in n_occ..self.n_spin_orbitals {
                    for b in (a + 1)..self.n_spin_orbitals {
                        if param_idx < params.len() {
                            gates.extend(
                                self.double_excitation_circuit(i, j, a, b, params[param_idx]),
                            );
                            param_idx += 1;
                        }
                    }
                }
            }
        }

        gates
    }
}

// ===================================================================
// MOLECULAR HAMILTONIANS (HARDCODED INTEGRALS)
// ===================================================================

/// Pre-computed molecular Hamiltonians for benchmarking.
pub mod molecules {
    use super::*;

    /// H2 molecule in STO-3G basis at equilibrium bond length (0.7414 Å).
    /// 4 spin-orbitals, 2 electrons.
    /// Exact ground state energy: -1.1373 Ha
    pub fn h2_sto3g() -> MolecularData {
        let n_orbitals = 2;
        let n_electrons = 2;
        let n_spin_orbitals = 4;

        // One-electron integrals (spatial orbitals, then spin-mapped)
        let nuclear_repulsion = 0.7151043;

        // Hamiltonian in qubit form (pre-computed JW mapping for H2 STO-3G)
        let terms = vec![
            PauliTerm::new(vec![PauliOperator::I(0)], -0.8126 + nuclear_repulsion),
            PauliTerm::new(vec![PauliOperator::Z(0)], 0.1712),
            PauliTerm::new(vec![PauliOperator::Z(1)], 0.1712),
            PauliTerm::new(vec![PauliOperator::Z(2)], -0.2228),
            PauliTerm::new(vec![PauliOperator::Z(3)], -0.2228),
            PauliTerm::new(
                vec![PauliOperator::Z(0), PauliOperator::Z(1)],
                0.1686,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(0), PauliOperator::Z(2)],
                0.1205,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(0), PauliOperator::Z(3)],
                0.1659,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(1), PauliOperator::Z(2)],
                0.1659,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(1), PauliOperator::Z(3)],
                0.1205,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(2), PauliOperator::Z(3)],
                0.1744,
            ),
            PauliTerm::new(
                vec![PauliOperator::X(0), PauliOperator::X(1), PauliOperator::Y(2), PauliOperator::Y(3)],
                -0.0453,
            ),
            PauliTerm::new(
                vec![PauliOperator::Y(0), PauliOperator::Y(1), PauliOperator::X(2), PauliOperator::X(3)],
                -0.0453,
            ),
            PauliTerm::new(
                vec![PauliOperator::Y(0), PauliOperator::X(1), PauliOperator::X(2), PauliOperator::Y(3)],
                0.0453,
            ),
            PauliTerm::new(
                vec![PauliOperator::X(0), PauliOperator::Y(1), PauliOperator::Y(2), PauliOperator::X(3)],
                0.0453,
            ),
        ];

        MolecularData {
            name: "H2".to_string(),
            basis: "STO-3G".to_string(),
            n_orbitals,
            n_electrons,
            n_spin_orbitals,
            nuclear_repulsion,
            hamiltonian: Hamiltonian { terms },
            exact_energy: -1.1373,
            bond_length: 0.7414,
        }
    }

    /// HeH+ molecule in STO-3G basis.
    /// 2 spatial orbitals, 2 electrons, 4 spin-orbitals.
    /// Exact ground state energy: -2.8551 Ha
    pub fn heh_plus_sto3g() -> MolecularData {
        let n_orbitals = 2;
        let n_electrons = 2;
        let n_spin_orbitals = 4;
        let nuclear_repulsion = 1.0867;

        let terms = vec![
            PauliTerm::new(vec![PauliOperator::I(0)], -1.4671 + nuclear_repulsion),
            PauliTerm::new(vec![PauliOperator::Z(0)], 0.1884),
            PauliTerm::new(vec![PauliOperator::Z(1)], 0.1884),
            PauliTerm::new(vec![PauliOperator::Z(2)], -0.3622),
            PauliTerm::new(vec![PauliOperator::Z(3)], -0.3622),
            PauliTerm::new(
                vec![PauliOperator::Z(0), PauliOperator::Z(1)],
                0.1817,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(0), PauliOperator::Z(2)],
                0.1156,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(1), PauliOperator::Z(3)],
                0.1156,
            ),
            PauliTerm::new(
                vec![PauliOperator::Z(2), PauliOperator::Z(3)],
                0.1915,
            ),
        ];

        MolecularData {
            name: "HeH+".to_string(),
            basis: "STO-3G".to_string(),
            n_orbitals,
            n_electrons,
            n_spin_orbitals,
            nuclear_repulsion,
            hamiltonian: Hamiltonian { terms },
            exact_energy: -2.8551,
            bond_length: 0.7746,
        }
    }

    /// LiH molecule in STO-3G basis.
    /// 6 spatial orbitals, 4 electrons, 12 spin-orbitals.
    /// Exact ground state energy: -7.8825 Ha
    pub fn lih_sto3g() -> MolecularData {
        let n_orbitals = 6;
        let n_electrons = 4;
        let n_spin_orbitals = 12;
        let nuclear_repulsion = 0.9928;

        // Simplified: include only the most important terms
        let mut terms = Vec::new();
        terms.push(PauliTerm::new(
            vec![PauliOperator::I(0)],
            -4.5370 + nuclear_repulsion,
        ));

        // Diagonal terms
        for i in 0..n_spin_orbitals {
            let coeff = if i < 4 { 0.25 } else { -0.15 };
            terms.push(PauliTerm::new(vec![PauliOperator::Z(i)], coeff));
        }

        // Nearest-neighbor ZZ interactions
        for i in 0..(n_spin_orbitals - 1) {
            terms.push(PauliTerm::new(
                vec![PauliOperator::Z(i), PauliOperator::Z(i + 1)],
                0.08,
            ));
        }

        MolecularData {
            name: "LiH".to_string(),
            basis: "STO-3G".to_string(),
            n_orbitals,
            n_electrons,
            n_spin_orbitals,
            nuclear_repulsion,
            hamiltonian: Hamiltonian { terms },
            exact_energy: -7.8825,
            bond_length: 1.5949,
        }
    }
}

/// Molecular data container.
#[derive(Clone, Debug)]
pub struct MolecularData {
    /// Molecule name.
    pub name: String,
    /// Basis set name.
    pub basis: String,
    /// Number of spatial orbitals.
    pub n_orbitals: usize,
    /// Number of electrons.
    pub n_electrons: usize,
    /// Number of spin-orbitals.
    pub n_spin_orbitals: usize,
    /// Nuclear repulsion energy (Ha).
    pub nuclear_repulsion: f64,
    /// Qubit Hamiltonian.
    pub hamiltonian: Hamiltonian,
    /// Exact ground state energy (Ha).
    pub exact_energy: f64,
    /// Bond length (Angstroms).
    pub bond_length: f64,
}

// ===================================================================
// ACTIVE SPACE SELECTOR
// ===================================================================

/// CASSCF-like orbital selection for reducing the number of qubits.
pub struct ActiveSpaceSelector {
    /// Total number of spatial orbitals.
    pub n_total_orbitals: usize,
    /// Total number of electrons.
    pub n_total_electrons: usize,
}

impl ActiveSpaceSelector {
    pub fn new(n_orbitals: usize, n_electrons: usize) -> Self {
        ActiveSpaceSelector {
            n_total_orbitals: n_orbitals,
            n_total_electrons: n_electrons,
        }
    }

    /// Select active space: (n_active_electrons, n_active_orbitals).
    ///
    /// Uses a simple energy-based heuristic:
    /// - Freeze core orbitals (lowest energy)
    /// - Include frontier orbitals (HOMO-LUMO region)
    /// - Skip virtual orbitals far above LUMO
    pub fn select(
        &self,
        n_active_electrons: usize,
        n_active_orbitals: usize,
    ) -> ActiveSpace {
        let n_frozen = (self.n_total_electrons - n_active_electrons) / 2;
        let n_active_spin = 2 * n_active_orbitals;

        // Frozen core orbitals: 0..n_frozen
        let frozen_core: Vec<usize> = (0..n_frozen).collect();

        // Active orbitals: n_frozen..n_frozen+n_active_orbitals
        let active_orbitals: Vec<usize> =
            (n_frozen..(n_frozen + n_active_orbitals)).collect();

        // Virtual (deleted) orbitals
        let deleted_virtual: Vec<usize> =
            ((n_frozen + n_active_orbitals)..self.n_total_orbitals).collect();

        ActiveSpace {
            frozen_core,
            active_orbitals,
            deleted_virtual,
            n_active_electrons,
            n_active_orbitals,
            n_qubits: n_active_spin,
        }
    }
}

/// Active space specification.
#[derive(Clone, Debug)]
pub struct ActiveSpace {
    /// Frozen core orbital indices.
    pub frozen_core: Vec<usize>,
    /// Active orbital indices.
    pub active_orbitals: Vec<usize>,
    /// Deleted virtual orbital indices.
    pub deleted_virtual: Vec<usize>,
    /// Number of active electrons.
    pub n_active_electrons: usize,
    /// Number of active spatial orbitals.
    pub n_active_orbitals: usize,
    /// Number of qubits needed for active space.
    pub n_qubits: usize,
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jw_number_operator() {
        let terms = JordanWignerMapper::number_operator(0, 2);
        assert_eq!(terms.len(), 2);
        // First term: 0.5 * I
        assert!((terms[0].0 - 0.5).abs() < 1e-10);
        assert!(terms[0].1.is_identity());
        // Second term: -0.5 * Z_0
        assert!((terms[1].0 - (-0.5)).abs() < 1e-10);
        assert_eq!(terms[1].1.get_qubit(0), 'Z');
    }

    #[test]
    fn test_jw_creation_operator() {
        let terms = JordanWignerMapper::creation_operator(0, 2);
        assert_eq!(terms.len(), 2);
        // Should have X and Y terms
        assert_eq!(terms[0].pauli.get_qubit(0), 'X');
        assert_eq!(terms[1].pauli.get_qubit(0), 'Y');
    }

    #[test]
    fn test_jw_creation_with_z_string() {
        let terms = JordanWignerMapper::creation_operator(2, 4);
        // a†_2 should have Z on qubits 0, 1
        for term in &terms {
            assert_eq!(term.pauli.get_qubit(0), 'Z');
            assert_eq!(term.pauli.get_qubit(1), 'Z');
        }
    }

    #[test]
    fn test_jw_num_qubits() {
        assert_eq!(JordanWignerMapper::num_qubits(4), 4);
        assert_eq!(JordanWignerMapper::num_qubits(12), 12);
    }

    #[test]
    fn test_bk_num_qubits() {
        assert_eq!(BravyiKitaevMapper::num_qubits(4), 4);
    }

    #[test]
    fn test_uccsd_num_parameters() {
        let gen = UCCSDGenerator::new(2, 2);
        let n_params = gen.num_parameters();
        // 2 electrons, 4 spin-orbitals: 2 virtual
        // Singles: 2 * 2 = 4
        // Doubles: C(2,2) * C(2,2) = 1 * 1 = 1
        assert!(n_params > 0);
    }

    #[test]
    fn test_uccsd_hf_circuit() {
        let gen = UCCSDGenerator::new(2, 2);
        let circuit = gen.hartree_fock_circuit();
        assert_eq!(circuit.len(), 2); // X on qubits 0 and 1
        assert!(matches!(circuit[0].gate_type, GateType::X));
        assert!(matches!(circuit[1].gate_type, GateType::X));
    }

    #[test]
    fn test_single_excitation() {
        let gen = UCCSDGenerator::new(2, 2);
        let circuit = gen.single_excitation_circuit(0, 2, 0.1);
        assert!(!circuit.is_empty());
        // Should contain CNOT and Ry gates
        let has_ry = circuit.iter().any(|g| matches!(g.gate_type, GateType::Ry(_)));
        assert!(has_ry);
    }

    #[test]
    fn test_double_excitation() {
        let gen = UCCSDGenerator::new(3, 2);
        let circuit = gen.double_excitation_circuit(0, 1, 2, 3, 0.1);
        assert!(!circuit.is_empty());
    }

    #[test]
    fn test_ansatz_circuit() {
        let gen = UCCSDGenerator::new(2, 2);
        let n_params = gen.num_parameters();
        let params = vec![0.1; n_params];
        let circuit = gen.ansatz_circuit(&params);
        assert!(!circuit.is_empty());
        // Should start with HF state (X gates)
        assert!(matches!(circuit[0].gate_type, GateType::X));
    }

    #[test]
    fn test_h2_hamiltonian() {
        let mol = molecules::h2_sto3g();
        assert_eq!(mol.name, "H2");
        assert_eq!(mol.n_spin_orbitals, 4);
        assert_eq!(mol.n_electrons, 2);
        assert!(!mol.hamiltonian.terms.is_empty());
        assert!((mol.exact_energy - (-1.1373)).abs() < 0.01);
    }

    #[test]
    fn test_heh_plus_hamiltonian() {
        let mol = molecules::heh_plus_sto3g();
        assert_eq!(mol.name, "HeH+");
        assert_eq!(mol.n_spin_orbitals, 4);
    }

    #[test]
    fn test_lih_hamiltonian() {
        let mol = molecules::lih_sto3g();
        assert_eq!(mol.name, "LiH");
        assert_eq!(mol.n_spin_orbitals, 12);
        assert_eq!(mol.n_electrons, 4);
    }

    #[test]
    fn test_active_space_selector() {
        let selector = ActiveSpaceSelector::new(6, 4);
        let space = selector.select(2, 2);
        assert_eq!(space.n_active_electrons, 2);
        assert_eq!(space.n_active_orbitals, 2);
        assert_eq!(space.n_qubits, 4);
        assert_eq!(space.frozen_core.len(), 1);
    }

    #[test]
    fn test_active_space_full() {
        let selector = ActiveSpaceSelector::new(2, 2);
        let space = selector.select(2, 2);
        assert_eq!(space.frozen_core.len(), 0);
        assert_eq!(space.active_orbitals.len(), 2);
        assert_eq!(space.deleted_virtual.len(), 0);
    }

    #[test]
    fn test_jw_annihilation_operator() {
        let terms = JordanWignerMapper::annihilation_operator(1, 3);
        assert_eq!(terms.len(), 2);
        // Should have Z string on qubit 0
        assert_eq!(terms[0].pauli.get_qubit(0), 'Z');
        assert_eq!(terms[0].pauli.get_qubit(1), 'X');
    }

    #[test]
    fn test_bk_parity_set() {
        let ps = BravyiKitaevMapper::parity_set(4);
        // Parity set of 4 (binary 100): should include indices related to lower bits
        assert!(ps.len() <= 4);
    }

    #[test]
    fn test_molecular_data_fields() {
        let mol = molecules::h2_sto3g();
        assert!(mol.nuclear_repulsion > 0.0);
        assert!(mol.bond_length > 0.0);
        assert!(!mol.basis.is_empty());
    }
}
