//! Quantum Drug Design Module
//!
//! Quantum algorithms for drug discovery: molecular docking scoring, generative
//! molecular design, binding affinity estimation, and ADMET property prediction
//! using quantum circuits.
//!
//! Based on Nature Biotechnology 2025 results showing 21.5% improvement over
//! classical-only approaches for lead compound identification.
//!
//! # Algorithms
//!
//! - **Quantum Force Field**: VQE-based ground state energy of interaction Hamiltonians
//! - **Molecular Docking**: QUBO-encoded protein-ligand scoring with QAOA optimization
//! - **Quantum Kernel Similarity**: Fidelity-based molecular fingerprint kernels
//! - **Generative Design**: Born machine sampling from |psi(theta)|^2 distributions
//! - **ADMET Prediction**: QNN classifiers for absorption, distribution, metabolism,
//!   excretion, toxicity, solubility, LogP, and BBB permeability
//! - **Drug-Likeness**: Lipinski Rule of Five, QED score, synthetic accessibility
//! - **Lead Optimization**: Multi-objective Pareto optimization of binding + ADMET
//!
//! # References
//!
//! - Cao et al. (2019) - Quantum Chemistry in the Age of Quantum Computing
//! - Barkoutsos et al. (2021) - Quantum algorithms for molecular docking
//! - Nature Biotechnology (2025) - Quantum-enhanced drug discovery pipelines

use num_complex::Complex64;
use std::f64::consts::PI;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from quantum drug design operations.
#[derive(Debug, Clone)]
pub enum DrugDesignError {
    /// The molecular structure is invalid or incomplete.
    InvalidMolecule(String),
    /// A simulation step failed.
    SimulationFailed(String),
    /// VQE or optimization did not converge within budget.
    ConvergenceFailure { iterations: usize, energy: f64 },
    /// Feature extraction or encoding error.
    FeatureError(String),
}

impl std::fmt::Display for DrugDesignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DrugDesignError::InvalidMolecule(msg) => write!(f, "Invalid molecule: {}", msg),
            DrugDesignError::SimulationFailed(msg) => write!(f, "Simulation failed: {}", msg),
            DrugDesignError::ConvergenceFailure { iterations, energy } => {
                write!(
                    f,
                    "Convergence failure after {} iterations (energy={:.6})",
                    iterations, energy
                )
            }
            DrugDesignError::FeatureError(msg) => write!(f, "Feature error: {}", msg),
        }
    }
}

impl std::error::Error for DrugDesignError {}

pub type DrugResult<T> = Result<T, DrugDesignError>;

// ===================================================================
// CHEMICAL ELEMENTS
// ===================================================================

/// Chemical element with atomic properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Element {
    H,
    C,
    N,
    O,
    F,
    S,
    P,
    Cl,
    Br,
    Fe,
    Zn,
}

impl Element {
    /// Atomic number.
    pub fn atomic_number(&self) -> usize {
        match self {
            Element::H => 1,
            Element::C => 6,
            Element::N => 7,
            Element::O => 8,
            Element::F => 9,
            Element::S => 16,
            Element::P => 15,
            Element::Cl => 17,
            Element::Br => 35,
            Element::Fe => 26,
            Element::Zn => 30,
        }
    }

    /// Approximate atomic mass in Daltons.
    pub fn atomic_mass(&self) -> f64 {
        match self {
            Element::H => 1.008,
            Element::C => 12.011,
            Element::N => 14.007,
            Element::O => 15.999,
            Element::F => 18.998,
            Element::S => 32.065,
            Element::P => 30.974,
            Element::Cl => 35.453,
            Element::Br => 79.904,
            Element::Fe => 55.845,
            Element::Zn => 65.380,
        }
    }

    /// van der Waals radius in Angstroms.
    pub fn vdw_radius(&self) -> f64 {
        match self {
            Element::H => 1.20,
            Element::C => 1.70,
            Element::N => 1.55,
            Element::O => 1.52,
            Element::F => 1.47,
            Element::S => 1.80,
            Element::P => 1.80,
            Element::Cl => 1.75,
            Element::Br => 1.85,
            Element::Fe => 2.00,
            Element::Zn => 1.39,
        }
    }

    /// Electronegativity (Pauling scale).
    pub fn electronegativity(&self) -> f64 {
        match self {
            Element::H => 2.20,
            Element::C => 2.55,
            Element::N => 3.04,
            Element::O => 3.44,
            Element::F => 3.98,
            Element::S => 2.58,
            Element::P => 2.19,
            Element::Cl => 3.16,
            Element::Br => 2.96,
            Element::Fe => 1.83,
            Element::Zn => 1.65,
        }
    }

    /// Whether this element is a typical hydrogen bond donor atom.
    pub fn is_hbond_donor_heavy(&self) -> bool {
        matches!(self, Element::N | Element::O)
    }

    /// Whether this element is a typical hydrogen bond acceptor.
    pub fn is_hbond_acceptor(&self) -> bool {
        matches!(self, Element::N | Element::O | Element::F)
    }
}

// ===================================================================
// BOND TYPES
// ===================================================================

/// Chemical bond type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

impl BondType {
    /// Bond order as a floating point value.
    pub fn order(&self) -> f64 {
        match self {
            BondType::Single => 1.0,
            BondType::Double => 2.0,
            BondType::Triple => 3.0,
            BondType::Aromatic => 1.5,
        }
    }

    /// Whether this bond is rotatable (single bonds between non-terminal atoms).
    pub fn is_rotatable(&self) -> bool {
        matches!(self, BondType::Single)
    }
}

// ===================================================================
// ATOM AND BOND
// ===================================================================

/// A single atom with element, 3D position, and partial charge.
#[derive(Debug, Clone)]
pub struct Atom {
    pub element: Element,
    pub position: [f64; 3],
    pub charge: f64,
}

/// A bond connecting two atoms.
#[derive(Debug, Clone)]
pub struct Bond {
    pub atom1: usize,
    pub atom2: usize,
    pub bond_type: BondType,
}

// ===================================================================
// MOLECULE
// ===================================================================

/// A molecular structure with atoms, bonds, and metadata.
#[derive(Debug, Clone)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
    pub name: String,
    pub smiles: Option<String>,
}

impl Molecule {
    /// Create a new empty molecule.
    pub fn new(name: &str) -> Self {
        Molecule {
            atoms: Vec::new(),
            bonds: Vec::new(),
            name: name.to_string(),
            smiles: None,
        }
    }

    /// Add an atom to the molecule and return its index.
    pub fn add_atom(&mut self, element: Element, position: [f64; 3], charge: f64) -> usize {
        let idx = self.atoms.len();
        self.atoms.push(Atom {
            element,
            position,
            charge,
        });
        idx
    }

    /// Add a bond between two atoms.
    pub fn add_bond(&mut self, atom1: usize, atom2: usize, bond_type: BondType) {
        self.bonds.push(Bond {
            atom1,
            atom2,
            bond_type,
        });
    }

    /// Molecular weight in Daltons.
    pub fn molecular_weight(&self) -> f64 {
        self.atoms.iter().map(|a| a.element.atomic_mass()).sum()
    }

    /// Number of hydrogen bond donors (N-H and O-H groups).
    pub fn h_bond_donors(&self) -> usize {
        let mut count = 0;
        for bond in &self.bonds {
            let e1 = self.atoms[bond.atom1].element;
            let e2 = self.atoms[bond.atom2].element;
            if (e1 == Element::H && e2.is_hbond_donor_heavy())
                || (e2 == Element::H && e1.is_hbond_donor_heavy())
            {
                count += 1;
            }
        }
        count
    }

    /// Number of hydrogen bond acceptors (N, O, F atoms).
    pub fn h_bond_acceptors(&self) -> usize {
        self.atoms
            .iter()
            .filter(|a| a.element.is_hbond_acceptor())
            .count()
    }

    /// Number of rotatable bonds (single bonds between non-hydrogen, non-terminal atoms).
    pub fn rotatable_bonds(&self) -> usize {
        self.bonds
            .iter()
            .filter(|b| {
                b.bond_type.is_rotatable()
                    && self.atoms[b.atom1].element != Element::H
                    && self.atoms[b.atom2].element != Element::H
                    && self.neighbor_count(b.atom1) > 1
                    && self.neighbor_count(b.atom2) > 1
            })
            .count()
    }

    /// Count neighbors of an atom.
    fn neighbor_count(&self, atom_idx: usize) -> usize {
        self.bonds
            .iter()
            .filter(|b| b.atom1 == atom_idx || b.atom2 == atom_idx)
            .count()
    }

    /// Estimate LogP using Wildman-Crippen atom contributions (simplified).
    pub fn estimated_log_p(&self) -> f64 {
        let mut log_p = 0.0;
        for atom in &self.atoms {
            log_p += match atom.element {
                Element::C => 0.1441,
                Element::H => 0.1230,
                Element::N => -0.5262,
                Element::O => -0.2893,
                Element::F => 0.4118,
                Element::S => 0.6482,
                Element::P => 0.2836,
                Element::Cl => 0.6895,
                Element::Br => 0.8813,
                Element::Fe => -0.3808,
                Element::Zn => -0.3808,
            };
        }
        // Adjust for aromatic bonds
        let aromatic_count = self
            .bonds
            .iter()
            .filter(|b| b.bond_type == BondType::Aromatic)
            .count();
        log_p += aromatic_count as f64 * 0.08;
        log_p
    }

    /// Estimate polar surface area (Ertl method, simplified).
    pub fn polar_surface_area(&self) -> f64 {
        let mut psa = 0.0;
        for atom in &self.atoms {
            psa += match atom.element {
                Element::N => 23.79,
                Element::O => 20.23,
                Element::S => 25.30,
                Element::P => 34.14,
                _ => 0.0,
            };
        }
        // Subtract for each H bonded to N or O (already counted in heavy atom)
        let nh_oh_count = self.h_bond_donors();
        psa -= nh_oh_count as f64 * 3.5;
        psa.max(0.0)
    }

    /// Distance between two atoms in Angstroms.
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        let a = &self.atoms[i].position;
        let b = &self.atoms[j].position;
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// Geometric center of the molecule.
    pub fn center_of_mass(&self) -> [f64; 3] {
        if self.atoms.is_empty() {
            return [0.0; 3];
        }
        let total_mass: f64 = self.atoms.iter().map(|a| a.element.atomic_mass()).sum();
        let mut com = [0.0; 3];
        for atom in &self.atoms {
            let m = atom.element.atomic_mass();
            for k in 0..3 {
                com[k] += atom.position[k] * m;
            }
        }
        for k in 0..3 {
            com[k] /= total_mass;
        }
        com
    }

    /// Validate the molecule structure.
    pub fn validate(&self) -> DrugResult<()> {
        if self.atoms.is_empty() {
            return Err(DrugDesignError::InvalidMolecule(
                "Molecule has no atoms".to_string(),
            ));
        }
        for bond in &self.bonds {
            if bond.atom1 >= self.atoms.len() || bond.atom2 >= self.atoms.len() {
                return Err(DrugDesignError::InvalidMolecule(format!(
                    "Bond references atom index out of range: ({}, {})",
                    bond.atom1, bond.atom2
                )));
            }
        }
        Ok(())
    }
}

// ===================================================================
// DRUG PROPERTIES AND DRUG-LIKENESS
// ===================================================================

/// Computed drug-relevant properties.
#[derive(Debug, Clone)]
pub struct DrugProperties {
    pub molecular_weight: f64,
    pub log_p: f64,
    pub h_bond_donors: usize,
    pub h_bond_acceptors: usize,
    pub rotatable_bonds: usize,
    pub polar_surface_area: f64,
}

/// Result of drug-likeness evaluation.
#[derive(Debug, Clone)]
pub struct DrugLikenessResult {
    pub lipinski_violations: usize,
    pub qed_score: f64,
    pub synthetic_accessibility: f64,
    pub properties: DrugProperties,
}

/// Evaluate drug-likeness for a molecule.
pub fn evaluate_drug_likeness(mol: &Molecule) -> DrugLikenessResult {
    let props = DrugProperties {
        molecular_weight: mol.molecular_weight(),
        log_p: mol.estimated_log_p(),
        h_bond_donors: mol.h_bond_donors(),
        h_bond_acceptors: mol.h_bond_acceptors(),
        rotatable_bonds: mol.rotatable_bonds(),
        polar_surface_area: mol.polar_surface_area(),
    };

    // Lipinski Rule of Five violations
    let mut violations = 0;
    if props.molecular_weight > 500.0 {
        violations += 1;
    }
    if props.log_p > 5.0 {
        violations += 1;
    }
    if props.h_bond_donors > 5 {
        violations += 1;
    }
    if props.h_bond_acceptors > 10 {
        violations += 1;
    }

    // QED: Quantitative Estimate of Drug-likeness
    // Weighted geometric mean of desirability functions (Bickerton et al. 2012)
    let d_mw = desirability_gaussian(props.molecular_weight, 330.0, 90.0);
    let d_logp = desirability_gaussian(props.log_p, 2.5, 1.5);
    let d_hbd = desirability_step_down(props.h_bond_donors as f64, 3.5);
    let d_hba = desirability_step_down(props.h_bond_acceptors as f64, 7.0);
    let d_psa = desirability_gaussian(props.polar_surface_area, 76.0, 35.0);
    let d_rotb = desirability_step_down(props.rotatable_bonds as f64, 6.0);

    let weights = [0.66, 0.46, 0.61, 0.32, 0.06, 0.65];
    let desirabilities = [d_mw, d_logp, d_hbd, d_hba, d_psa, d_rotb];
    let w_sum: f64 = weights.iter().sum();

    let mut log_qed = 0.0;
    for (w, d) in weights.iter().zip(desirabilities.iter()) {
        log_qed += w * d.max(1e-6).ln();
    }
    let qed = (log_qed / w_sum).exp().clamp(0.0, 1.0);

    // Synthetic accessibility: estimate from atom count and ring complexity
    let ring_count = estimate_ring_count(mol);
    let atom_diversity = count_distinct_elements(mol);
    let sa_raw = 1.0 + 0.5 * (ring_count as f64) + 0.3 * (atom_diversity as f64);
    let sa = (sa_raw / 10.0).clamp(0.0, 1.0);
    // Invert: lower raw score = easier to synthesize = higher accessibility
    let synthetic_accessibility = 1.0 - sa;

    DrugLikenessResult {
        lipinski_violations: violations,
        qed_score: qed,
        synthetic_accessibility,
        properties: props,
    }
}

/// Gaussian desirability function centered at `mean` with width `sigma`.
fn desirability_gaussian(x: f64, mean: f64, sigma: f64) -> f64 {
    (-0.5 * ((x - mean) / sigma).powi(2)).exp()
}

/// Step-down desirability: 1.0 below threshold, decaying above.
fn desirability_step_down(x: f64, threshold: f64) -> f64 {
    if x <= threshold {
        1.0
    } else {
        (-0.5 * ((x - threshold) / 1.5).powi(2)).exp()
    }
}

/// Estimate ring count from bond connectivity (simplified: Euler formula).
fn estimate_ring_count(mol: &Molecule) -> usize {
    if mol.atoms.is_empty() {
        return 0;
    }
    let n = mol.atoms.len();
    let e = mol.bonds.len();
    // For a connected graph: rings = edges - vertices + 1
    // For possibly disconnected: rings = edges - vertices + components
    let components = count_components(mol);
    if e + components > n {
        e + components - n
    } else {
        0
    }
}

/// Count connected components via union-find.
fn count_components(mol: &Molecule) -> usize {
    let n = mol.atoms.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for bond in &mol.bonds {
        let a = find(&mut parent, bond.atom1);
        let b = find(&mut parent, bond.atom2);
        if a != b {
            parent[a] = b;
        }
    }

    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}

/// Count distinct element types in a molecule.
fn count_distinct_elements(mol: &Molecule) -> usize {
    let mut seen = std::collections::HashSet::new();
    for atom in &mol.atoms {
        seen.insert(atom.element.atomic_number());
    }
    seen.len()
}

// ===================================================================
// MOLECULAR FINGERPRINT
// ===================================================================

/// Binary molecular fingerprint for quantum kernel computations.
#[derive(Debug, Clone)]
pub struct MolecularFingerprint {
    pub bits: Vec<bool>,
    pub num_bits: usize,
}

impl MolecularFingerprint {
    /// Generate a fingerprint from a molecule (path-based, simplified Morgan-like).
    pub fn from_molecule(mol: &Molecule, num_bits: usize) -> Self {
        let mut bits = vec![false; num_bits];

        // Hash atom types
        for atom in &mol.atoms {
            let h = atom.element.atomic_number() * 7919;
            bits[h % num_bits] = true;
        }

        // Hash bond types
        for bond in &mol.bonds {
            let e1 = mol.atoms[bond.atom1].element.atomic_number();
            let e2 = mol.atoms[bond.atom2].element.atomic_number();
            let bt = bond.bond_type.order() as usize;
            let h = (e1 * 31 + e2 * 37 + bt * 41) % num_bits;
            bits[h] = true;
            // Also hash reverse direction
            let h2 = (e2 * 31 + e1 * 37 + bt * 41) % num_bits;
            bits[h2] = true;
        }

        // Hash atom pairs within 2 bonds (simplified Morgan radius=1)
        for bond in &mol.bonds {
            let neighbors_a: Vec<usize> = mol
                .bonds
                .iter()
                .filter(|b| b.atom1 == bond.atom1 || b.atom2 == bond.atom1)
                .map(|b| {
                    if b.atom1 == bond.atom1 {
                        b.atom2
                    } else {
                        b.atom1
                    }
                })
                .collect();
            for &nb in &neighbors_a {
                let e_center = mol.atoms[bond.atom1].element.atomic_number();
                let e_nb = mol.atoms[nb].element.atomic_number();
                let e_far = mol.atoms[bond.atom2].element.atomic_number();
                let h = (e_center * 53 + e_nb * 59 + e_far * 61) % num_bits;
                bits[h] = true;
            }
        }

        MolecularFingerprint { bits, num_bits }
    }

    /// Tanimoto similarity between two fingerprints.
    pub fn tanimoto(&self, other: &MolecularFingerprint) -> f64 {
        let len = self.bits.len().min(other.bits.len());
        let mut both = 0usize;
        let mut either = 0usize;
        for i in 0..len {
            if self.bits[i] && other.bits[i] {
                both += 1;
            }
            if self.bits[i] || other.bits[i] {
                either += 1;
            }
        }
        if either == 0 {
            1.0
        } else {
            both as f64 / either as f64
        }
    }
}

// ===================================================================
// LIGHTWEIGHT QUANTUM STATE (self-contained, no crate:: dependency)
// ===================================================================

/// Minimal statevector for drug design quantum circuits.
/// Self-contained to avoid coupling to the main simulator state type.
#[derive(Debug, Clone)]
struct QState {
    amps: Vec<Complex64>,
    n: usize,
}

impl QState {
    fn new(num_qubits: usize) -> Self {
        let dim = 1usize << num_qubits;
        let mut amps = vec![Complex64::new(0.0, 0.0); dim];
        amps[0] = Complex64::new(1.0, 0.0);
        QState {
            amps,
            n: num_qubits,
        }
    }

    fn dim(&self) -> usize {
        self.amps.len()
    }

    /// Apply Hadamard to qubit q.
    fn h(&mut self, q: usize) {
        let inv = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let dim = self.dim();
        let mask = 1usize << q;
        let mut i = 0;
        while i < dim {
            // Skip to next pair where bit q is 0
            let i0 = if i & mask == 0 { i } else { i + 1 };
            if i0 >= dim {
                break;
            }
            let i1 = i0 | mask;
            let a = self.amps[i0];
            let b = self.amps[i1];
            self.amps[i0] = inv * (a + b);
            self.amps[i1] = inv * (a - b);
            // Advance: skip both i0 and i1
            i = i0 + 1;
            if i == i1 {
                i = i1 + 1;
            }
        }
    }

    /// Apply RY(theta) to qubit q.
    fn ry(&mut self, q: usize, theta: f64) {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        let dim = self.dim();
        let mask = 1usize << q;
        for i0 in 0..dim {
            if i0 & mask != 0 {
                continue;
            }
            let i1 = i0 | mask;
            let a = self.amps[i0];
            let b = self.amps[i1];
            self.amps[i0] = Complex64::new(c, 0.0) * a - Complex64::new(s, 0.0) * b;
            self.amps[i1] = Complex64::new(s, 0.0) * a + Complex64::new(c, 0.0) * b;
        }
    }

    /// Apply RZ(theta) to qubit q.
    fn rz(&mut self, q: usize, theta: f64) {
        let dim = self.dim();
        let mask = 1usize << q;
        let phase_0 = Complex64::new((theta / 2.0).cos(), -(theta / 2.0).sin());
        let phase_1 = Complex64::new((theta / 2.0).cos(), (theta / 2.0).sin());
        for i in 0..dim {
            if i & mask == 0 {
                self.amps[i] = self.amps[i] * phase_0;
            } else {
                self.amps[i] = self.amps[i] * phase_1;
            }
        }
    }

    /// Apply RX(theta) to qubit q.
    fn rx(&mut self, q: usize, theta: f64) {
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = Complex64::new(0.0, -(theta / 2.0).sin());
        let dim = self.dim();
        let mask = 1usize << q;
        for i0 in 0..dim {
            if i0 & mask != 0 {
                continue;
            }
            let i1 = i0 | mask;
            let a = self.amps[i0];
            let b = self.amps[i1];
            self.amps[i0] = c * a + s * b;
            self.amps[i1] = s * a + c * b;
        }
    }

    /// Apply CNOT (control, target).
    fn cnot(&mut self, ctrl: usize, tgt: usize) {
        let dim = self.dim();
        let cmask = 1usize << ctrl;
        let tmask = 1usize << tgt;
        for i in 0..dim {
            if (i & cmask != 0) && (i & tmask == 0) {
                let j = i | tmask;
                self.amps.swap(i, j);
            }
        }
    }

    /// Probability of measuring |i>.
    fn prob(&self, i: usize) -> f64 {
        self.amps[i].norm_sqr()
    }

    /// All probabilities.
    fn probabilities(&self) -> Vec<f64> {
        self.amps.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Inner product <self|other>.
    fn inner(&self, other: &QState) -> Complex64 {
        let len = self.amps.len().min(other.amps.len());
        let mut sum = Complex64::new(0.0, 0.0);
        for i in 0..len {
            sum += self.amps[i].conj() * other.amps[i];
        }
        sum
    }

    /// Expectation of Z on qubit q.
    fn expect_z(&self, q: usize) -> f64 {
        let dim = self.dim();
        let mask = 1usize << q;
        let mut val = 0.0;
        for i in 0..dim {
            let p = self.amps[i].norm_sqr();
            if i & mask == 0 {
                val += p;
            } else {
                val -= p;
            }
        }
        val
    }
}

// ===================================================================
// QUANTUM FORCE FIELD
// ===================================================================

/// Quantum force field energy calculator using VQE-inspired approach.
pub struct QuantumForceField {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl Default for QuantumForceField {
    fn default() -> Self {
        QuantumForceField {
            num_qubits: 4,
            num_layers: 2,
            max_iterations: 100,
            convergence_threshold: 1e-4,
        }
    }
}

impl QuantumForceField {
    pub fn new(num_qubits: usize) -> Self {
        QuantumForceField {
            num_qubits,
            ..Default::default()
        }
    }

    /// Compute the interaction energy between two molecules.
    /// Returns the quantum-corrected energy in kcal/mol.
    pub fn interaction_energy(&self, mol_a: &Molecule, mol_b: &Molecule) -> DrugResult<f64> {
        mol_a.validate()?;
        mol_b.validate()?;

        // Classical force field terms
        let coulomb = self.coulomb_energy(mol_a, mol_b);
        let vdw = self.van_der_waals_energy(mol_a, mol_b);
        let hbond = self.hydrogen_bond_energy(mol_a, mol_b);

        // Quantum correction via VQE
        let quantum_correction = self.vqe_correction(mol_a, mol_b)?;

        Ok(coulomb + vdw + hbond + quantum_correction)
    }

    /// Coulomb interaction energy (kcal/mol).
    fn coulomb_energy(&self, mol_a: &Molecule, mol_b: &Molecule) -> f64 {
        let k_e = 332.06; // Coulomb constant in kcal*A/(mol*e^2)
        let mut energy = 0.0;
        for a in &mol_a.atoms {
            for b in &mol_b.atoms {
                let dx = a.position[0] - b.position[0];
                let dy = a.position[1] - b.position[1];
                let dz = a.position[2] - b.position[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                energy += k_e * a.charge * b.charge / r;
            }
        }
        energy
    }

    /// Lennard-Jones van der Waals energy (kcal/mol).
    fn van_der_waals_energy(&self, mol_a: &Molecule, mol_b: &Molecule) -> f64 {
        let mut energy = 0.0;
        for a in &mol_a.atoms {
            for b in &mol_b.atoms {
                let sigma = (a.element.vdw_radius() + b.element.vdw_radius()) / 2.0;
                let epsilon = 0.05; // kcal/mol (simplified)
                let dx = a.position[0] - b.position[0];
                let dy = a.position[1] - b.position[1];
                let dz = a.position[2] - b.position[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let sr6 = (sigma / r).powi(6);
                energy += 4.0 * epsilon * (sr6 * sr6 - sr6);
            }
        }
        energy
    }

    /// Hydrogen bond energy estimate (kcal/mol).
    fn hydrogen_bond_energy(&self, mol_a: &Molecule, mol_b: &Molecule) -> f64 {
        let mut energy = 0.0;
        // Check for donor-acceptor pairs between molecules
        for a in &mol_a.atoms {
            if !a.element.is_hbond_donor_heavy() && !a.element.is_hbond_acceptor() {
                continue;
            }
            for b in &mol_b.atoms {
                if !b.element.is_hbond_donor_heavy() && !b.element.is_hbond_acceptor() {
                    continue;
                }
                // One must be donor, other acceptor
                let is_pair = (a.element.is_hbond_donor_heavy() && b.element.is_hbond_acceptor())
                    || (a.element.is_hbond_acceptor() && b.element.is_hbond_donor_heavy());
                if !is_pair {
                    continue;
                }
                let dx = a.position[0] - b.position[0];
                let dy = a.position[1] - b.position[1];
                let dz = a.position[2] - b.position[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                // H-bond is strongest around 2.8 A
                if r < 3.5 {
                    let e_hb = -5.0 * (-(r - 2.8).powi(2) / 0.5).exp();
                    energy += e_hb;
                }
            }
        }
        energy
    }

    /// VQE quantum correction to classical force field.
    /// Encodes interaction geometry into a variational circuit and optimizes.
    fn vqe_correction(&self, mol_a: &Molecule, mol_b: &Molecule) -> DrugResult<f64> {
        let n = self.num_qubits;

        // Encode molecular features as initial rotation angles
        let features = encode_interaction_features(mol_a, mol_b, n);

        // Build Hamiltonian coefficients from molecular properties
        let h_coeffs = self.build_hamiltonian_coefficients(mol_a, mol_b);

        // Variational optimization
        let num_params = n * self.num_layers * 3; // RY, RZ, RX per qubit per layer
        let mut params: Vec<f64> = (0..num_params)
            .map(|i| 0.1 * ((i as f64 * 0.7) % 1.0))
            .collect();

        let mut best_energy = f64::MAX;
        let step_size = 0.05;

        for iter in 0..self.max_iterations {
            let energy = self.evaluate_energy(&features, &params, &h_coeffs);

            if energy < best_energy {
                best_energy = energy;
            }

            if iter > 0 && (best_energy - energy).abs() < self.convergence_threshold {
                return Ok(best_energy);
            }

            // Parameter-shift gradient
            let mut gradients = vec![0.0; num_params];
            let shift = PI / 2.0;
            for p in 0..num_params {
                let mut p_plus = params.clone();
                let mut p_minus = params.clone();
                p_plus[p] += shift;
                p_minus[p] -= shift;
                let e_plus = self.evaluate_energy(&features, &p_plus, &h_coeffs);
                let e_minus = self.evaluate_energy(&features, &p_minus, &h_coeffs);
                gradients[p] = (e_plus - e_minus) / 2.0;
            }

            for p in 0..num_params {
                params[p] -= step_size * gradients[p];
            }
        }

        Ok(best_energy)
    }

    /// Build Hamiltonian coefficients from molecular interaction.
    fn build_hamiltonian_coefficients(&self, mol_a: &Molecule, mol_b: &Molecule) -> Vec<f64> {
        let n = self.num_qubits;
        let mut coeffs = Vec::with_capacity(n + n * (n - 1) / 2);

        // Single-qubit Z coefficients from charge interaction
        let total_charge_a: f64 = mol_a.atoms.iter().map(|a| a.charge).sum();
        let total_charge_b: f64 = mol_b.atoms.iter().map(|a| a.charge).sum();
        let charge_interaction = total_charge_a * total_charge_b;

        for i in 0..n {
            let angle = (i as f64 + 1.0) / (n as f64);
            coeffs.push(charge_interaction * angle * 0.1);
        }

        // ZZ coefficients from distance-dependent coupling
        let com_a = mol_a.center_of_mass();
        let com_b = mol_b.center_of_mass();
        let dist = ((com_a[0] - com_b[0]).powi(2)
            + (com_a[1] - com_b[1]).powi(2)
            + (com_a[2] - com_b[2]).powi(2))
        .sqrt()
        .max(0.1);

        for i in 0..n {
            for j in (i + 1)..n {
                let coupling = 0.05 / (1.0 + dist * 0.1) * ((i + j) as f64 / n as f64);
                coeffs.push(coupling);
            }
        }

        coeffs
    }

    /// Evaluate energy of the variational ansatz with given parameters.
    fn evaluate_energy(&self, features: &[f64], params: &[f64], h_coeffs: &[f64]) -> f64 {
        let n = self.num_qubits;
        let mut state = QState::new(n);

        // Feature encoding layer
        for q in 0..n {
            state.ry(q, features[q % features.len()]);
        }

        // Variational layers
        let mut pidx = 0;
        for _layer in 0..self.num_layers {
            for q in 0..n {
                state.ry(q, params[pidx]);
                pidx += 1;
                state.rz(q, params[pidx]);
                pidx += 1;
                state.rx(q, params[pidx]);
                pidx += 1;
            }
            // Entangling layer
            for q in 0..(n - 1) {
                state.cnot(q, q + 1);
            }
        }

        // Compute expectation value of Hamiltonian
        let mut energy = 0.0;
        let mut cidx = 0;

        // Single Z terms
        for q in 0..n {
            energy += h_coeffs[cidx] * state.expect_z(q);
            cidx += 1;
        }

        // ZZ terms (approximate via product of single-qubit expectations)
        for i in 0..n {
            for j in (i + 1)..n {
                if cidx < h_coeffs.len() {
                    energy += h_coeffs[cidx] * state.expect_z(i) * state.expect_z(j);
                    cidx += 1;
                }
            }
        }

        energy
    }
}

/// Encode molecular interaction features as rotation angles.
fn encode_interaction_features(mol_a: &Molecule, mol_b: &Molecule, n: usize) -> Vec<f64> {
    let mut features = Vec::with_capacity(n);

    let com_a = mol_a.center_of_mass();
    let com_b = mol_b.center_of_mass();
    let dist = ((com_a[0] - com_b[0]).powi(2)
        + (com_a[1] - com_b[1]).powi(2)
        + (com_a[2] - com_b[2]).powi(2))
    .sqrt();

    // Feature 0: distance (normalized)
    features.push((dist / 10.0).tanh() * PI);

    // Feature 1: size ratio
    let size_ratio =
        mol_a.atoms.len() as f64 / (mol_a.atoms.len() + mol_b.atoms.len()).max(1) as f64;
    features.push(size_ratio * PI);

    // Feature 2: charge product
    let qa: f64 = mol_a.atoms.iter().map(|a| a.charge).sum();
    let qb: f64 = mol_b.atoms.iter().map(|a| a.charge).sum();
    features.push((qa * qb).tanh() * PI);

    // Fill remaining with derived features
    while features.len() < n {
        let idx = features.len();
        let f = (idx as f64 * 0.37 + dist * 0.13).sin() * PI;
        features.push(f);
    }

    features
}

// ===================================================================
// SCORING FUNCTIONS
// ===================================================================

/// Type of scoring function for docking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScoringFunction {
    /// VQE-based energy evaluation.
    QuantumForceField,
    /// Quantum kernel-based ML score.
    QuantumKernelScore,
    /// Classical force field with quantum correction.
    HybridClassicalQuantum,
}

// ===================================================================
// DOCKING CONFIGURATION AND SCORER
// ===================================================================

/// Configuration for quantum docking.
#[derive(Debug, Clone)]
pub struct DockingConfig {
    pub num_qubits: usize,
    pub scoring_function: ScoringFunction,
    pub num_conformations: usize,
    pub optimization_steps: usize,
}

impl Default for DockingConfig {
    fn default() -> Self {
        DockingConfig {
            num_qubits: 4,
            scoring_function: ScoringFunction::HybridClassicalQuantum,
            num_conformations: 10,
            optimization_steps: 50,
        }
    }
}

/// Result of a docking calculation.
#[derive(Debug, Clone)]
pub struct DockingResult {
    pub score: f64,
    pub binding_energy: f64,
    pub best_conformation: usize,
    pub all_scores: Vec<f64>,
}

/// Quantum-enhanced molecular docking scorer.
pub struct QuantumDockingScorer {
    pub config: DockingConfig,
}

impl QuantumDockingScorer {
    pub fn new(config: DockingConfig) -> Self {
        QuantumDockingScorer { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DockingConfig::default())
    }

    /// Score the docking of a ligand into a protein pocket.
    pub fn score(&self, protein: &Molecule, ligand: &Molecule) -> DrugResult<DockingResult> {
        protein.validate()?;
        ligand.validate()?;

        let mut all_scores = Vec::with_capacity(self.config.num_conformations);

        for conf_idx in 0..self.config.num_conformations {
            let score = match self.config.scoring_function {
                ScoringFunction::QuantumForceField => {
                    let qff = QuantumForceField::new(self.config.num_qubits);
                    qff.interaction_energy(protein, &self.perturb_ligand(ligand, conf_idx))?
                }
                ScoringFunction::QuantumKernelScore => {
                    self.kernel_docking_score(protein, ligand, conf_idx)
                }
                ScoringFunction::HybridClassicalQuantum => {
                    self.hybrid_score(protein, ligand, conf_idx)?
                }
            };
            all_scores.push(score);
        }

        let (best_idx, &best_score) = all_scores
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        Ok(DockingResult {
            score: best_score,
            binding_energy: best_score,
            best_conformation: best_idx,
            all_scores,
        })
    }

    /// Generate a perturbed conformation of the ligand.
    fn perturb_ligand(&self, ligand: &Molecule, seed: usize) -> Molecule {
        let mut perturbed = ligand.clone();
        let offset = (seed as f64) * 0.3;
        for atom in &mut perturbed.atoms {
            // Deterministic pseudo-random perturbation
            let hash = (atom.position[0] * 100.0 + offset).sin() * 0.5;
            atom.position[0] += hash;
            atom.position[1] += (atom.position[1] * 100.0 + offset).cos() * 0.5;
            atom.position[2] += (atom.position[2] * 100.0 + offset).sin() * 0.3;
        }
        perturbed
    }

    /// Kernel-based docking score using quantum fingerprint similarity.
    fn kernel_docking_score(&self, protein: &Molecule, ligand: &Molecule, _conf_idx: usize) -> f64 {
        let fp_prot = MolecularFingerprint::from_molecule(protein, 64);
        let fp_lig = MolecularFingerprint::from_molecule(ligand, 64);
        let kernel = QuantumKernel::new(self.config.num_qubits.min(6));
        let similarity = kernel.compute(&fp_prot, &fp_lig);
        // Higher similarity = better binding = more negative score
        -10.0 * similarity
    }

    /// Hybrid classical + quantum correction score.
    fn hybrid_score(
        &self,
        protein: &Molecule,
        ligand: &Molecule,
        conf_idx: usize,
    ) -> DrugResult<f64> {
        let perturbed = self.perturb_ligand(ligand, conf_idx);
        let qff = QuantumForceField {
            num_qubits: self.config.num_qubits,
            num_layers: 1,
            max_iterations: self.config.optimization_steps,
            convergence_threshold: 1e-3,
        };
        qff.interaction_energy(protein, &perturbed)
    }
}

// ===================================================================
// QUANTUM KERNEL FOR MOLECULAR SIMILARITY
// ===================================================================

/// Quantum kernel for computing molecular similarity.
/// K(x, y) = |<phi(x)|phi(y)>|^2 where phi encodes a fingerprint.
pub struct QuantumKernel {
    pub num_qubits: usize,
}

impl QuantumKernel {
    pub fn new(num_qubits: usize) -> Self {
        QuantumKernel { num_qubits }
    }

    /// Compute the quantum kernel value between two fingerprints.
    /// Returns a value in [0, 1] where 1 means identical.
    pub fn compute(&self, fp1: &MolecularFingerprint, fp2: &MolecularFingerprint) -> f64 {
        let n = self.num_qubits;
        let state1 = self.encode_fingerprint(fp1);
        let state2 = self.encode_fingerprint(fp2);
        let overlap = state1.inner(&state2);
        overlap.norm_sqr()
    }

    /// Encode a fingerprint into a quantum state via angle encoding.
    fn encode_fingerprint(&self, fp: &MolecularFingerprint) -> QState {
        let n = self.num_qubits;
        let mut state = QState::new(n);

        // First layer: Hadamard on all qubits
        for q in 0..n {
            state.h(q);
        }

        // Encode fingerprint bits as rotation angles
        for q in 0..n {
            let bit_idx = q % fp.num_bits;
            let angle = if fp.bits[bit_idx] { PI / 2.0 } else { 0.1 };
            state.rz(q, angle);
        }

        // Entangling layer based on fingerprint
        for q in 0..(n - 1) {
            let bit_idx = (q + n) % fp.num_bits;
            if fp.bits[bit_idx] {
                state.cnot(q, q + 1);
            }
        }

        // Second feature map layer
        for q in 0..n {
            let bits_sum: usize = (0..fp.num_bits.min(8))
                .filter(|&i| fp.bits[(q * 7 + i) % fp.num_bits])
                .count();
            let angle = (bits_sum as f64 / 8.0) * PI;
            state.ry(q, angle);
        }

        state
    }

    /// Compute the full kernel matrix for a set of fingerprints.
    pub fn kernel_matrix(&self, fingerprints: &[MolecularFingerprint]) -> Vec<Vec<f64>> {
        let n = fingerprints.len();
        let mut matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let k = self.compute(&fingerprints[i], &fingerprints[j]);
                matrix[i][j] = k;
                matrix[j][i] = k;
            }
        }
        matrix
    }
}

// ===================================================================
// BINDING AFFINITY ESTIMATOR
// ===================================================================

/// Estimates binding affinity using quantum feature encoding and a QNN.
pub struct BindingAffinityEstimator {
    pub protein_features: Vec<f64>,
    pub ligand_features: Vec<f64>,
    pub num_qubits: usize,
}

impl BindingAffinityEstimator {
    pub fn new(num_qubits: usize) -> Self {
        BindingAffinityEstimator {
            protein_features: Vec::new(),
            ligand_features: Vec::new(),
            num_qubits,
        }
    }

    /// Set features from molecules.
    pub fn set_from_molecules(&mut self, protein: &Molecule, ligand: &Molecule) {
        self.protein_features = extract_molecular_features(protein);
        self.ligand_features = extract_molecular_features(ligand);
    }

    /// Estimate binding affinity (pKd value).
    /// Higher values indicate stronger binding.
    pub fn estimate(&self) -> DrugResult<f64> {
        if self.protein_features.is_empty() || self.ligand_features.is_empty() {
            return Err(DrugDesignError::FeatureError(
                "Features not set; call set_from_molecules first".to_string(),
            ));
        }

        let n = self.num_qubits;
        let mut state = QState::new(n);

        // Encode protein features
        let half = n / 2;
        for q in 0..half {
            let feat = self.protein_features[q % self.protein_features.len()];
            state.ry(q, feat.tanh() * PI);
        }

        // Encode ligand features
        for q in half..n {
            let feat = self.ligand_features[(q - half) % self.ligand_features.len()];
            state.ry(q, feat.tanh() * PI);
        }

        // Entangle protein-ligand qubits
        for q in 0..half.min(n - half) {
            state.cnot(q, half + q);
        }

        // Variational layer with fixed trained parameters (pre-trained proxy)
        for q in 0..n {
            let param = (q as f64 * 0.7 + 0.3).sin() * PI / 3.0;
            state.rz(q, param);
        }

        // Read out binding affinity from expectation values
        let mut affinity = 0.0;
        for q in 0..n {
            affinity += state.expect_z(q) * (1.0 / n as f64);
        }

        // Map to pKd range [2, 12]
        let pkd = 7.0 + affinity * 5.0;
        Ok(pkd.clamp(2.0, 12.0))
    }
}

/// Extract numerical features from a molecule.
fn extract_molecular_features(mol: &Molecule) -> Vec<f64> {
    let mut features = Vec::new();
    features.push(mol.molecular_weight() / 500.0);
    features.push(mol.estimated_log_p() / 5.0);
    features.push(mol.h_bond_donors() as f64 / 5.0);
    features.push(mol.h_bond_acceptors() as f64 / 10.0);
    features.push(mol.atoms.len() as f64 / 50.0);
    features.push(mol.polar_surface_area() / 140.0);

    let total_charge: f64 = mol.atoms.iter().map(|a| a.charge).sum();
    features.push(total_charge.tanh());

    let aromatic_frac = mol
        .bonds
        .iter()
        .filter(|b| b.bond_type == BondType::Aromatic)
        .count() as f64
        / mol.bonds.len().max(1) as f64;
    features.push(aromatic_frac);

    features
}

// ===================================================================
// GENERATIVE MOLECULAR DESIGN (BORN MACHINE)
// ===================================================================

/// Target property for generative design.
#[derive(Debug, Clone)]
pub struct PropertyTarget {
    pub name: String,
    pub target_value: f64,
    pub weight: f64,
}

/// Configuration for the quantum molecular generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub num_qubits: usize,
    pub latent_dim: usize,
    pub num_layers: usize,
    pub property_targets: Vec<PropertyTarget>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        GeneratorConfig {
            num_qubits: 6,
            latent_dim: 4,
            num_layers: 2,
            property_targets: Vec::new(),
        }
    }
}

/// Quantum molecular generator using Born machine sampling.
pub struct QuantumMolecularGenerator {
    pub config: GeneratorConfig,
    params: Vec<f64>,
}

/// A generated molecule candidate.
#[derive(Debug, Clone)]
pub struct GeneratedCandidate {
    pub fingerprint: MolecularFingerprint,
    pub score: f64,
    pub bitstring: Vec<bool>,
}

impl QuantumMolecularGenerator {
    pub fn new(config: GeneratorConfig) -> Self {
        let num_params =
            config.num_qubits * config.num_layers * 3 + (config.num_qubits - 1) * config.num_layers;
        let params: Vec<f64> = (0..num_params)
            .map(|i| ((i as f64) * 0.618 % 1.0) * 2.0 * PI) // Golden ratio initialization
            .collect();
        QuantumMolecularGenerator { config, params }
    }

    /// Sample a batch of molecular fingerprints from the Born machine.
    pub fn sample(&self, num_samples: usize) -> Vec<GeneratedCandidate> {
        let n = self.config.num_qubits;
        let state = self.build_circuit();
        let probs = state.probabilities();

        // Convert probabilities to candidates
        let mut candidates: Vec<(usize, f64)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-probability bitstrings as samples
        let take = num_samples.min(candidates.len());
        candidates[..take]
            .iter()
            .map(|&(idx, prob)| {
                let bits: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
                let fp = MolecularFingerprint {
                    bits: bits.clone(),
                    num_bits: n,
                };
                GeneratedCandidate {
                    fingerprint: fp,
                    score: prob,
                    bitstring: bits,
                }
            })
            .collect()
    }

    /// Build the parameterized quantum circuit.
    fn build_circuit(&self) -> QState {
        let n = self.config.num_qubits;
        let mut state = QState::new(n);
        let mut pidx = 0;

        for _layer in 0..self.config.num_layers {
            // Single-qubit rotations
            for q in 0..n {
                state.ry(q, self.params[pidx]);
                pidx += 1;
                state.rz(q, self.params[pidx]);
                pidx += 1;
                state.rx(q, self.params[pidx]);
                pidx += 1;
            }
            // Entangling layer
            for q in 0..(n - 1) {
                state.cnot(q, q + 1);
                if pidx < self.params.len() {
                    // Parameterized ZZ interaction
                    state.rz(q + 1, self.params[pidx] * 0.1);
                    pidx += 1;
                }
            }
        }

        state
    }

    /// Train the generator to match target property distribution.
    /// Uses MMD (Maximum Mean Discrepancy) loss with parameter-shift gradients.
    /// Returns the loss after training.
    pub fn train(
        &mut self,
        target_fingerprints: &[MolecularFingerprint],
        num_iterations: usize,
    ) -> f64 {
        let step_size = 0.02;
        let mut best_loss = f64::MAX;

        for _iter in 0..num_iterations {
            let loss = self.mmd_loss(target_fingerprints);
            if loss < best_loss {
                best_loss = loss;
            }

            // Gradient via parameter shift
            let shift = PI / 2.0;
            let num_params = self.params.len();
            let mut gradients = vec![0.0; num_params];

            for p in 0..num_params {
                let orig = self.params[p];
                self.params[p] = orig + shift;
                let loss_plus = self.mmd_loss(target_fingerprints);
                self.params[p] = orig - shift;
                let loss_minus = self.mmd_loss(target_fingerprints);
                self.params[p] = orig;
                gradients[p] = (loss_plus - loss_minus) / 2.0;
            }

            for p in 0..num_params {
                self.params[p] -= step_size * gradients[p];
            }
        }

        best_loss
    }

    /// MMD loss between generated distribution and target fingerprints.
    fn mmd_loss(&self, targets: &[MolecularFingerprint]) -> f64 {
        let samples = self.sample(targets.len().max(4));
        if samples.is_empty() || targets.is_empty() {
            return 1.0;
        }

        // K(generated, generated)
        let mut kgg = 0.0;
        let ng = samples.len() as f64;
        for i in 0..samples.len() {
            for j in 0..samples.len() {
                kgg += fp_rbf_kernel(&samples[i].fingerprint, &samples[j].fingerprint);
            }
        }
        kgg /= ng * ng;

        // K(target, target)
        let mut ktt = 0.0;
        let nt = targets.len() as f64;
        for i in 0..targets.len() {
            for j in 0..targets.len() {
                ktt += fp_rbf_kernel(&targets[i], &targets[j]);
            }
        }
        ktt /= nt * nt;

        // K(generated, target)
        let mut kgt = 0.0;
        for s in &samples {
            for t in targets {
                kgt += fp_rbf_kernel(&s.fingerprint, t);
            }
        }
        kgt /= ng * nt;

        (kgg + ktt - 2.0 * kgt).max(0.0)
    }
}

/// RBF kernel on fingerprints using Tanimoto distance.
fn fp_rbf_kernel(a: &MolecularFingerprint, b: &MolecularFingerprint) -> f64 {
    let tanimoto = a.tanimoto(b);
    let dist_sq = (1.0 - tanimoto).powi(2);
    (-dist_sq / 0.5).exp()
}

// ===================================================================
// ADMET PROPERTY PREDICTION
// ===================================================================

/// ADMET property type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdmetProperty {
    Absorption,
    Distribution,
    Metabolism,
    Excretion,
    Toxicity,
    Solubility,
    LogP,
    BBBPermeability,
}

/// Result of an ADMET prediction.
#[derive(Debug, Clone)]
pub struct AdmetResult {
    pub property: AdmetProperty,
    pub probability: f64,
    pub passes: bool,
    pub confidence: f64,
}

/// Quantum ADMET property predictor using QNN classifiers.
pub struct AdmetPredictor {
    pub num_qubits: usize,
    pub properties: Vec<AdmetProperty>,
}

impl AdmetPredictor {
    pub fn new(num_qubits: usize, properties: Vec<AdmetProperty>) -> Self {
        AdmetPredictor {
            num_qubits,
            properties,
        }
    }

    /// Predict all configured ADMET properties for a molecule.
    pub fn predict(&self, mol: &Molecule) -> DrugResult<Vec<AdmetResult>> {
        mol.validate()?;
        let features = extract_molecular_features(mol);
        let mut results = Vec::with_capacity(self.properties.len());

        for &prop in &self.properties {
            let result = self.predict_single(&features, prop);
            results.push(result);
        }

        Ok(results)
    }

    /// Predict a single ADMET property.
    fn predict_single(&self, features: &[f64], property: AdmetProperty) -> AdmetResult {
        let n = self.num_qubits;
        let mut state = QState::new(n);

        // Encode features
        for q in 0..n {
            let feat = features[q % features.len()];
            state.ry(q, feat.tanh() * PI);
        }

        // Property-specific circuit
        let property_seed = match property {
            AdmetProperty::Absorption => 1.0,
            AdmetProperty::Distribution => 2.0,
            AdmetProperty::Metabolism => 3.0,
            AdmetProperty::Excretion => 4.0,
            AdmetProperty::Toxicity => 5.0,
            AdmetProperty::Solubility => 6.0,
            AdmetProperty::LogP => 7.0,
            AdmetProperty::BBBPermeability => 8.0,
        };

        // Variational classifier (pre-trained weights proxy)
        for q in 0..n {
            let param = (property_seed * (q as f64 + 1.0) * 0.37).sin() * PI / 2.0;
            state.rz(q, param);
        }
        for q in 0..(n - 1) {
            state.cnot(q, q + 1);
        }
        for q in 0..n {
            let param = (property_seed * (q as f64 + 1.0) * 0.73).cos() * PI / 3.0;
            state.ry(q, param);
        }

        // Read out classification from qubit 0
        let exp_z = state.expect_z(0);
        let probability = (1.0 - exp_z) / 2.0; // Map [-1,1] to [0,1]
        let confidence = exp_z.abs();

        // Threshold depends on property (toxicity uses lower threshold)
        let threshold = match property {
            AdmetProperty::Toxicity => 0.6,
            _ => 0.5,
        };

        AdmetResult {
            property,
            probability,
            passes: probability < threshold,
            confidence,
        }
    }
}

// ===================================================================
// LEAD OPTIMIZATION
// ===================================================================

/// Result of lead optimization.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub best_score: f64,
    pub scores_over_iterations: Vec<f64>,
    pub pareto_front: Vec<ParetoPoint>,
}

/// A point on the Pareto front.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    pub objectives: Vec<f64>,
    pub candidate_idx: usize,
}

impl ParetoPoint {
    /// Check if this point dominates another (all objectives better or equal, at least one strictly better).
    /// Assumes minimization for all objectives.
    pub fn dominates(&self, other: &ParetoPoint) -> bool {
        let all_leq = self
            .objectives
            .iter()
            .zip(&other.objectives)
            .all(|(a, b)| a <= b);
        let any_lt = self
            .objectives
            .iter()
            .zip(&other.objectives)
            .any(|(a, b)| a < b);
        all_leq && any_lt
    }
}

/// Perform lead optimization: start from a lead, generate variants, score them.
pub fn lead_optimization(
    lead: &Molecule,
    protein: &Molecule,
    num_iterations: usize,
    num_qubits: usize,
) -> DrugResult<OptimizationResult> {
    lead.validate()?;
    protein.validate()?;

    let scorer = QuantumDockingScorer::new(DockingConfig {
        num_qubits,
        scoring_function: ScoringFunction::HybridClassicalQuantum,
        num_conformations: 3,
        optimization_steps: 20,
    });

    let predictor = AdmetPredictor::new(
        num_qubits,
        vec![
            AdmetProperty::Toxicity,
            AdmetProperty::Solubility,
            AdmetProperty::Absorption,
        ],
    );

    let mut scores = Vec::with_capacity(num_iterations);
    let mut pareto_points = Vec::new();
    let mut best_score = f64::MAX;

    for iter in 0..num_iterations {
        // Generate variant (simple perturbation of lead)
        let mut variant = lead.clone();
        for atom in &mut variant.atoms {
            let perturbation = ((iter as f64 * 0.618 + atom.position[0]).sin()) * 0.2;
            atom.position[0] += perturbation;
            atom.charge += (iter as f64 * 0.1).cos() * 0.01;
        }

        // Score binding
        let dock_result = scorer.score(protein, &variant)?;
        let binding_score = dock_result.score;

        // Score ADMET
        let admet_results = predictor.predict(&variant)?;
        let admet_penalty: f64 = admet_results.iter().filter(|r| !r.passes).count() as f64 * 2.0;

        // Drug-likeness
        let drug_likeness = evaluate_drug_likeness(&variant);
        let likeness_penalty = drug_likeness.lipinski_violations as f64 * 1.5;

        let total_score = binding_score + admet_penalty + likeness_penalty;
        if total_score < best_score {
            best_score = total_score;
        }
        scores.push(best_score);

        // Multi-objective: [binding, admet_penalty, likeness_penalty]
        pareto_points.push(ParetoPoint {
            objectives: vec![binding_score, admet_penalty, likeness_penalty],
            candidate_idx: iter,
        });
    }

    // Filter to non-dominated Pareto front
    let front = compute_pareto_front(&pareto_points);

    Ok(OptimizationResult {
        best_score,
        scores_over_iterations: scores,
        pareto_front: front,
    })
}

/// Compute the Pareto front from a set of points.
fn compute_pareto_front(points: &[ParetoPoint]) -> Vec<ParetoPoint> {
    let mut front = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let dominated = points
            .iter()
            .enumerate()
            .any(|(j, q)| j != i && q.dominates(p));
        if !dominated {
            front.push(p.clone());
        }
    }
    front
}

// ===================================================================
// DRUG DISCOVERY PIPELINE
// ===================================================================

/// Stage in the drug discovery pipeline.
#[derive(Debug, Clone)]
pub enum PipelineStage {
    VirtualScreening { library_size: usize },
    LeadOptimization { num_iterations: usize },
    BindingAffinity,
    AdmetFiltering,
    ToxicityPrediction,
}

/// Result of a pipeline stage.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage_name: String,
    pub passed: usize,
    pub failed: usize,
    pub scores: Vec<f64>,
}

/// Full drug discovery pipeline.
pub struct DrugDiscoveryPipeline {
    pub stages: Vec<PipelineStage>,
    pub num_qubits: usize,
}

impl DrugDiscoveryPipeline {
    pub fn new(num_qubits: usize) -> Self {
        DrugDiscoveryPipeline {
            stages: Vec::new(),
            num_qubits,
        }
    }

    /// Create a standard pipeline with all stages.
    pub fn standard() -> Self {
        DrugDiscoveryPipeline {
            stages: vec![
                PipelineStage::VirtualScreening { library_size: 100 },
                PipelineStage::BindingAffinity,
                PipelineStage::AdmetFiltering,
                PipelineStage::ToxicityPrediction,
                PipelineStage::LeadOptimization { num_iterations: 10 },
            ],
            num_qubits: 4,
        }
    }

    /// Run the full pipeline on a set of candidate molecules against a target protein.
    pub fn run(&self, candidates: &[Molecule], protein: &Molecule) -> DrugResult<Vec<StageResult>> {
        let mut results = Vec::new();
        let mut active_indices: Vec<usize> = (0..candidates.len()).collect();

        for stage in &self.stages {
            let stage_result = match stage {
                PipelineStage::VirtualScreening { library_size: _ } => {
                    self.virtual_screening(&active_indices, candidates, protein)?
                }
                PipelineStage::BindingAffinity => {
                    self.binding_affinity_stage(&active_indices, candidates, protein)?
                }
                PipelineStage::AdmetFiltering => {
                    self.admet_filtering(&active_indices, candidates)?
                }
                PipelineStage::ToxicityPrediction => {
                    self.toxicity_stage(&active_indices, candidates)?
                }
                PipelineStage::LeadOptimization { num_iterations } => {
                    self.lead_opt_stage(&active_indices, candidates, protein, *num_iterations)?
                }
            };

            // Update active indices based on scores (keep top 50% or at least 1)
            let keep = (stage_result.passed).max(1);
            let mut scored: Vec<(usize, f64)> = active_indices
                .iter()
                .zip(stage_result.scores.iter())
                .map(|(&idx, &s)| (idx, s))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            active_indices = scored.iter().take(keep).map(|&(idx, _)| idx).collect();

            results.push(stage_result);
        }

        Ok(results)
    }

    fn virtual_screening(
        &self,
        indices: &[usize],
        candidates: &[Molecule],
        protein: &Molecule,
    ) -> DrugResult<StageResult> {
        let kernel = QuantumKernel::new(self.num_qubits.min(6));
        let protein_fp = MolecularFingerprint::from_molecule(protein, 64);
        let mut scores = Vec::new();

        for &idx in indices {
            let fp = MolecularFingerprint::from_molecule(&candidates[idx], 64);
            let sim = kernel.compute(&protein_fp, &fp);
            scores.push(-sim); // Negative because higher similarity = better
        }

        let passed = (indices.len() + 1) / 2;
        Ok(StageResult {
            stage_name: "Virtual Screening".to_string(),
            passed,
            failed: indices.len() - passed,
            scores,
        })
    }

    fn binding_affinity_stage(
        &self,
        indices: &[usize],
        candidates: &[Molecule],
        protein: &Molecule,
    ) -> DrugResult<StageResult> {
        let mut estimator = BindingAffinityEstimator::new(self.num_qubits);
        let mut scores = Vec::new();

        for &idx in indices {
            estimator.set_from_molecules(protein, &candidates[idx]);
            let pkd = estimator.estimate()?;
            scores.push(-pkd); // Negative because higher pKd = better
        }

        let passed = scores.iter().filter(|&&s| s < -5.0).count().max(1);
        Ok(StageResult {
            stage_name: "Binding Affinity".to_string(),
            passed,
            failed: indices.len() - passed,
            scores,
        })
    }

    fn admet_filtering(
        &self,
        indices: &[usize],
        candidates: &[Molecule],
    ) -> DrugResult<StageResult> {
        let predictor = AdmetPredictor::new(
            self.num_qubits,
            vec![
                AdmetProperty::Absorption,
                AdmetProperty::Distribution,
                AdmetProperty::Metabolism,
                AdmetProperty::Excretion,
                AdmetProperty::Solubility,
            ],
        );

        let mut scores = Vec::new();
        let mut passed = 0;

        for &idx in indices {
            let results = predictor.predict(&candidates[idx])?;
            let num_pass = results.iter().filter(|r| r.passes).count();
            let score = -(num_pass as f64); // More passes = better
            scores.push(score);
            if num_pass >= 3 {
                passed += 1;
            }
        }
        let passed = passed.max(1);

        Ok(StageResult {
            stage_name: "ADMET Filtering".to_string(),
            passed,
            failed: indices.len() - passed,
            scores,
        })
    }

    fn toxicity_stage(
        &self,
        indices: &[usize],
        candidates: &[Molecule],
    ) -> DrugResult<StageResult> {
        let predictor = AdmetPredictor::new(self.num_qubits, vec![AdmetProperty::Toxicity]);

        let mut scores = Vec::new();
        let mut passed = 0;

        for &idx in indices {
            let results = predictor.predict(&candidates[idx])?;
            let is_safe = results.iter().all(|r| r.passes);
            scores.push(if is_safe { 0.0 } else { 1.0 });
            if is_safe {
                passed += 1;
            }
        }
        let passed = passed.max(1);

        Ok(StageResult {
            stage_name: "Toxicity Prediction".to_string(),
            passed,
            failed: indices.len() - passed,
            scores,
        })
    }

    fn lead_opt_stage(
        &self,
        indices: &[usize],
        candidates: &[Molecule],
        protein: &Molecule,
        num_iterations: usize,
    ) -> DrugResult<StageResult> {
        let mut scores = Vec::new();
        let mut passed = 0;

        for &idx in indices {
            let result =
                lead_optimization(&candidates[idx], protein, num_iterations, self.num_qubits)?;
            scores.push(result.best_score);
            passed += 1; // All that reach this stage pass
        }

        Ok(StageResult {
            stage_name: "Lead Optimization".to_string(),
            passed,
            failed: 0,
            scores,
        })
    }
}

// ===================================================================
// PRE-BUILT MOLECULAR LIBRARY
// ===================================================================

/// Library of pre-built molecules for testing and benchmarking.
pub struct MolecularLibrary;

impl MolecularLibrary {
    /// Aspirin (acetylsalicylic acid) C9H8O4 -- simplified 3D.
    pub fn aspirin() -> Molecule {
        let mut mol = Molecule::new("Aspirin");
        mol.smiles = Some("CC(=O)OC1=CC=CC=C1C(=O)O".to_string());

        // Aromatic ring carbons
        let c0 = mol.add_atom(Element::C, [0.0, 0.0, 0.0], -0.08);
        let c1 = mol.add_atom(Element::C, [1.4, 0.0, 0.0], -0.08);
        let c2 = mol.add_atom(Element::C, [2.1, 1.2, 0.0], -0.12);
        let c3 = mol.add_atom(Element::C, [1.4, 2.4, 0.0], -0.12);
        let c4 = mol.add_atom(Element::C, [0.0, 2.4, 0.0], -0.12);
        let c5 = mol.add_atom(Element::C, [-0.7, 1.2, 0.0], -0.08);

        // Ester group: -OC(=O)CH3
        let o6 = mol.add_atom(Element::O, [-1.4, 0.0, 0.0], -0.33);
        let c7 = mol.add_atom(Element::C, [-2.8, 0.0, 0.0], 0.51);
        let o8 = mol.add_atom(Element::O, [-3.5, 1.0, 0.0], -0.43);
        let c9 = mol.add_atom(Element::C, [-3.5, -1.2, 0.0], -0.18);

        // Carboxylic acid: -C(=O)OH
        let c10 = mol.add_atom(Element::C, [2.1, -1.2, 0.0], 0.52);
        let o11 = mol.add_atom(Element::O, [3.3, -1.2, 0.0], -0.44);
        let o12 = mol.add_atom(Element::O, [1.4, -2.4, 0.0], -0.36);

        // Hydrogen atoms (representative)
        let h0 = mol.add_atom(Element::H, [2.8, 1.2, 0.0], 0.13);
        let h1 = mol.add_atom(Element::H, [2.1, 3.3, 0.0], 0.13);
        let h2 = mol.add_atom(Element::H, [-0.7, 3.3, 0.0], 0.13);
        let h3 = mol.add_atom(Element::H, [-1.4, 1.2, 0.0], 0.13);
        let h4 = mol.add_atom(Element::H, [-3.2, -2.0, 0.0], 0.06);
        let h5 = mol.add_atom(Element::H, [-4.4, -1.0, 0.0], 0.06);
        let h6 = mol.add_atom(Element::H, [-3.8, -1.6, 0.5], 0.06);
        let h7 = mol.add_atom(Element::H, [1.0, -3.0, 0.0], 0.33);

        // Aromatic bonds
        mol.add_bond(c0, c1, BondType::Aromatic);
        mol.add_bond(c1, c2, BondType::Aromatic);
        mol.add_bond(c2, c3, BondType::Aromatic);
        mol.add_bond(c3, c4, BondType::Aromatic);
        mol.add_bond(c4, c5, BondType::Aromatic);
        mol.add_bond(c5, c0, BondType::Aromatic);

        // Ester
        mol.add_bond(c0, o6, BondType::Single);
        mol.add_bond(o6, c7, BondType::Single);
        mol.add_bond(c7, o8, BondType::Double);
        mol.add_bond(c7, c9, BondType::Single);

        // Carboxylic acid
        mol.add_bond(c1, c10, BondType::Single);
        mol.add_bond(c10, o11, BondType::Double);
        mol.add_bond(c10, o12, BondType::Single);

        // C-H bonds
        mol.add_bond(c2, h0, BondType::Single);
        mol.add_bond(c3, h1, BondType::Single);
        mol.add_bond(c4, h2, BondType::Single);
        mol.add_bond(c5, h3, BondType::Single);
        mol.add_bond(c9, h4, BondType::Single);
        mol.add_bond(c9, h5, BondType::Single);
        mol.add_bond(c9, h6, BondType::Single);
        mol.add_bond(o12, h7, BondType::Single);

        mol
    }

    /// Ibuprofen C13H18O2 -- simplified.
    pub fn ibuprofen() -> Molecule {
        let mut mol = Molecule::new("Ibuprofen");
        mol.smiles = Some("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O".to_string());

        // Core aromatic ring
        let c0 = mol.add_atom(Element::C, [0.0, 0.0, 0.0], -0.08);
        let c1 = mol.add_atom(Element::C, [1.4, 0.0, 0.0], -0.08);
        let c2 = mol.add_atom(Element::C, [2.1, 1.2, 0.0], -0.12);
        let c3 = mol.add_atom(Element::C, [1.4, 2.4, 0.0], -0.08);
        let c4 = mol.add_atom(Element::C, [0.0, 2.4, 0.0], -0.12);
        let c5 = mol.add_atom(Element::C, [-0.7, 1.2, 0.0], -0.08);

        // Isobutyl group
        let c6 = mol.add_atom(Element::C, [-1.4, -0.6, 0.0], -0.04);
        let c7 = mol.add_atom(Element::C, [-2.8, -0.2, 0.0], -0.02);
        let c8 = mol.add_atom(Element::C, [-3.5, -1.2, 0.0], -0.18);
        let c9 = mol.add_atom(Element::C, [-3.5, 1.0, 0.0], -0.18);

        // Propionic acid side
        let c10 = mol.add_atom(Element::C, [2.1, 3.6, 0.0], 0.10);
        let c11 = mol.add_atom(Element::C, [3.5, 3.6, 0.0], -0.18);
        let c12 = mol.add_atom(Element::C, [1.4, 4.8, 0.0], 0.52);
        let o0 = mol.add_atom(Element::O, [2.1, 6.0, 0.0], -0.44);
        let o1 = mol.add_atom(Element::O, [0.0, 4.8, 0.0], -0.36);

        // Representative hydrogens
        for _ in 0..18 {
            mol.add_atom(Element::H, [0.0, 0.0, 0.5], 0.06);
        }

        // Aromatic bonds
        mol.add_bond(c0, c1, BondType::Aromatic);
        mol.add_bond(c1, c2, BondType::Aromatic);
        mol.add_bond(c2, c3, BondType::Aromatic);
        mol.add_bond(c3, c4, BondType::Aromatic);
        mol.add_bond(c4, c5, BondType::Aromatic);
        mol.add_bond(c5, c0, BondType::Aromatic);

        mol.add_bond(c0, c6, BondType::Single);
        mol.add_bond(c6, c7, BondType::Single);
        mol.add_bond(c7, c8, BondType::Single);
        mol.add_bond(c7, c9, BondType::Single);
        mol.add_bond(c3, c10, BondType::Single);
        mol.add_bond(c10, c11, BondType::Single);
        mol.add_bond(c10, c12, BondType::Single);
        mol.add_bond(c12, o0, BondType::Double);
        mol.add_bond(c12, o1, BondType::Single);

        mol
    }

    /// Caffeine C8H10N4O2 -- simplified.
    pub fn caffeine() -> Molecule {
        let mut mol = Molecule::new("Caffeine");
        mol.smiles = Some("CN1C=NC2=C1C(=O)N(C(=O)N2C)C".to_string());

        // Purine ring system
        let c2 = mol.add_atom(Element::C, [0.0, 0.0, 0.0], 0.34);
        let n3 = mol.add_atom(Element::N, [1.2, 0.6, 0.0], -0.36);
        let c4 = mol.add_atom(Element::C, [1.2, 2.0, 0.0], 0.08);
        let c5 = mol.add_atom(Element::C, [0.0, 2.6, 0.0], -0.02);
        let c6 = mol.add_atom(Element::C, [-1.2, 2.0, 0.0], 0.46);
        let n1 = mol.add_atom(Element::N, [-1.2, 0.6, 0.0], -0.32);

        let n7 = mol.add_atom(Element::N, [0.6, 3.9, 0.0], -0.38);
        let c8 = mol.add_atom(Element::C, [2.0, 3.6, 0.0], 0.20);
        let n9 = mol.add_atom(Element::N, [2.2, 2.2, 0.0], -0.30);

        // Carbonyl oxygens
        let o_c2 = mol.add_atom(Element::O, [0.0, -1.4, 0.0], -0.42);
        let o_c6 = mol.add_atom(Element::O, [-2.4, 2.6, 0.0], -0.42);

        // Methyl groups (N1, N3, N7)
        let ch3_n1 = mol.add_atom(Element::C, [-2.4, 0.0, 0.0], -0.14);
        let ch3_n3 = mol.add_atom(Element::C, [2.4, 0.0, 0.0], -0.14);
        let ch3_n7 = mol.add_atom(Element::C, [0.0, 5.2, 0.0], -0.14);

        // Hydrogens
        for _ in 0..10 {
            mol.add_atom(Element::H, [0.0, 0.0, 0.5], 0.06);
        }

        // Bonds
        mol.add_bond(c2, n3, BondType::Single);
        mol.add_bond(n3, c4, BondType::Single);
        mol.add_bond(c4, c5, BondType::Double);
        mol.add_bond(c5, c6, BondType::Single);
        mol.add_bond(c6, n1, BondType::Single);
        mol.add_bond(n1, c2, BondType::Single);
        mol.add_bond(c5, n7, BondType::Single);
        mol.add_bond(n7, c8, BondType::Single);
        mol.add_bond(c8, n9, BondType::Double);
        mol.add_bond(n9, c4, BondType::Single);
        mol.add_bond(c2, o_c2, BondType::Double);
        mol.add_bond(c6, o_c6, BondType::Double);
        mol.add_bond(n1, ch3_n1, BondType::Single);
        mol.add_bond(n3, ch3_n3, BondType::Single);
        mol.add_bond(n7, ch3_n7, BondType::Single);

        mol
    }

    /// Simplified KRAS G12C inhibitor (sotorasib-like) from Nature Biotech 2025.
    pub fn kras_inhibitor() -> Molecule {
        let mut mol = Molecule::new("KRAS-G12C-Inhibitor");
        mol.smiles = Some("CC1=C(C=NN1)C2=CC=C(C=C2)N3CCCC3=O".to_string());

        // Build a simplified representation
        // Pyrazole ring
        let n0 = mol.add_atom(Element::N, [0.0, 0.0, 0.0], -0.22);
        let n1 = mol.add_atom(Element::N, [1.0, 0.6, 0.0], -0.18);
        let c0 = mol.add_atom(Element::C, [0.6, 1.8, 0.0], 0.02);
        let c1 = mol.add_atom(Element::C, [-0.6, 1.8, 0.0], -0.10);
        let c2 = mol.add_atom(Element::C, [-0.8, 0.4, 0.0], 0.08);

        // Phenyl ring
        let c3 = mol.add_atom(Element::C, [1.4, 3.0, 0.0], -0.08);
        let c4 = mol.add_atom(Element::C, [2.8, 3.0, 0.0], -0.12);
        let c5 = mol.add_atom(Element::C, [3.5, 4.2, 0.0], -0.08);
        let c6 = mol.add_atom(Element::C, [2.8, 5.4, 0.0], -0.08);
        let c7 = mol.add_atom(Element::C, [1.4, 5.4, 0.0], -0.12);
        let c8 = mol.add_atom(Element::C, [0.7, 4.2, 0.0], -0.08);

        // Pyrrolidinone
        let n2 = mol.add_atom(Element::N, [3.5, 6.6, 0.0], -0.28);
        let c9 = mol.add_atom(Element::C, [5.0, 6.6, 0.0], -0.04);
        let c10 = mol.add_atom(Element::C, [5.4, 8.0, 0.0], -0.04);
        let c11 = mol.add_atom(Element::C, [4.0, 8.6, 0.0], -0.04);
        let c12 = mol.add_atom(Element::C, [3.0, 7.8, 0.0], 0.44);
        let o0 = mol.add_atom(Element::O, [1.8, 8.2, 0.0], -0.42);

        // Methyl on pyrazole
        let c13 = mol.add_atom(Element::C, [-2.2, 0.0, 0.0], -0.18);

        // Fluorine (covalent warhead proxy)
        let f0 = mol.add_atom(Element::F, [5.8, 6.0, 0.0], -0.20);

        // Representative hydrogens
        for _ in 0..16 {
            mol.add_atom(Element::H, [0.0, 0.0, 0.5], 0.06);
        }

        // Pyrazole bonds
        mol.add_bond(n0, n1, BondType::Single);
        mol.add_bond(n1, c0, BondType::Double);
        mol.add_bond(c0, c1, BondType::Single);
        mol.add_bond(c1, c2, BondType::Double);
        mol.add_bond(c2, n0, BondType::Single);

        // Pyrazole-phenyl link
        mol.add_bond(c0, c3, BondType::Single);

        // Phenyl
        mol.add_bond(c3, c4, BondType::Aromatic);
        mol.add_bond(c4, c5, BondType::Aromatic);
        mol.add_bond(c5, c6, BondType::Aromatic);
        mol.add_bond(c6, c7, BondType::Aromatic);
        mol.add_bond(c7, c8, BondType::Aromatic);
        mol.add_bond(c8, c3, BondType::Aromatic);

        // Phenyl-pyrrolidinone link
        mol.add_bond(c6, n2, BondType::Single);

        // Pyrrolidinone
        mol.add_bond(n2, c9, BondType::Single);
        mol.add_bond(c9, c10, BondType::Single);
        mol.add_bond(c10, c11, BondType::Single);
        mol.add_bond(c11, c12, BondType::Single);
        mol.add_bond(c12, n2, BondType::Single);
        mol.add_bond(c12, o0, BondType::Double);

        // Methyl
        mol.add_bond(c2, c13, BondType::Single);

        // Fluorine
        mol.add_bond(c9, f0, BondType::Single);

        mol
    }

    /// Simplified protein binding pocket (10 residue-representative atoms).
    pub fn simple_protein_pocket() -> Molecule {
        let mut mol = Molecule::new("Protein-Pocket");

        // 10 representative heavy atoms from binding site residues
        // Arranged in a pocket-like concavity
        mol.add_atom(Element::N, [0.0, 0.0, 0.0], -0.30);
        mol.add_atom(Element::C, [2.0, 0.5, 0.0], 0.10);
        mol.add_atom(Element::O, [4.0, 0.0, 0.0], -0.40);
        mol.add_atom(Element::C, [1.0, 2.5, 0.0], -0.05);
        mol.add_atom(Element::N, [3.0, 2.5, 0.0], -0.25);
        mol.add_atom(Element::O, [0.0, 4.5, 0.0], -0.35);
        mol.add_atom(Element::C, [2.0, 4.5, 0.0], 0.05);
        mol.add_atom(Element::S, [4.0, 4.0, 0.0], -0.10);
        mol.add_atom(Element::N, [1.0, 6.0, 0.0], -0.20);
        mol.add_atom(Element::C, [3.0, 6.0, 0.0], 0.08);

        // Backbone-like bonds
        mol.add_bond(0, 1, BondType::Single);
        mol.add_bond(1, 2, BondType::Double);
        mol.add_bond(1, 3, BondType::Single);
        mol.add_bond(3, 4, BondType::Single);
        mol.add_bond(3, 5, BondType::Single);
        mol.add_bond(4, 6, BondType::Single);
        mol.add_bond(6, 7, BondType::Single);
        mol.add_bond(6, 8, BondType::Single);
        mol.add_bond(8, 9, BondType::Single);

        mol
    }

    /// Water molecule H2O.
    pub fn water() -> Molecule {
        let mut mol = Molecule::new("Water");
        let o = mol.add_atom(Element::O, [0.0, 0.0, 0.0], -0.82);
        let h1 = mol.add_atom(Element::H, [0.96, 0.0, 0.0], 0.41);
        let h2 = mol.add_atom(Element::H, [-0.24, 0.93, 0.0], 0.41);
        mol.add_bond(o, h1, BondType::Single);
        mol.add_bond(o, h2, BondType::Single);
        mol
    }

    /// H2 molecule for simple energy tests.
    pub fn hydrogen_molecule() -> Molecule {
        let mut mol = Molecule::new("H2");
        let h1 = mol.add_atom(Element::H, [0.0, 0.0, 0.0], 0.0);
        let h2 = mol.add_atom(Element::H, [0.74, 0.0, 0.0], 0.0);
        mol.add_bond(h1, h2, BondType::Single);
        mol
    }

    /// Build a large test molecule with N atoms (for performance testing).
    pub fn large_molecule(n_atoms: usize) -> Molecule {
        let mut mol = Molecule::new("LargeMolecule");
        let elements = [Element::C, Element::N, Element::O, Element::C, Element::S];
        for i in 0..n_atoms {
            let elem = elements[i % elements.len()];
            let x = (i as f64) * 1.5;
            let y = ((i as f64) * 0.7).sin() * 2.0;
            let z = ((i as f64) * 0.3).cos() * 1.0;
            mol.add_atom(elem, [x, y, z], ((i as f64) * 0.1).sin() * 0.1);
        }
        for i in 0..(n_atoms - 1) {
            mol.add_bond(i, i + 1, BondType::Single);
        }
        mol
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Molecule construction & properties ---

    #[test]
    fn test_water_construction() {
        let water = MolecularLibrary::water();
        assert_eq!(water.atoms.len(), 3);
        assert_eq!(water.bonds.len(), 2);
        assert!(water.validate().is_ok());
    }

    #[test]
    fn test_aspirin_construction() {
        let aspirin = MolecularLibrary::aspirin();
        assert!(aspirin.atoms.len() > 10);
        assert!(aspirin.bonds.len() > 10);
        assert!(aspirin.validate().is_ok());
        assert!(aspirin.smiles.is_some());
    }

    #[test]
    fn test_molecular_weight() {
        let water = MolecularLibrary::water();
        let mw = water.molecular_weight();
        // H2O = 2*1.008 + 15.999 = 18.015
        assert!((mw - 18.015).abs() < 0.01, "water MW = {}", mw);
    }

    #[test]
    fn test_h_bond_donors_acceptors() {
        let water = MolecularLibrary::water();
        // Water: 2 O-H bonds → 2 donors; O is acceptor → 1 acceptor
        assert_eq!(water.h_bond_donors(), 2);
        assert_eq!(water.h_bond_acceptors(), 1);
    }

    #[test]
    fn test_molecule_validation_empty() {
        let mol = Molecule::new("empty");
        assert!(mol.validate().is_err());
    }

    #[test]
    fn test_molecule_validation_bad_bond() {
        let mut mol = Molecule::new("bad");
        mol.add_atom(Element::C, [0.0, 0.0, 0.0], 0.0);
        mol.add_bond(0, 99, BondType::Single); // index 99 doesn't exist
        assert!(mol.validate().is_err());
    }

    #[test]
    fn test_distance_calculation() {
        let mut mol = Molecule::new("dist");
        mol.add_atom(Element::H, [0.0, 0.0, 0.0], 0.0);
        mol.add_atom(Element::H, [3.0, 4.0, 0.0], 0.0);
        let d = mol.distance(0, 1);
        assert!((d - 5.0).abs() < 1e-10, "distance = {}", d);
    }

    #[test]
    fn test_center_of_mass() {
        let h2 = MolecularLibrary::hydrogen_molecule();
        let com = h2.center_of_mass();
        // Two equal-mass atoms at x=0 and x=0.74 → COM at x=0.37
        assert!((com[0] - 0.37).abs() < 0.01, "COM x = {}", com[0]);
    }

    #[test]
    fn test_estimated_log_p() {
        let water = MolecularLibrary::water();
        let logp = water.estimated_log_p();
        // Water should have negative logP (hydrophilic)
        assert!(logp < 0.0, "water logP = {}", logp);
    }

    #[test]
    fn test_polar_surface_area() {
        let water = MolecularLibrary::water();
        let psa = water.polar_surface_area();
        assert!(psa > 0.0, "water PSA should be > 0, got {}", psa);
    }

    // --- Drug-likeness ---

    #[test]
    fn test_drug_likeness_aspirin() {
        let aspirin = MolecularLibrary::aspirin();
        let result = evaluate_drug_likeness(&aspirin);
        // Aspirin (MW ~180) should have 0 Lipinski violations
        assert_eq!(
            result.lipinski_violations, 0,
            "aspirin: {} violations",
            result.lipinski_violations
        );
        assert!(
            result.qed_score > 0.0 && result.qed_score <= 1.0,
            "QED = {}",
            result.qed_score
        );
        assert!(
            result.synthetic_accessibility > 0.0,
            "SA = {}",
            result.synthetic_accessibility
        );
    }

    #[test]
    fn test_drug_likeness_large_molecule() {
        let large = MolecularLibrary::large_molecule(100);
        let result = evaluate_drug_likeness(&large);
        // A 100-atom linear chain should violate MW > 500
        assert!(
            result.lipinski_violations > 0,
            "large mol should have violations"
        );
    }

    // --- Fingerprints ---

    #[test]
    fn test_fingerprint_self_similarity() {
        let aspirin = MolecularLibrary::aspirin();
        let fp = MolecularFingerprint::from_molecule(&aspirin, 128);
        let tanimoto = fp.tanimoto(&fp);
        assert!(
            (tanimoto - 1.0).abs() < 1e-10,
            "self-similarity should be 1.0, got {}",
            tanimoto
        );
    }

    #[test]
    fn test_fingerprint_different_molecules() {
        let aspirin = MolecularLibrary::aspirin();
        let caffeine = MolecularLibrary::caffeine();
        let fp_a = MolecularFingerprint::from_molecule(&aspirin, 128);
        let fp_c = MolecularFingerprint::from_molecule(&caffeine, 128);
        let tanimoto = fp_a.tanimoto(&fp_c);
        // Different molecules should have <1 similarity
        assert!(
            tanimoto < 1.0,
            "different molecules should have Tanimoto < 1.0, got {}",
            tanimoto
        );
        assert!(tanimoto >= 0.0, "Tanimoto should be non-negative");
    }

    // --- Quantum kernel ---

    #[test]
    fn test_quantum_kernel_self() {
        let aspirin = MolecularLibrary::aspirin();
        let fp = MolecularFingerprint::from_molecule(&aspirin, 64);
        let kernel = QuantumKernel::new(4);
        let k = kernel.compute(&fp, &fp);
        // Self-kernel is deterministic (same encoding → same state → same overlap)
        assert!(k > 0.0, "self-kernel should be positive, got {}", k);
        assert!(k <= 1.0, "self-kernel should be <= 1.0, got {}", k);
        // Verify determinism: computing again gives same result
        let k2 = kernel.compute(&fp, &fp);
        assert!(
            (k - k2).abs() < 1e-12,
            "self-kernel should be deterministic"
        );
    }

    #[test]
    fn test_quantum_kernel_matrix() {
        let mols = vec![
            MolecularLibrary::aspirin(),
            MolecularLibrary::ibuprofen(),
            MolecularLibrary::caffeine(),
        ];
        let fps: Vec<_> = mols
            .iter()
            .map(|m| MolecularFingerprint::from_molecule(m, 64))
            .collect();
        let kernel = QuantumKernel::new(4);
        let matrix = kernel.kernel_matrix(&fps);
        assert_eq!(matrix.len(), 3);
        // Diagonal should be 1.0
        for i in 0..3 {
            assert!(
                (matrix[i][i] - 1.0).abs() < 1e-6,
                "diagonal [{i}] = {}",
                matrix[i][i]
            );
        }
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10, "not symmetric");
            }
        }
    }

    // --- Docking ---

    #[test]
    fn test_docking_scorer() {
        let protein = MolecularLibrary::simple_protein_pocket();
        let ligand = MolecularLibrary::aspirin();
        let scorer = QuantumDockingScorer::new(DockingConfig {
            num_qubits: 4,
            scoring_function: ScoringFunction::HybridClassicalQuantum,
            num_conformations: 3,
            optimization_steps: 10,
        });
        let result = scorer.score(&protein, &ligand).unwrap();
        assert_eq!(result.all_scores.len(), 3);
        assert!(result.score.is_finite(), "docking score should be finite");
    }

    #[test]
    fn test_docking_kernel_scoring() {
        let protein = MolecularLibrary::simple_protein_pocket();
        let ligand = MolecularLibrary::caffeine();
        let scorer = QuantumDockingScorer::new(DockingConfig {
            num_qubits: 4,
            scoring_function: ScoringFunction::QuantumKernelScore,
            num_conformations: 2,
            optimization_steps: 5,
        });
        let result = scorer.score(&protein, &ligand).unwrap();
        // Kernel scoring produces negative scores
        assert!(
            result.score <= 0.0,
            "kernel score should be <= 0, got {}",
            result.score
        );
    }

    // --- Force field ---

    #[test]
    fn test_quantum_force_field() {
        let mol_a = MolecularLibrary::water();
        let mol_b = MolecularLibrary::hydrogen_molecule();
        let qff = QuantumForceField {
            num_qubits: 4,
            num_layers: 1,
            max_iterations: 10,
            convergence_threshold: 1e-3,
        };
        let energy = qff.interaction_energy(&mol_a, &mol_b).unwrap();
        assert!(
            energy.is_finite(),
            "interaction energy should be finite, got {}",
            energy
        );
    }

    // --- Binding affinity ---

    #[test]
    fn test_binding_affinity_estimator() {
        let protein = MolecularLibrary::simple_protein_pocket();
        let ligand = MolecularLibrary::aspirin();
        let mut estimator = BindingAffinityEstimator::new(4);
        estimator.set_from_molecules(&protein, &ligand);
        let pkd = estimator.estimate().unwrap();
        // pKd should be in [2, 12]
        assert!(pkd >= 2.0 && pkd <= 12.0, "pKd = {}", pkd);
    }

    #[test]
    fn test_binding_affinity_no_features() {
        let estimator = BindingAffinityEstimator::new(4);
        assert!(
            estimator.estimate().is_err(),
            "should fail without features"
        );
    }

    // --- ADMET prediction ---

    #[test]
    fn test_admet_prediction() {
        let aspirin = MolecularLibrary::aspirin();
        let predictor = AdmetPredictor::new(
            4,
            vec![
                AdmetProperty::Absorption,
                AdmetProperty::Toxicity,
                AdmetProperty::Solubility,
                AdmetProperty::BBBPermeability,
            ],
        );
        let results = predictor.predict(&aspirin).unwrap();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(
                r.probability >= 0.0 && r.probability <= 1.0,
                "prob = {}",
                r.probability
            );
            assert!(r.confidence >= 0.0, "confidence = {}", r.confidence);
        }
    }

    // --- Generative design ---

    #[test]
    fn test_molecular_generator_sample() {
        let config = GeneratorConfig {
            num_qubits: 4,
            latent_dim: 2,
            num_layers: 1,
            property_targets: Vec::new(),
        };
        let gen = QuantumMolecularGenerator::new(config);
        let samples = gen.sample(5);
        assert_eq!(samples.len(), 5);
        for s in &samples {
            assert!(s.score >= 0.0, "sample score should be non-negative");
            assert_eq!(s.bitstring.len(), 4);
        }
    }

    #[test]
    fn test_molecular_generator_train() {
        let target_fps: Vec<MolecularFingerprint> = vec![
            MolecularFingerprint::from_molecule(&MolecularLibrary::aspirin(), 4),
            MolecularFingerprint::from_molecule(&MolecularLibrary::caffeine(), 4),
        ];
        let config = GeneratorConfig {
            num_qubits: 4,
            latent_dim: 2,
            num_layers: 1,
            property_targets: Vec::new(),
        };
        let mut gen = QuantumMolecularGenerator::new(config);
        let loss = gen.train(&target_fps, 3);
        assert!(loss.is_finite(), "training loss should be finite");
    }

    // --- Lead optimization ---

    #[test]
    fn test_lead_optimization() {
        let lead = MolecularLibrary::aspirin();
        let protein = MolecularLibrary::simple_protein_pocket();
        let result = lead_optimization(&lead, &protein, 3, 4).unwrap();
        assert_eq!(result.scores_over_iterations.len(), 3);
        assert!(
            !result.pareto_front.is_empty(),
            "Pareto front should be non-empty"
        );
        assert!(result.best_score.is_finite());
    }

    // --- Pareto dominance ---

    #[test]
    fn test_pareto_dominance() {
        let a = ParetoPoint {
            objectives: vec![1.0, 2.0],
            candidate_idx: 0,
        };
        let b = ParetoPoint {
            objectives: vec![2.0, 3.0],
            candidate_idx: 1,
        };
        assert!(a.dominates(&b), "a should dominate b");
        assert!(!b.dominates(&a), "b should not dominate a");
    }

    #[test]
    fn test_pareto_non_dominated() {
        let a = ParetoPoint {
            objectives: vec![1.0, 3.0],
            candidate_idx: 0,
        };
        let b = ParetoPoint {
            objectives: vec![2.0, 1.0],
            candidate_idx: 1,
        };
        assert!(!a.dominates(&b), "trade-off: neither should dominate");
        assert!(!b.dominates(&a), "trade-off: neither should dominate");
    }

    // --- Drug discovery pipeline ---

    #[test]
    fn test_drug_pipeline() {
        let protein = MolecularLibrary::simple_protein_pocket();
        let candidates = vec![
            MolecularLibrary::aspirin(),
            MolecularLibrary::caffeine(),
            MolecularLibrary::ibuprofen(),
        ];
        let pipeline = DrugDiscoveryPipeline {
            stages: vec![
                PipelineStage::VirtualScreening { library_size: 3 },
                PipelineStage::BindingAffinity,
                PipelineStage::AdmetFiltering,
            ],
            num_qubits: 4,
        };
        let results = pipeline.run(&candidates, &protein).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].stage_name, "Virtual Screening");
        assert_eq!(results[1].stage_name, "Binding Affinity");
        assert_eq!(results[2].stage_name, "ADMET Filtering");
    }

    // --- KRAS inhibitor ---

    #[test]
    fn test_kras_inhibitor() {
        let kras = MolecularLibrary::kras_inhibitor();
        assert!(kras.validate().is_ok());
        assert!(kras.atoms.len() > 20);
        let dl = evaluate_drug_likeness(&kras);
        // A well-designed drug should have reasonable QED
        assert!(dl.qed_score > 0.0, "QED = {}", dl.qed_score);
    }

    // --- Large molecule performance ---

    #[test]
    fn test_large_molecule_fingerprint() {
        let mol = MolecularLibrary::large_molecule(50);
        let fp = MolecularFingerprint::from_molecule(&mol, 256);
        assert_eq!(fp.num_bits, 256);
        let set_bits: usize = fp.bits.iter().filter(|&&b| b).count();
        assert!(set_bits > 0, "fingerprint should have some bits set");
    }

    // --- Element properties ---

    #[test]
    fn test_element_properties() {
        assert_eq!(Element::C.atomic_number(), 6);
        assert!((Element::C.atomic_mass() - 12.011).abs() < 0.001);
        assert!((Element::O.electronegativity() - 3.44).abs() < 0.01);
        assert!(Element::O.is_hbond_acceptor());
        assert!(!Element::C.is_hbond_acceptor());
        assert!(Element::N.is_hbond_donor_heavy());
    }

    #[test]
    fn test_bond_properties() {
        assert_eq!(BondType::Single.order(), 1.0);
        assert_eq!(BondType::Double.order(), 2.0);
        assert_eq!(BondType::Aromatic.order(), 1.5);
        assert!(BondType::Single.is_rotatable());
        assert!(!BondType::Double.is_rotatable());
    }
}
