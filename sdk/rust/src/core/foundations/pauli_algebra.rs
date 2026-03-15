//! Universal Sparse Pauli String Representation
//!
//! Efficient Pauli algebra using packed bitstrings for x/z representation.
//! Supports operations: multiplication, commutation check, weight, inner product.
//!
//! Used by: Pauli propagation, near-Clifford, quantum chemistry, error mitigation,
//! stabilizer tensor networks.

use crate::C64;
use std::fmt;

// ===================================================================
// PAULI STRING (BITSTRING REPRESENTATION)
// ===================================================================

/// A Pauli string on n qubits, represented as two bitstrings (x, z).
///
/// For qubit j:
/// - (x=0, z=0) -> I
/// - (x=1, z=0) -> X
/// - (x=1, z=1) -> Y (up to phase)
/// - (x=0, z=1) -> Z
///
/// Packed into `Vec<u64>` to support millions of qubits efficiently.
/// Each u64 holds 64 qubit positions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PauliString {
    /// X bitstring: bit j set means X or Y on qubit j.
    pub x_bits: Vec<u64>,
    /// Z bitstring: bit j set means Z or Y on qubit j.
    pub z_bits: Vec<u64>,
    /// Number of qubits.
    pub num_qubits: usize,
}

impl PauliString {
    /// Number of u64 words needed for `n` qubits.
    #[inline]
    fn num_words(n: usize) -> usize {
        (n + 63) / 64
    }

    /// Create the identity Pauli string on `n` qubits.
    pub fn identity(num_qubits: usize) -> Self {
        let words = Self::num_words(num_qubits);
        PauliString {
            x_bits: vec![0u64; words],
            z_bits: vec![0u64; words],
            num_qubits,
        }
    }

    /// Create a single-qubit Pauli string.
    pub fn single(num_qubits: usize, qubit: usize, pauli: char) -> Self {
        let mut ps = Self::identity(num_qubits);
        ps.set_qubit(qubit, pauli);
        ps
    }

    /// Create from a string like "IXYZ" (qubit 0 = leftmost).
    pub fn from_str_rep(s: &str) -> Self {
        let n = s.len();
        let mut ps = Self::identity(n);
        for (i, ch) in s.chars().enumerate() {
            ps.set_qubit(i, ch);
        }
        ps
    }

    /// Set the Pauli operator on a specific qubit.
    pub fn set_qubit(&mut self, qubit: usize, pauli: char) {
        let word = qubit / 64;
        let bit = qubit % 64;
        let mask = 1u64 << bit;

        match pauli {
            'I' | 'i' => {
                self.x_bits[word] &= !mask;
                self.z_bits[word] &= !mask;
            }
            'X' | 'x' => {
                self.x_bits[word] |= mask;
                self.z_bits[word] &= !mask;
            }
            'Y' | 'y' => {
                self.x_bits[word] |= mask;
                self.z_bits[word] |= mask;
            }
            'Z' | 'z' => {
                self.x_bits[word] &= !mask;
                self.z_bits[word] |= mask;
            }
            _ => panic!("Invalid Pauli operator: {}", pauli),
        }
    }

    /// Get the Pauli operator on a specific qubit.
    pub fn get_qubit(&self, qubit: usize) -> char {
        let word = qubit / 64;
        let bit = qubit % 64;
        let x = (self.x_bits[word] >> bit) & 1;
        let z = (self.z_bits[word] >> bit) & 1;
        match (x, z) {
            (0, 0) => 'I',
            (1, 0) => 'X',
            (1, 1) => 'Y',
            (0, 1) => 'Z',
            _ => unreachable!(),
        }
    }

    /// Compute the Pauli weight (number of non-identity operators).
    pub fn weight(&self) -> usize {
        let mut w = 0usize;
        for i in 0..self.x_bits.len() {
            // Non-identity where x OR z is set
            w += (self.x_bits[i] | self.z_bits[i]).count_ones() as usize;
        }
        w
    }

    /// Check if this is the identity string.
    pub fn is_identity(&self) -> bool {
        self.x_bits.iter().all(|&x| x == 0) && self.z_bits.iter().all(|&z| z == 0)
    }

    /// Check if two Pauli strings commute.
    ///
    /// Two Pauli strings commute iff the number of positions where they
    /// anti-commute is even. At position j, they anti-commute iff
    /// x1_j * z2_j + z1_j * x2_j is odd (symplectic inner product).
    pub fn commutes_with(&self, other: &PauliString) -> bool {
        assert_eq!(self.num_qubits, other.num_qubits);
        let mut count = 0u32;
        for i in 0..self.x_bits.len() {
            // Symplectic inner product per word
            count += (self.x_bits[i] & other.z_bits[i]).count_ones();
            count += (self.z_bits[i] & other.x_bits[i]).count_ones();
        }
        count % 2 == 0
    }

    /// Multiply two Pauli strings: P1 * P2 = phase * P3.
    ///
    /// Returns (phase, result) where phase is a power of i: i^phase_power.
    /// phase_power is in {0, 1, 2, 3}.
    pub fn multiply(&self, other: &PauliString) -> (u8, PauliString) {
        assert_eq!(self.num_qubits, other.num_qubits);
        let words = self.x_bits.len();
        let mut result_x = vec![0u64; words];
        let mut result_z = vec![0u64; words];
        let mut phase = 0u32;

        for w in 0..words {
            // Per-qubit multiplication via symplectic algebra
            let x1 = self.x_bits[w];
            let z1 = self.z_bits[w];
            let x2 = other.x_bits[w];
            let z2 = other.z_bits[w];

            // Result Pauli: XOR
            result_x[w] = x1 ^ x2;
            result_z[w] = z1 ^ z2;

            // Phase computation: count anti-commutations
            // When P_a * P_b on same qubit:
            // X*Y = iZ, Y*X = -iZ, X*Z = -iY, Z*X = iY, Y*Z = iX, Z*Y = -iX
            // Phase contribution = 2*(x1&z2 - z1&x2) mod 4 per qubit,
            // but we need to count more carefully.

            // Simplified: phase += popcount(x1 & z2) - popcount(z1 & x2)
            // This counts the symplectic contribution.
            // For the Y = XZ convention, the full phase is:
            // i^(2 * sum_j (x1_j * z1_j + x2_j * z2_j - result_x_j * result_z_j)) ...
            // We use the standard formula: phase from Pauli multiplication lookup.
            // For each qubit: (x1,z1)*(x2,z2) -> phase_contribution

            // Count each case using bit tricks
            let y1 = x1 & z1; // Y positions in P1
            let y2 = x2 & z2; // Y positions in P2

            // X*Y = iZ: x1 & !z1 & x2 & z2
            phase += (x1 & !z1 & y2).count_ones();
            // Y*Z = iX: x1 & z1 & !x2 & z2
            phase += (y1 & !x2 & z2).count_ones();
            // Z*X = iY: !x1 & z1 & x2 & !z2
            phase += (!x1 & z1 & x2 & !z2).count_ones();

            // Y*X = -iZ = i^3 Z
            phase += 3 * (y1 & x2 & !z2).count_ones();
            // Z*Y = -iX = i^3 X
            phase += 3 * (!x1 & z1 & y2).count_ones();
            // X*Z = -iY = i^3 Y
            phase += 3 * (x1 & !z1 & !x2 & z2).count_ones();
        }

        let result = PauliString {
            x_bits: result_x,
            z_bits: result_z,
            num_qubits: self.num_qubits,
        };

        ((phase % 4) as u8, result)
    }

    /// Convert to a vector of chars for compatibility.
    pub fn to_chars(&self) -> Vec<char> {
        (0..self.num_qubits).map(|i| self.get_qubit(i)).collect()
    }

    /// Compute the symplectic inner product with another Pauli string.
    /// Returns 0 (commute) or 1 (anti-commute).
    pub fn symplectic_inner_product(&self, other: &PauliString) -> u8 {
        if self.commutes_with(other) {
            0
        } else {
            1
        }
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.num_qubits {
            write!(f, "{}", self.get_qubit(i))?;
        }
        Ok(())
    }
}

// ===================================================================
// WEIGHTED PAULI STRING
// ===================================================================

/// A Pauli string with a complex coefficient.
#[derive(Clone, Debug)]
pub struct WeightedPauliString {
    /// The Pauli string.
    pub pauli: PauliString,
    /// Complex coefficient.
    pub coeff: C64,
}

impl WeightedPauliString {
    /// Create a new weighted Pauli string.
    pub fn new(pauli: PauliString, coeff: C64) -> Self {
        WeightedPauliString { pauli, coeff }
    }

    /// Create with unit coefficient.
    pub fn unit(pauli: PauliString) -> Self {
        Self::new(pauli, C64::new(1.0, 0.0))
    }

    /// Multiply two weighted Pauli strings.
    pub fn multiply(&self, other: &WeightedPauliString) -> WeightedPauliString {
        let (phase, result_pauli) = self.pauli.multiply(&other.pauli);

        // phase is i^phase_power
        let phase_factor = match phase {
            0 => C64::new(1.0, 0.0),
            1 => C64::new(0.0, 1.0),
            2 => C64::new(-1.0, 0.0),
            3 => C64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        WeightedPauliString {
            pauli: result_pauli,
            coeff: self.coeff * other.coeff * phase_factor,
        }
    }

    /// Check if the coefficient is negligible.
    pub fn is_negligible(&self, threshold: f64) -> bool {
        self.coeff.norm() < threshold
    }
}

impl fmt::Display for WeightedPauliString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({:.4}+{:.4}i) {}",
            self.coeff.re, self.coeff.im, self.pauli
        )
    }
}

// ===================================================================
// PAULI PROPAGATOR TRAIT
// ===================================================================

/// Trait for propagating Pauli strings through gates (Heisenberg picture).
///
/// Given a gate U, transforms P -> U† P U.
pub trait PauliPropagator {
    /// Conjugate a Pauli string by this gate: P -> U† P U.
    /// May return multiple strings (e.g., for non-Clifford gates like T).
    fn conjugate(&self, pauli: &WeightedPauliString) -> Vec<WeightedPauliString>;
}

// ===================================================================
// PAULI SUM (SPARSE OPERATOR)
// ===================================================================

/// A sparse representation of a quantum operator as a sum of weighted Pauli strings.
#[derive(Clone, Debug)]
pub struct PauliSum {
    pub terms: Vec<WeightedPauliString>,
    pub num_qubits: usize,
}

impl PauliSum {
    /// Create an empty Pauli sum.
    pub fn new(num_qubits: usize) -> Self {
        PauliSum {
            terms: Vec::new(),
            num_qubits,
        }
    }

    /// Create from a single Pauli string.
    pub fn from_term(term: WeightedPauliString) -> Self {
        let n = term.pauli.num_qubits;
        PauliSum {
            terms: vec![term],
            num_qubits: n,
        }
    }

    /// Add a term.
    pub fn add_term(&mut self, term: WeightedPauliString) {
        self.terms.push(term);
    }

    /// Simplify by combining identical Pauli strings and removing negligible terms.
    pub fn simplify(&mut self, threshold: f64) {
        use std::collections::HashMap;

        let mut map: HashMap<PauliString, C64> = HashMap::new();
        for term in &self.terms {
            let entry = map.entry(term.pauli.clone()).or_insert(C64::new(0.0, 0.0));
            *entry += term.coeff;
        }

        self.terms = map
            .into_iter()
            .filter(|(_, c)| c.norm() >= threshold)
            .map(|(p, c)| WeightedPauliString::new(p, c))
            .collect();
    }

    /// Total number of terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Compute the trace: Tr(P) / 2^n.
    /// Only the identity term contributes.
    pub fn trace_normalized(&self) -> C64 {
        let mut sum = C64::new(0.0, 0.0);
        for term in &self.terms {
            if term.pauli.is_identity() {
                sum += term.coeff;
            }
        }
        sum
    }
}

// ===================================================================
// CLIFFORD CONJUGATION TABLE
// ===================================================================

/// Lookup table for Clifford gate conjugation of single-qubit Paulis.
///
/// For each Clifford gate C, stores C† P C for P in {X, Y, Z}.
/// Result is (new_pauli, phase_sign) where phase_sign is +1 or -1.
pub struct CliffordConjugationTable;

impl CliffordConjugationTable {
    /// Conjugate single-qubit Pauli `p` by Hadamard: H† P H.
    /// H†XH = Z, H†YH = -Y, H†ZH = X
    pub fn hadamard(p: char) -> (char, i8) {
        match p {
            'X' => ('Z', 1),
            'Y' => ('Y', -1),
            'Z' => ('X', 1),
            'I' => ('I', 1),
            _ => panic!("Invalid Pauli: {}", p),
        }
    }

    /// Conjugate single-qubit Pauli `p` by S gate: S† P S.
    /// S†XS = Y, S†YS = -X, S†ZS = Z
    pub fn s_gate(p: char) -> (char, i8) {
        match p {
            'X' => ('Y', 1),
            'Y' => ('X', -1),
            'Z' => ('Z', 1),
            'I' => ('I', 1),
            _ => panic!("Invalid Pauli: {}", p),
        }
    }

    /// Conjugate single-qubit Pauli `p` by X gate: X† P X.
    /// X†XX = X, X†YX = -Y, X†ZX = -Z
    pub fn x_gate(p: char) -> (char, i8) {
        match p {
            'X' => ('X', 1),
            'Y' => ('Y', -1),
            'Z' => ('Z', -1),
            'I' => ('I', 1),
            _ => panic!("Invalid Pauli: {}", p),
        }
    }

    /// Conjugate single-qubit Pauli `p` by Y gate: Y† P Y.
    /// Y†XY = -X, Y†YY = Y, Y†ZY = -Z
    pub fn y_gate(p: char) -> (char, i8) {
        match p {
            'X' => ('X', -1),
            'Y' => ('Y', 1),
            'Z' => ('Z', -1),
            'I' => ('I', 1),
            _ => panic!("Invalid Pauli: {}", p),
        }
    }

    /// Conjugate single-qubit Pauli `p` by Z gate: Z† P Z.
    /// Z†XZ = -X, Z†YZ = -Y, Z†ZZ = Z
    pub fn z_gate(p: char) -> (char, i8) {
        match p {
            'X' => ('X', -1),
            'Y' => ('Y', -1),
            'Z' => ('Z', 1),
            'I' => ('I', 1),
            _ => panic!("Invalid Pauli: {}", p),
        }
    }

    /// Conjugate a two-qubit Pauli pair (p_ctrl, p_targ) by CNOT.
    /// CNOT† (P_c ⊗ P_t) CNOT:
    ///   IX -> IX, XI -> XX, IZ -> ZZ, ZI -> ZI
    ///   XZ -> -YY, ZX -> ZX, XX -> XI, ZZ -> IZ
    pub fn cnot(p_ctrl: char, p_targ: char) -> (char, char, i8) {
        match (p_ctrl, p_targ) {
            ('I', 'I') => ('I', 'I', 1),
            ('I', 'X') => ('I', 'X', 1),
            ('I', 'Y') => ('Z', 'Y', 1),
            ('I', 'Z') => ('Z', 'Z', 1),
            ('X', 'I') => ('X', 'X', 1),
            ('X', 'X') => ('X', 'I', 1),
            ('X', 'Y') => ('Y', 'Z', -1),
            ('X', 'Z') => ('Y', 'Y', 1),
            ('Y', 'I') => ('Y', 'X', 1),
            ('Y', 'X') => ('Y', 'I', 1),
            ('Y', 'Y') => ('X', 'Z', 1),
            ('Y', 'Z') => ('X', 'Y', -1),
            ('Z', 'I') => ('Z', 'I', 1),
            ('Z', 'X') => ('Z', 'X', 1),
            ('Z', 'Y') => ('I', 'Y', 1),
            ('Z', 'Z') => ('I', 'Z', 1),
            _ => panic!("Invalid Paulis: ({}, {})", p_ctrl, p_targ),
        }
    }

    /// Conjugate a two-qubit Pauli pair by CZ.
    /// CZ† (P_a ⊗ P_b) CZ:
    ///   XI -> XZ, IX -> ZX, ZI -> ZI, IZ -> IZ
    pub fn cz(p_a: char, p_b: char) -> (char, char, i8) {
        match (p_a, p_b) {
            ('I', 'I') => ('I', 'I', 1),
            ('I', 'X') => ('Z', 'X', 1),
            ('I', 'Y') => ('Z', 'Y', 1),
            ('I', 'Z') => ('I', 'Z', 1),
            ('X', 'I') => ('X', 'Z', 1),
            ('X', 'X') => ('Y', 'Y', -1),
            ('X', 'Y') => ('Y', 'X', 1),
            ('X', 'Z') => ('X', 'I', 1),
            ('Y', 'I') => ('Y', 'Z', 1),
            ('Y', 'X') => ('X', 'Y', 1),
            ('Y', 'Y') => ('X', 'X', -1),
            ('Y', 'Z') => ('Y', 'I', 1),
            ('Z', 'I') => ('Z', 'I', 1),
            ('Z', 'X') => ('I', 'X', 1),
            ('Z', 'Y') => ('I', 'Y', 1),
            ('Z', 'Z') => ('Z', 'Z', 1),
            _ => panic!("Invalid Paulis: ({}, {})", p_a, p_b),
        }
    }

    /// Conjugate a two-qubit Pauli pair by SWAP.
    pub fn swap(p_a: char, p_b: char) -> (char, char, i8) {
        // SWAP just exchanges the Paulis
        (p_b, p_a, 1)
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_string_identity() {
        let ps = PauliString::identity(4);
        assert_eq!(ps.weight(), 0);
        assert!(ps.is_identity());
        assert_eq!(ps.to_string(), "IIII");
    }

    #[test]
    fn test_pauli_string_single() {
        let ps = PauliString::single(3, 1, 'X');
        assert_eq!(ps.get_qubit(0), 'I');
        assert_eq!(ps.get_qubit(1), 'X');
        assert_eq!(ps.get_qubit(2), 'I');
        assert_eq!(ps.weight(), 1);
    }

    #[test]
    fn test_pauli_string_from_str() {
        let ps = PauliString::from_str_rep("IXYZ");
        assert_eq!(ps.get_qubit(0), 'I');
        assert_eq!(ps.get_qubit(1), 'X');
        assert_eq!(ps.get_qubit(2), 'Y');
        assert_eq!(ps.get_qubit(3), 'Z');
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_commutation() {
        // XX commutes with ZZ
        let p1 = PauliString::from_str_rep("XX");
        let p2 = PauliString::from_str_rep("ZZ");
        assert!(p1.commutes_with(&p2));

        // XZ anti-commutes on qubit 0 and anti-commutes on qubit 1
        // -> two anti-commutations -> commutes
        assert!(p1.commutes_with(&p2));

        // XI and ZI anti-commute (one anti-commutation)
        let p3 = PauliString::from_str_rep("XI");
        let p4 = PauliString::from_str_rep("ZI");
        assert!(!p3.commutes_with(&p4));

        // XY and ZI: anti-commute at qubit 0, commute at qubit 1 -> anti-commute
        let p5 = PauliString::from_str_rep("XY");
        let p6 = PauliString::from_str_rep("ZI");
        assert!(!p5.commutes_with(&p6));
    }

    #[test]
    fn test_multiplication() {
        // X * Y = iZ
        let p1 = PauliString::single(1, 0, 'X');
        let p2 = PauliString::single(1, 0, 'Y');
        let (phase, result) = p1.multiply(&p2);
        assert_eq!(result.get_qubit(0), 'Z');
        assert_eq!(phase, 1); // i^1 = i

        // X * X = I
        let p3 = PauliString::single(1, 0, 'X');
        let (phase2, result2) = p3.multiply(&p3);
        assert!(result2.is_identity());
        assert_eq!(phase2, 0); // i^0 = 1
    }

    #[test]
    fn test_weighted_multiplication() {
        let wp1 = WeightedPauliString::new(PauliString::single(1, 0, 'X'), C64::new(2.0, 0.0));
        let wp2 = WeightedPauliString::new(PauliString::single(1, 0, 'Y'), C64::new(3.0, 0.0));
        let result = wp1.multiply(&wp2);
        // 2 * 3 * i = 6i
        assert!((result.coeff.re - 0.0).abs() < 1e-10);
        assert!((result.coeff.im - 6.0).abs() < 1e-10);
        assert_eq!(result.pauli.get_qubit(0), 'Z');
    }

    #[test]
    fn test_pauli_sum_simplify() {
        let mut sum = PauliSum::new(2);
        sum.add_term(WeightedPauliString::new(
            PauliString::from_str_rep("XZ"),
            C64::new(1.0, 0.0),
        ));
        sum.add_term(WeightedPauliString::new(
            PauliString::from_str_rep("XZ"),
            C64::new(2.0, 0.0),
        ));
        sum.add_term(WeightedPauliString::new(
            PauliString::from_str_rep("ZX"),
            C64::new(0.5, 0.0),
        ));
        sum.simplify(0.01);
        assert_eq!(sum.len(), 2);

        // Find the XZ term
        let xz_term = sum
            .terms
            .iter()
            .find(|t| t.pauli.to_string() == "XZ")
            .unwrap();
        assert!((xz_term.coeff.re - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clifford_hadamard_conjugation() {
        let (p, s) = CliffordConjugationTable::hadamard('X');
        assert_eq!(p, 'Z');
        assert_eq!(s, 1);

        let (p, s) = CliffordConjugationTable::hadamard('Y');
        assert_eq!(p, 'Y');
        assert_eq!(s, -1);

        let (p, s) = CliffordConjugationTable::hadamard('Z');
        assert_eq!(p, 'X');
        assert_eq!(s, 1);
    }

    #[test]
    fn test_clifford_s_conjugation() {
        let (p, s) = CliffordConjugationTable::s_gate('X');
        assert_eq!(p, 'Y');
        assert_eq!(s, 1);

        let (p, s) = CliffordConjugationTable::s_gate('Y');
        assert_eq!(p, 'X');
        assert_eq!(s, -1);
    }

    #[test]
    fn test_cnot_conjugation() {
        // CNOT† (XI) CNOT = XX
        let (pc, pt, s) = CliffordConjugationTable::cnot('X', 'I');
        assert_eq!(pc, 'X');
        assert_eq!(pt, 'X');
        assert_eq!(s, 1);

        // CNOT† (IZ) CNOT = ZZ
        let (pc, pt, s) = CliffordConjugationTable::cnot('I', 'Z');
        assert_eq!(pc, 'Z');
        assert_eq!(pt, 'Z');
        assert_eq!(s, 1);
    }

    #[test]
    fn test_large_pauli_string() {
        // Test with >64 qubits (requires multiple words)
        let n = 100;
        let mut ps = PauliString::identity(n);
        ps.set_qubit(0, 'X');
        ps.set_qubit(63, 'Y');
        ps.set_qubit(64, 'Z');
        ps.set_qubit(99, 'X');

        assert_eq!(ps.get_qubit(0), 'X');
        assert_eq!(ps.get_qubit(63), 'Y');
        assert_eq!(ps.get_qubit(64), 'Z');
        assert_eq!(ps.get_qubit(99), 'X');
        assert_eq!(ps.weight(), 4);
    }

    #[test]
    fn test_pauli_display() {
        let ps = PauliString::from_str_rep("XYZIXZ");
        assert_eq!(format!("{}", ps), "XYZIXZ");
    }

    #[test]
    fn test_swap_conjugation() {
        let (pa, pb, s) = CliffordConjugationTable::swap('X', 'Z');
        assert_eq!(pa, 'Z');
        assert_eq!(pb, 'X');
        assert_eq!(s, 1);
    }
}
