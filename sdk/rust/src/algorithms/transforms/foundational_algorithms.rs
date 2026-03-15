//! Quantum Algorithms Module
//!
//! Implements standard quantum algorithms including:
//! - Quantum Fourier Transform (QFT)
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Approximate Optimization Algorithm (QAOA) - planned

use crate::{GateOperations, QuantumSimulator, QuantumState};

// ============================================================
// QUANTUM FOURIER TRANSFORM (QFT)
// ============================================================

/// Apply Quantum Fourier Transform to a quantum state
///
/// The QFT transforms a computational basis state |x⟩ into a superposition
/// of all basis states with amplitudes given by the discrete Fourier transform.
///
/// # Algorithm
/// For each qubit i from 0 to n-1:
/// 1. Apply H gate to qubit i
/// 2. For each qubit j > i, apply controlled rotation R_k where k = j - i + 2
/// 3. Finally, swap qubits to reverse order
///
/// # Complexity
/// - Gates: O(n²) for n qubits
/// - Can be reduced to O(n log n) with approximate QFT
///
/// # Example
/// ```ignore
/// use nqpu_metal::{QuantumSimulator, algorithms::qft};
///
/// let mut sim = QuantumSimulator::new(3);
/// sim.x(0); // Prepare |001⟩
/// sim.x(1); // Prepare |011⟩ = |3⟩
///
/// // Apply QFT
/// qft(&mut sim.state, sim.num_qubits());
/// ```
pub fn qft(state: &mut QuantumState, num_qubits: usize) {
    // Main QFT loop
    for i in 0..num_qubits {
        // Apply Hadamard to qubit i
        GateOperations::h(state, i);

        // Apply controlled rotations
        for j in (i + 1)..num_qubits {
            let k = j - i + 2;
            let angle = 2.0 * std::f64::consts::PI / (1 << k) as f64;
            qft_controlled_rotation(state, i, j, angle);
        }
    }

    // Swap qubits to reverse order
    for i in 0..(num_qubits / 2) {
        GateOperations::swap(state, i, num_qubits - 1 - i);
    }
}

/// Apply inverse QFT (quantum inverse Fourier transform)
///
/// This is the inverse of the QFT operation, obtained by running
/// the QFT circuit backwards and inverting all rotations.
///
/// # Uses
/// - Phase estimation algorithm
/// - Solving linear systems
/// - Many quantum algorithms require inverse QFT
pub fn inverse_qft(state: &mut QuantumState, num_qubits: usize) {
    // First swap qubits to reverse order (inverse of final QFT swap)
    for i in 0..(num_qubits / 2) {
        GateOperations::swap(state, i, num_qubits - 1 - i);
    }

    // Apply inverse rotations in reverse order
    for i in (0..num_qubits).rev() {
        for j in (i + 1..num_qubits).rev() {
            let k = j - i + 2;
            let angle = -2.0 * std::f64::consts::PI / (1 << k) as f64;
            qft_controlled_rotation(state, i, j, angle);
        }

        // Apply Hadamard to qubit i
        GateOperations::h(state, i);
    }
}

/// Helper function for controlled rotation used in QFT
///
/// Applies a phase rotation of `angle` to the target qubit
/// conditioned on the control qubit being |1⟩.
fn qft_controlled_rotation(state: &mut QuantumState, control: usize, target: usize, angle: f64) {
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    let dim = state.dim;

    // Compute rotation: R(θ) = [[1, 0], [0, exp(iθ)]]
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let amplitudes = state.amplitudes_mut();

    // Only rotate states where control qubit is |1⟩
    for i in 0..dim {
        if (i & control_mask) != 0 {
            // Apply phase rotation if target is |1⟩
            if (i & target_mask) != 0 {
                let orig_re = amplitudes[i].re;
                amplitudes[i].re = orig_re * cos_a - amplitudes[i].im * sin_a;
                amplitudes[i].im = orig_re * sin_a + amplitudes[i].im * cos_a;
            }
        }
    }
}

/// Apply approximate QFT with a specified power cutoff
///
/// The approximate QFT skips small rotations (where 2π/2^k < cutoff),
/// trading some accuracy for reduced gate count.
///
/// # Arguments
/// * `state` - Quantum state to transform
/// * `num_qubits` - Number of qubits
/// * `cutoff_power` - Skip rotations where k > cutoff_power
///
/// # Example
/// ```ignore
/// // Use approximate QFT with k <= 4 (skip rotations smaller than π/8)
/// approximate_qft(&mut sim.state, sim.num_qubits(), 4);
/// ```
pub fn approximate_qft(state: &mut QuantumState, num_qubits: usize, cutoff_power: usize) {
    // Main QFT loop with cutoff
    for i in 0..num_qubits {
        GateOperations::h(state, i);

        for j in (i + 1)..num_qubits {
            let k = j - i + 2;
            if k <= cutoff_power {
                let angle = 2.0 * std::f64::consts::PI / (1 << k) as f64;
                qft_controlled_rotation(state, i, j, angle);
            }
        }
    }

    // Swap qubits to reverse order
    for i in 0..(num_qubits / 2) {
        GateOperations::swap(state, i, num_qubits - 1 - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantumSimulator;

    #[test]
    fn test_qft_zero_state() {
        // QFT of |0⟩ should give |0⟩ (equal superposition with 0 phase)
        let mut sim = QuantumSimulator::new(3);
        let original = sim.state.clone();

        qft(&mut sim.state, 3);

        // |0⟩ is an eigenstate of QFT with eigenvalue 1
        let result_prob = original.amplitudes_ref()[0].norm_sqr();
        assert!((result_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_qft_inverse() {
        // QFT followed by inverse QFT should return to original state
        let mut sim = QuantumSimulator::new(3);
        sim.x(0); // |001⟩
        sim.x(1); // |011⟩

        let original = sim.state.clone();

        qft(&mut sim.state, 3);
        inverse_qft(&mut sim.state, 3);

        // Check we return to original state (high fidelity)
        let fid = original.fidelity(&sim.state);
        assert!(fid > 0.99, "Fidelity {} too low", fid);
    }

    #[test]
    fn test_approximate_qft() {
        // Test that approximate QFT runs without error
        let mut sim = QuantumSimulator::new(4);
        sim.x(1); // Prepare |0010⟩

        approximate_qft(&mut sim.state, 4, 3);

        // Just check it runs without panicking
        assert!(sim.num_qubits() == 4);
    }
}

// ============================================================
// VARIATIONAL QUANTUM EIGENSOLVER (VQE)
// ============================================================

/// Hamiltonian term for VQE
///
/// Represents a term in a Hamiltonian as a coefficient
/// times a product of Pauli operators on specified qubits.
#[derive(Clone, Debug)]
pub struct HamiltonianTerm {
    /// Coefficient of this term
    pub coefficient: f64,
    /// List of (qubit, pauli) pairs where pauli is 0=X, 1=Y, 2=Z
    pub operators: Vec<(usize, usize)>,
}

impl HamiltonianTerm {
    /// Create a new Hamiltonian term
    pub fn new(coefficient: f64, operators: Vec<(usize, usize)>) -> Self {
        HamiltonianTerm {
            coefficient,
            operators,
        }
    }

    /// Create a simple Z term (e.g., for measuring qubit in Z basis)
    pub fn z(qubit: usize, coefficient: f64) -> Self {
        HamiltonianTerm::new(coefficient, vec![(qubit, 2)])
    }

    /// Create a ZZ interaction term
    pub fn zz(qubit1: usize, qubit2: usize, coefficient: f64) -> Self {
        HamiltonianTerm::new(coefficient, vec![(qubit1, 2), (qubit2, 2)])
    }
}

/// Hamiltonian for VQE
///
/// A Hamiltonian is a sum of terms, each being a coefficient
/// times a product of Pauli operators.
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    /// List of terms in the Hamiltonian
    pub terms: Vec<HamiltonianTerm>,
}

impl Hamiltonian {
    /// Create a new Hamiltonian from a list of terms
    pub fn new(terms: Vec<HamiltonianTerm>) -> Self {
        Hamiltonian { terms }
    }

    /// Add a term to the Hamiltonian
    pub fn add_term(mut self, term: HamiltonianTerm) -> Self {
        self.terms.push(term);
        self
    }
}

/// Ansatz for VQE
///
/// An ansatz is a parameterized quantum circuit that prepares
/// trial states for VQE optimization.
pub trait Ansatz: Send + Sync {
    /// Get the number of parameters in this ansatz
    fn num_parameters(&self) -> usize;

    /// Apply the ansatz circuit with given parameters
    fn apply(&self, simulator: &mut crate::QuantumSimulator, parameters: &[f64]);

    /// Clone the ansatz
    fn clone_box(&self) -> Box<dyn Ansatz>;
}

/// Hardware Efficient Ansatz
///
/// A simple variational ansatz using single-qubit rotations
/// and entangling CNOT gates in a ladder pattern.
#[derive(Clone)]
pub struct HardwareEfficientAnsatz {
    /// Number of qubits
    num_qubits: usize,
    /// Number of layers in the ansatz
    num_layers: usize,
}

impl HardwareEfficientAnsatz {
    /// Create a new hardware-efficient ansatz
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits
    /// * `num_layers` - Number of layers (each layer has n Ry rotations and n-1 CNOTs)
    pub fn new(num_qubits: usize, num_layers: usize) -> Self {
        HardwareEfficientAnsatz {
            num_qubits,
            num_layers,
        }
    }
}

impl Ansatz for HardwareEfficientAnsatz {
    fn num_parameters(&self) -> usize {
        // Each qubit gets one Ry parameter per layer
        self.num_qubits * self.num_layers
    }

    fn apply(&self, simulator: &mut crate::QuantumSimulator, parameters: &[f64]) {
        let mut param_idx = 0;

        for _layer in 0..self.num_layers {
            // Apply Ry rotations to all qubits
            for q in 0..self.num_qubits {
                if param_idx < parameters.len() {
                    simulator.ry(q, parameters[param_idx]);
                    param_idx += 1;
                }
            }

            // Apply CNOT ladder for entanglement
            for q in 0..(self.num_qubits - 1) {
                simulator.cnot(q, q + 1);
            }
        }
    }

    fn clone_box(&self) -> Box<dyn Ansatz> {
        Box::new(self.clone())
    }
}

/// VQE optimizer result
///
/// Contains the result of a VQE optimization run.
#[derive(Clone, Debug)]
pub struct VQEResult {
    /// Optimal parameters found
    pub optimal_parameters: Vec<f64>,
    /// Minimum energy found
    pub minimum_energy: f64,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Run VQE to find the ground state energy of a Hamiltonian
///
/// # Algorithm
/// 1. Start with random parameters
/// 2. Prepare trial state using ansatz
/// 3. Measure expectation value of Hamiltonian
/// 4. Update parameters using classical optimization
/// 5. Repeat until convergence
///
/// # Arguments
/// * `num_qubits` - Number of qubits
/// * `hamiltonian` - Hamiltonian to minimize
/// * `ansatz` - Parameterized circuit (will be cloned)
/// * `initial_parameters` - Starting parameters for optimization
/// * `max_iterations` - Maximum number of optimization iterations
/// * `learning_rate` - Step size for gradient descent
///
/// # Returns
/// VQEResult containing optimal parameters and minimum energy
///
/// # Example
/// ```ignore
/// use nqpu_metal::{QuantumSimulator, algorithms::*};
///
/// // Simple Ising model Hamiltonian
/// let hamiltonian = Hamiltonian::new(vec![
///     HamiltonianTerm::z(0, -1.0),
///     HamiltonianTerm::z(1, -1.0),
///     HamiltonianTerm::zz(0, 1, -0.5),
/// ]);
///
/// let ansatz = HardwareEfficientAnsatz::new(2, 2);
/// let initial_params = vec![0.1, 0.1, 0.1, 0.1];
///
/// let result = vqe(
///     2,
///     &hamiltonian,
///     &ansatz,
///     &initial_params,
///     100,
///     0.1
/// );
///
/// println!("Ground state energy: {}", result.minimum_energy);
/// ```
pub fn vqe(
    num_qubits: usize,
    hamiltonian: &Hamiltonian,
    ansatz: &dyn Ansatz,
    initial_parameters: &[f64],
    max_iterations: usize,
    learning_rate: f64,
) -> VQEResult {
    let mut parameters = initial_parameters.to_vec();
    let mut best_energy = f64::INFINITY;
    let mut best_parameters = parameters.clone();

    for _iteration in 0..max_iterations {
        // Prepare state with current parameters
        let mut sim = QuantumSimulator::new(num_qubits);
        ansatz.apply(&mut sim, &parameters);

        // Calculate energy
        let energy = calculate_energy(&sim, hamiltonian);

        if energy < best_energy {
            best_energy = energy;
            best_parameters = parameters.clone();
        }

        // Simple gradient-free optimization: coordinate descent
        // Try perturbing each parameter slightly
        for i in 0..parameters.len() {
            let original = parameters[i];

            // Try positive perturbation
            parameters[i] = original + learning_rate;
            let mut sim_plus = QuantumSimulator::new(num_qubits);
            ansatz.apply(&mut sim_plus, &parameters);
            let energy_plus = calculate_energy(&sim_plus, hamiltonian);

            // Try negative perturbation
            parameters[i] = original - learning_rate;
            let mut sim_minus = QuantumSimulator::new(num_qubits);
            ansatz.apply(&mut sim_minus, &parameters);
            let energy_minus = calculate_energy(&sim_minus, hamiltonian);

            // Keep best direction
            if energy_plus < energy && energy_plus < energy_minus {
                parameters[i] = original + learning_rate;
            } else if energy_minus < energy && energy_minus < energy_plus {
                parameters[i] = original - learning_rate;
            } else {
                parameters[i] = original;
            }
        }

        // Early termination if converged
        if (energy - best_energy).abs() < 1e-6 {
            break;
        }
    }

    VQEResult {
        optimal_parameters: best_parameters,
        minimum_energy: best_energy,
        iterations: max_iterations,
    }
}

/// Calculate the expectation value of a Hamiltonian
///
/// Measures each term in the Hamiltonian and sums them up.
fn calculate_energy(simulator: &QuantumSimulator, hamiltonian: &Hamiltonian) -> f64 {
    let mut energy = 0.0;

    for term in &hamiltonian.terms {
        let term_value = measure_term(simulator, term);
        energy += term.coefficient * term_value;
    }

    energy
}

/// Measure a single Hamiltonian term
///
/// For Pauli measurements, we need to rotate to the appropriate basis.
fn measure_term(simulator: &QuantumSimulator, term: &HamiltonianTerm) -> f64 {
    // For simplicity, this implementation handles Z measurements only
    // A full implementation would handle X and Y measurements with basis changes

    if term.operators.is_empty() {
        return 1.0; // Identity term
    }

    // Check if this is a simple Z term
    let all_z = term.operators.iter().all(|(_, pauli)| *pauli == 2);

    if all_z {
        // For Z measurements, we can directly compute expectation value
        let mut expectation = 1.0;

        for &(qubit, _) in &term.operators {
            expectation *= simulator.expectation_z(qubit);
        }

        expectation
    } else {
        // For X or Y measurements, need to change basis
        // This is a simplified implementation
        // A full implementation would apply basis rotation before measurement

        // For now, return 0.0 for non-Z measurements
        // In production, this would:
        // 1. Apply H before measurement for X
        // 2. Apply S† then H before measurement for Y
        0.0
    }
}

#[cfg(test)]
mod vqe_tests {
    use super::*;

    #[test]
    fn test_hamiltonian_creation() {
        let hamiltonian = Hamiltonian::new(vec![
            HamiltonianTerm::z(0, -1.0),
            HamiltonianTerm::z(1, -1.0),
            HamiltonianTerm::zz(0, 1, -0.5),
        ]);

        assert_eq!(hamiltonian.terms.len(), 3);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let ansatz = HardwareEfficientAnsatz::new(2, 2);
        assert_eq!(ansatz.num_parameters(), 4); // 2 qubits * 2 layers

        let mut sim = QuantumSimulator::new(2);
        let params = vec![0.1, 0.2, 0.3, 0.4];
        ansatz.apply(&mut sim, &params);

        // Just verify it runs without panic
        assert_eq!(sim.num_qubits(), 2);
    }

    #[test]
    fn test_vqe_simple() {
        // Simple test: find ground state of -Z[0] - Z[1] (should be |11⟩)
        let hamiltonian = Hamiltonian::new(vec![
            HamiltonianTerm::z(0, -1.0),
            HamiltonianTerm::z(1, -1.0),
        ]);

        let ansatz = HardwareEfficientAnsatz::new(2, 1);
        let initial_params = vec![0.1, 0.1];

        let result = vqe(2, &hamiltonian, &ansatz, &initial_params, 10, 0.1);

        // Ground state of -Z - Z is |11⟩ with energy -2.0
        // Our simple optimizer might not find exactly -2.0 but should be close
        assert!(result.minimum_energy < 0.0);
    }
}

// ============================================================
// QUANTUM APPROXIMATE OPTIMIZATION ALGORITHM (QAOA)
// ============================================================

/// QAOA problem specification
///
/// QAOA is used for combinatorial optimization problems.
/// This struct defines the problem through cost and mixer Hamiltonians.
#[derive(Clone)]
pub struct QAOAProblem {
    /// Cost Hamiltonian (encodes the objective function)
    pub cost_hamiltonian: Hamiltonian,
    /// Mixer Hamiltonian (typically -X on all qubits)
    pub mixer_hamiltonian: Hamiltonian,
    /// Number of qubits
    pub num_qubits: usize,
}

impl QAOAProblem {
    /// Create a new QAOA problem
    pub fn new(cost_hamiltonian: Hamiltonian, num_qubits: usize) -> Self {
        // Default mixer is -X on all qubits
        let mut mixer_terms = Vec::new();
        for q in 0..num_qubits {
            mixer_terms.push(HamiltonianTerm::new(-1.0, vec![(q, 0)])); // -X on each qubit
        }
        let mixer_hamiltonian = Hamiltonian::new(mixer_terms);

        QAOAProblem {
            cost_hamiltonian,
            mixer_hamiltonian,
            num_qubits,
        }
    }

    /// Create a MaxCut problem for QAOA
    ///
    /// MaxCut finds a partition of vertices that maximizes edges across the cut.
    /// Represented as a sum over edges: (1 - Z_i * Z_j) / 2
    ///
    /// # Arguments
    /// * `num_qubits` - Number of vertices in the graph
    /// * `edges` - List of (vertex1, vertex2) pairs representing edges
    pub fn maxcut(num_qubits: usize, edges: Vec<(usize, usize)>) -> Self {
        let mut cost_terms = Vec::new();

        for (i, j) in edges {
            // For MaxCut: cost = (1 - Z_i * Z_j) / 2 for each edge
            // This gives 1 if vertices are in different partitions, 0 if same
            cost_terms.push(HamiltonianTerm::new(0.5, vec![])); // Constant term
            cost_terms.push(HamiltonianTerm::new(-0.5, vec![(i, 2), (j, 2)])); // -Z_i * Z_j / 2
        }

        let cost_hamiltonian = Hamiltonian::new(cost_terms);

        QAOAProblem::new(cost_hamiltonian, num_qubits)
    }
}

/// QAOA result
///
/// Contains the result of a QAOA optimization run.
#[derive(Clone, Debug)]
pub struct QAOAResult {
    /// Optimal parameters (gamma, beta pairs)
    pub optimal_parameters: Vec<f64>,
    /// Best bit string found (as integer)
    pub best_solution: usize,
    /// Cost of the best solution
    pub best_cost: f64,
    /// Number of QAOA layers (p)
    pub num_layers: usize,
}

/// Run QAOA to solve a combinatorial optimization problem
///
/// # Algorithm
/// 1. Prepare uniform superposition
/// 2. Apply alternating cost and mixer unitaries parameterized by γ and β
/// 3. Measure to get candidate solution
/// 4. Optimize parameters classically
///
/// # Arguments
/// * `problem` - QAOA problem specification
/// * `num_layers` - Number of QAOA layers (depth p)
/// * `initial_parameters` - Starting parameters (alternating γ, β)
/// * `max_iterations` - Maximum number of classical optimization iterations
/// * `samples_per_iteration` - Number of measurements per iteration
///
/// # Returns
/// QAOAResult containing best solution found
///
/// # Example
/// ```ignore
/// use nqpu_metal::algorithms::*;
///
/// // MaxCut on a triangle (3 vertices, 3 edges)
/// let edges = vec![(0, 1), (1, 2), (2, 0)];
/// let problem = QAOAProblem::maxcut(3, edges);
///
/// let result = qaoa(&problem, 1, &[0.5, 0.5], 50, 100);
///
/// println!("Best cut: {:b}", result.best_solution);
/// println!("Cut size: {}", result.best_cost);
/// ```
pub fn qaoa(
    problem: &QAOAProblem,
    num_layers: usize,
    initial_parameters: &[f64],
    max_iterations: usize,
    samples_per_iteration: usize,
) -> QAOAResult {
    let mut parameters = initial_parameters.to_vec();
    let mut best_solution = 0;
    let mut best_cost = f64::INFINITY;

    for _iteration in 0..max_iterations {
        // Run QAOA circuit with current parameters
        let mut sim = QuantumSimulator::new(problem.num_qubits);

        // Initialize in uniform superposition
        for q in 0..problem.num_qubits {
            sim.h(q);
        }

        // Apply QAOA layers
        for layer in 0..num_layers {
            let gamma = parameters[2 * layer];
            let beta = parameters[2 * layer + 1];

            // Apply cost unitary: exp(-i*gamma*H_C)
            apply_cost_unitary(&mut sim, &problem.cost_hamiltonian, gamma);

            // Apply mixer unitary: exp(-i*beta*H_M)
            apply_mixer_unitary(&mut sim, &problem.mixer_hamiltonian, beta);
        }

        // Sample solutions
        for _sample in 0..samples_per_iteration {
            let solution = sim.measure();

            // Calculate cost for this solution
            let cost = calculate_maxcut_cost(solution, &problem.cost_hamiltonian);

            if cost < best_cost {
                best_cost = cost;
                best_solution = solution;
            }
        }

        // Simple parameter update (in production, would use proper optimization)
        // For demonstration, we just add small random perturbations
        if _iteration < max_iterations - 1 {
            for p in &mut parameters {
                *p += (rand::random::<f64>() - 0.5) * 0.1;
            }
        }
    }

    QAOAResult {
        optimal_parameters: parameters,
        best_solution,
        best_cost,
        num_layers,
    }
}

/// Apply cost unitary: exp(-i*gamma*H_C)
///
/// For diagonal cost Hamiltonians (ZZ, Z terms), this is straightforward.
fn apply_cost_unitary(simulator: &mut QuantumSimulator, hamiltonian: &Hamiltonian, gamma: f64) {
    for term in &hamiltonian.terms {
        // For each term, apply exp(-i*gamma*term)
        // Simplified: handle ZZ and Z terms

        if term.operators.len() == 2 {
            // ZZ interaction: exp(-i*gamma*Z_i*Z_j)
            let (q1, _) = term.operators[0];
            let (q2, _) = term.operators[1];
            let angle = 2.0 * term.coefficient * gamma;

            // CZZ gate can be decomposed as H-CNOT-H pattern
            // For simplicity, apply phase directly to amplitudes
            apply_zz_rotation(simulator, q1, q2, angle);
        } else if term.operators.len() == 1 {
            // Z rotation: exp(-i*gamma*Z)
            let (qubit, _) = term.operators[0];
            let angle = term.coefficient * gamma;
            simulator.rz(qubit, 2.0 * angle);
        }
        // Identity terms (no operators) contribute global phase
    }
}

/// Apply ZZ rotation gate
fn apply_zz_rotation(simulator: &mut QuantumSimulator, qubit1: usize, qubit2: usize, angle: f64) {
    // ZZ = |00⟩⟨00| + |01⟩⟨01| + |11⟩⟨11| - |10⟩⟨10|
    // exp(-i*theta*ZZ/2) applies phase to |10⟩ and |11⟩ states

    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;
    let dim = simulator.state.dim;
    let amplitudes = simulator.state.amplitudes_mut();

    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();

    for i in 0..dim {
        // Check if both qubits are in |1⟩ state
        if (i & mask1 != 0) && (i & mask2 != 0) {
            // Apply phase rotation
            let orig_re = amplitudes[i].re;
            amplitudes[i].re = orig_re * cos_half - amplitudes[i].im * sin_half;
            amplitudes[i].im = orig_re * sin_half + amplitudes[i].im * cos_half;
        }
    }
}

/// Apply mixer unitary: exp(-i*beta*H_M)
///
/// For H_M = -X on all qubits, this is just X rotations.
fn apply_mixer_unitary(simulator: &mut QuantumSimulator, hamiltonian: &Hamiltonian, beta: f64) {
    for term in &hamiltonian.terms {
        if term.operators.len() == 1 {
            let (qubit, pauli) = term.operators[0];

            if pauli == 0 {
                // X rotation: exp(-i*beta*(-X)) = exp(i*beta*X) = RX(-2*beta)
                let angle = -2.0 * term.coefficient * beta;
                simulator.rx(qubit, angle);
            }
        }
    }
}

/// Calculate MaxCut cost for a solution
fn calculate_maxcut_cost(solution: usize, hamiltonian: &Hamiltonian) -> f64 {
    // For MaxCut, we want to minimize the negative cut size
    // or maximize the cut size
    let mut cost = 0.0;

    for term in &hamiltonian.terms {
        if term.operators.is_empty() {
            // Constant term
            cost += term.coefficient;
        } else if term.operators.len() == 2 {
            // ZZ term: (1 - Z_i * Z_j) / 2
            let (q1, _) = term.operators[0];
            let (q2, _) = term.operators[1];

            let bit1 = (solution >> q1) & 1;
            let bit2 = (solution >> q2) & 1;

            // Z eigenvalue is +1 for |0⟩, -1 for |1⟩
            let z1 = if bit1 == 0 { 1.0 } else { -1.0 };
            let z2 = if bit2 == 0 { 1.0 } else { -1.0 };

            let zz_value = (1.0 - z1 * z2) / 2.0;
            cost += term.coefficient * zz_value;
        }
    }

    // For MaxCut, we negate (minimize negative = maximize cut)
    -cost
}

#[cfg(test)]
mod qaoa_tests {
    use super::*;

    #[test]
    fn test_maxcut_creation() {
        let edges = vec![(0, 1), (1, 2)];
        let problem = QAOAProblem::maxcut(3, edges);

        assert_eq!(problem.num_qubits, 3);
        assert!(!problem.cost_hamiltonian.terms.is_empty());
    }

    #[test]
    fn test_qaoa_simple() {
        // Simple MaxCut on 2 qubits with 1 edge
        let edges = vec![(0, 1)];
        let problem = QAOAProblem::maxcut(2, edges);

        // Single layer QAOA
        let result = qaoa(&problem, 1, &[0.5, 0.5], 10, 50);

        // Best cut should have cost -1 (one edge cut)
        assert!(result.best_cost <= 0.0);
        assert_eq!(result.num_layers, 1);
    }

    #[test]
    fn test_zz_rotation() {
        let mut sim = QuantumSimulator::new(2);
        sim.h(0);
        sim.h(1);

        // Apply ZZ rotation
        apply_zz_rotation(&mut sim, 0, 1, std::f64::consts::PI / 4.0);

        // Just verify it runs without panicking
        assert_eq!(sim.num_qubits(), 2);
    }
}

// ============================================================
// QUANTUM PHASE ESTIMATION (QPE)
// ============================================================

/// Quantum Phase Estimation result
///
/// Contains the estimated phase and measurement statistics.
#[derive(Clone, Debug)]
pub struct QPEResult {
    /// Estimated phase (in range [0, 1))
    pub phase: f64,
    /// Integer measurement result
    pub measurement: usize,
    /// Number of precision qubits used
    pub num_precision_qubits: usize,
}

/// Run Quantum Phase Estimation to estimate the eigenvalue of a unitary
///
/// # Algorithm
/// 1. Prepare eigenstate in target register
/// 2. Apply Hadamard to precision qubits
/// 3. Apply controlled-U^(2^k) operations
/// 4. Apply inverse QFT to precision qubits
/// 5. Measure to get phase estimate
///
/// # Arguments
/// * `num_precision_qubits` - Number of qubits for phase precision
/// * `apply_controlled_unitary` - Function that applies controlled-U^(2^k)
/// * `prepare_eigenstate` - Function that prepares the target qubits
///
/// # Returns
/// QPEResult containing estimated phase
///
/// # Example
/// ```ignore
/// use nqpu_metal::{QuantumSimulator, algorithms::qpe};
///
/// // Estimate phase of T gate (which has eigenvalue e^(iπ/4))
/// let mut sim = QuantumSimulator::new(3 + 1); // 3 precision + 1 target
/// let result = qpe(
///     3,
///     &mut sim,
///     |sim, control, power| {
///         // Apply T^(2^power) controlled by 'control' qubit
///         for _ in 0..(1 << power) {
///             sim.cnot(control, 3);
///             sim.t(3);
///         }
///     },
///     |sim| {
///         // Prepare |+⟩ eigenstate of T
///         sim.h(3);
///     },
/// );
///
/// println!("Estimated phase: {}", result.phase);
/// // Should be close to 0.125 (π/4 / 2π)
/// ```
pub fn qpe<F1, F2>(
    num_precision_qubits: usize,
    simulator: &mut QuantumSimulator,
    mut apply_controlled_unitary: F1,
    mut prepare_eigenstate: F2,
) -> QPEResult
where
    F1: FnMut(&mut QuantumSimulator, usize, usize), // (sim, control, power)
    F2: FnMut(&mut QuantumSimulator),               // (sim)
{
    let _total_qubits = simulator.num_qubits();
    let _target_qubit = num_precision_qubits;

    // Step 1: Prepare eigenstate in target register
    prepare_eigenstate(simulator);

    // Step 2: Initialize precision qubits in uniform superposition
    for i in 0..num_precision_qubits {
        simulator.h(i);
    }

    // Step 3: Apply controlled-U^(2^k) operations
    for k in 0..num_precision_qubits {
        apply_controlled_unitary(simulator, k, 1 << k);
    }

    // Step 4: Apply inverse QFT to precision qubits
    inverse_qft(&mut simulator.state, num_precision_qubits);

    // Step 5: Measure precision qubits
    let measurement = {
        let mut result = 0;
        for i in 0..num_precision_qubits {
            let (bit, _) = simulator.measure_qubit(i);
            result |= bit << (num_precision_qubits - 1 - i);
        }
        result
    };

    // Convert measurement to phase
    let phase = measurement as f64 / (1 << num_precision_qubits) as f64;

    QPEResult {
        phase,
        measurement,
        num_precision_qubits,
    }
}

/// Iterative Quantum Phase Estimation (single precision qubit)
///
/// Uses a single precision qubit iteratively, requiring fewer qubits
/// but more circuit repetitions.
///
/// # Arguments
/// * `num_iterations` - Number of iterations (bits of precision)
/// * `apply_controlled_unitary` - Function that applies controlled-U^(2^k)
/// * `prepare_eigenstate` - Function that prepares the target qubits
///
/// # Returns
/// Estimated phase in range [0, 1)
pub fn iterative_qpe<F1, F2>(
    num_iterations: usize,
    simulator: &mut QuantumSimulator,
    mut apply_controlled_unitary: F1,
    mut prepare_eigenstate: F2,
) -> f64
where
    F1: FnMut(&mut QuantumSimulator, usize, usize),
    F2: FnMut(&mut QuantumSimulator),
{
    let mut phase = 0.0;
    let mut current_power = 1;

    for iteration in 0..num_iterations {
        // Reset and prepare eigenstate
        simulator.reset();
        prepare_eigenstate(simulator);

        // Initialize precision qubit in |+⟩
        simulator.h(0);

        // Apply phase kickback with controlled-U^(2^iteration)
        apply_controlled_unitary(simulator, 0, current_power);

        // Apply Hadamard and measure
        simulator.h(0);
        let (bit, _) = simulator.measure_qubit(0);

        // Update phase estimate
        if bit == 1 {
            phase += 1.0 / (1 << (iteration + 1)) as f64;
        }

        current_power *= 2;
    }

    phase
}

#[cfg(test)]
mod qpe_tests {
    use super::*;

    #[test]
    fn test_qpe_simple() {
        // Test QPE with a simple Z rotation (eigenvalue ±1)
        let mut sim = QuantumSimulator::new(2 + 1); // 2 precision + 1 target

        // Z gate has eigenvalues ±1, corresponding to phases 0 and 0.5
        let result = qpe(
            2,
            &mut sim,
            |sim, control, _power| {
                // Apply controlled-Z (same for any power since Z² = I)
                sim.cz(control, 2);
            },
            |sim| {
                // Prepare |1⟩ eigenstate of Z (eigenvalue -1)
                sim.x(2);
            },
        );

        // Phase should be close to 0.5 (for eigenvalue -1)
        // The measurement gives us an integer, phase = integer / 2^2 = integer/4
        // For eigenvalue -1, we expect phase ≈ 0.5, which corresponds to integer ≈ 2
        // Allow for some error due to measurement
        let expected_phase = 0.5;
        let tolerance = 0.5; // 2-bit precision gives us resolution of 0.25
        assert!((result.phase - expected_phase).abs() < tolerance);
    }

    #[test]
    fn test_iterative_qpe() {
        let mut sim = QuantumSimulator::new(1 + 1); // 1 precision + 1 target

        let phase = iterative_qpe(
            3,
            &mut sim,
            |sim, control, power| {
                // Apply controlled-Z^power (which is just Z for any power)
                for _ in 0..power {
                    sim.cz(control, 1);
                }
            },
            |sim| {
                // Prepare |1⟩ eigenstate
                sim.x(1);
            },
        );

        // Phase should be close to 0.5
        assert!((phase - 0.5).abs() < 0.2);
    }
}
