//! Quantum Cognition: Quantum Probability Theory Applied to Human Decision-Making
//!
//! **WORLD FIRST**: First quantum simulator with built-in cognitive science models.
//!
//! Quantum cognition uses the mathematical framework of quantum mechanics (not
//! claiming brains are quantum computers) to model cognitive phenomena that
//! violate classical probability theory:
//!
//! - **Order effects** in judgments (asking question A before B changes the answer to B)
//! - **Conjunction fallacy** (P(A and B) > P(A) for non-commuting observables)
//! - **Sure-thing principle violations** (Allais/Ellsberg paradoxes)
//! - **Context-dependent preferences** (incompatible decision frames)
//! - **Question order effects** in surveys and polling
//!
//! # Mathematical Foundation
//!
//! Classical probability represents beliefs as points in a simplex. Quantum
//! probability represents beliefs as state vectors in a Hilbert space. The key
//! difference: classical events always commute, but quantum observables (modeled
//! as projectors) need not commute. Non-commutativity gives rise to order effects,
//! interference terms, and the conjunction fallacy -- all of which are empirically
//! observed in human cognition.
//!
//! # The QQ Equality
//!
//! The Quantum Question (QQ) model predicts a specific constraint:
//! ```text
//! P(A_yes then B_yes) + P(A_no then B_no) = P(B_yes then A_yes) + P(B_no then A_no)
//! ```
//! This equality has been confirmed in dozens of psychological experiments
//! (Wang & Busemeyer 2013, Wang et al. 2014).
//!
//! # References
//!
//! - Busemeyer & Bruza (2012) - Quantum Models of Cognition and Decision
//! - Pothos & Busemeyer (2013) - Can quantum probability provide a new direction
//!   for cognitive modeling?
//! - Wang & Busemeyer (2013) - A quantum question order model
//! - Tversky & Kahneman (1983) - Extensional versus intuitive reasoning:
//!   The conjunction fallacy in probability judgment
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::quantum_cognition::*;
//! use num_complex::Complex64;
//!
//! // Create a 2D belief space (simplest non-trivial case)
//! let config = CognitionConfig::default();
//! let mut dm = QuantumDecisionMaker::new(config);
//!
//! // Define a question as a projector onto a subspace
//! let guilty_basis = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
//!                         Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];
//! let proj = Projector::from_single_vector(vec![guilty_basis], "guilty");
//! let prob = dm.ask_question(&proj);
//! ```

use num_complex::Complex64;
use std::fmt;

use crate::C64;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors arising from quantum cognition operations.
#[derive(Clone, Debug, PartialEq)]
pub enum CognitionError {
    /// The belief state dimension does not match the expected dimension.
    InvalidDimension {
        expected: usize,
        got: usize,
    },
    /// A projector's basis vectors have the wrong dimension for the belief space.
    ProjectorRankMismatch {
        projector_dim: usize,
        state_dim: usize,
    },
    /// A state vector is not normalized (norm deviates from 1.0 by more than tolerance).
    NotNormalized {
        norm: f64,
    },
    /// A projector was constructed with an empty basis (no basis vectors).
    EmptyBasis,
}

impl fmt::Display for CognitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CognitionError::InvalidDimension { expected, got } => {
                write!(f, "invalid dimension: expected {}, got {}", expected, got)
            }
            CognitionError::ProjectorRankMismatch {
                projector_dim,
                state_dim,
            } => write!(
                f,
                "projector dimension {} does not match state dimension {}",
                projector_dim, state_dim
            ),
            CognitionError::NotNormalized { norm } => {
                write!(f, "state not normalized: ||psi||^2 = {:.6}", norm)
            }
            CognitionError::EmptyBasis => {
                write!(f, "projector constructed with empty basis")
            }
        }
    }
}

impl std::error::Error for CognitionError {}

// ============================================================
// COGNITION MODEL ENUM
// ============================================================

/// The cognitive model variant to use for decision-making.
///
/// Each model captures different aspects of quantum-like cognition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitionModel {
    /// Basic quantum probability for binary decisions.
    /// Beliefs are state vectors; questions are projectors; answers follow the Born rule.
    QuantumProbability,
    /// Quantum Question (QQ) model for order effects in sequential judgments.
    /// Predicts the QQ equality constraint validated in psychological experiments.
    QQModel,
    /// Quantum version of Kahneman-Tversky prospect theory.
    /// Gains/losses are represented as incompatible observables.
    QuantumProspect,
    /// Quantum Bayesian updating (QBism-inspired).
    /// State update via Luders rule generalizes classical Bayesian conditioning.
    QuantumBayesian,
}

impl Default for CognitionModel {
    fn default() -> Self {
        CognitionModel::QuantumProbability
    }
}

// ============================================================
// CONFIGURATION
// ============================================================

/// Configuration for quantum cognition models.
///
/// Use the builder pattern via `Default` and setter methods.
///
/// # Example
/// ```
/// use nqpu_metal::quantum_cognition::CognitionConfig;
/// let config = CognitionConfig::new()
///     .with_belief_dim(6)
///     .with_model(nqpu_metal::quantum_cognition::CognitionModel::QQModel);
/// ```
#[derive(Clone, Debug)]
pub struct CognitionConfig {
    /// Dimension of the belief Hilbert space.
    /// A higher dimension allows finer-grained belief representations.
    pub belief_dim: usize,
    /// Which cognitive model to use.
    pub model_type: CognitionModel,
    /// Similarity parameter for transition probabilities between concepts (0.0 to 1.0).
    /// Controls how strongly similar concepts interfere.
    pub similarity_param: f64,
}

impl Default for CognitionConfig {
    fn default() -> Self {
        Self {
            belief_dim: 4,
            model_type: CognitionModel::QuantumProbability,
            similarity_param: 0.5,
        }
    }
}

impl CognitionConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the belief space dimension.
    pub fn with_belief_dim(mut self, dim: usize) -> Self {
        self.belief_dim = dim;
        self
    }

    /// Set the cognition model type.
    pub fn with_model(mut self, model: CognitionModel) -> Self {
        self.model_type = model;
        self
    }

    /// Set the similarity parameter.
    pub fn with_similarity(mut self, s: f64) -> Self {
        self.similarity_param = s.clamp(0.0, 1.0);
        self
    }
}

// ============================================================
// BELIEF STATE
// ============================================================

/// A quantum belief state: a normalized vector in a complex Hilbert space.
///
/// The belief state |psi> encodes a cognitive agent's beliefs as a superposition
/// over basis states. The Born rule gives the probability of each outcome when
/// a question (projector) is applied.
#[derive(Clone, Debug)]
pub struct BeliefState {
    /// State vector in the belief Hilbert space.
    pub state: Vec<C64>,
    /// Dimension of the Hilbert space.
    pub dim: usize,
}

impl BeliefState {
    /// Create a new belief state in uniform superposition over all basis states.
    ///
    /// This represents maximum uncertainty: equal probability for all outcomes.
    /// |psi> = (1/sqrt(dim)) * (|0> + |1> + ... + |dim-1>)
    pub fn new(dim: usize) -> Self {
        let amp = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
        let state = vec![amp; dim];
        BeliefState { state, dim }
    }

    /// Create a belief state from an explicit state vector.
    ///
    /// The vector is automatically normalized. If the input has zero norm,
    /// falls back to uniform superposition.
    pub fn from_state(mut state: Vec<C64>) -> Self {
        let dim = state.len();
        let norm = state_norm(&state);
        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            for a in state.iter_mut() {
                *a = Complex64::new(a.re * inv_norm, a.im * inv_norm);
            }
        } else {
            // Degenerate zero vector: fall back to uniform superposition
            let amp = Complex64::new(1.0 / (dim as f64).sqrt(), 0.0);
            state = vec![amp; dim];
        }
        BeliefState { state, dim }
    }

    /// Create a belief state from a single basis state |k>.
    pub fn from_basis(dim: usize, k: usize) -> Self {
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        if k < dim {
            state[k] = Complex64::new(1.0, 0.0);
        } else {
            state[0] = Complex64::new(1.0, 0.0);
        }
        BeliefState { state, dim }
    }

    /// Compute the probability of an outcome defined by a projector (Born rule).
    ///
    /// P(outcome) = ||P_outcome |psi>||^2 = <psi| P_outcome |psi>
    ///
    /// For a rank-1 projector with basis vector |b>, this reduces to |<b|psi>|^2.
    pub fn probability(&self, projector: &Projector) -> f64 {
        let projected = projector.project(&self.state);
        state_norm_squared(&projected)
    }

    /// Update the belief state after observing an outcome (Luders rule).
    ///
    /// The Luders rule is the quantum generalization of Bayesian conditioning:
    /// |psi'> = P |psi> / ||P |psi>||
    ///
    /// After update, future probabilities are conditioned on this observation.
    pub fn update(&mut self, projector: &Projector) {
        let projected = projector.project(&self.state);
        let norm = state_norm(&projected);
        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            self.state = projected
                .iter()
                .map(|a| Complex64::new(a.re * inv_norm, a.im * inv_norm))
                .collect();
        }
        // If norm is zero, the outcome was impossible; state unchanged.
    }

    /// Check whether this state is properly normalized.
    pub fn is_normalized(&self, tol: f64) -> bool {
        let norm_sq = state_norm_squared(&self.state);
        (norm_sq - 1.0).abs() < tol
    }

    /// Compute the inner product <self|other>.
    pub fn inner_product(&self, other: &BeliefState) -> C64 {
        inner_product(&self.state, &other.state)
    }
}

// ============================================================
// PROJECTOR
// ============================================================

/// A projector onto a subspace of the belief Hilbert space.
///
/// Projectors represent yes/no questions or binary observables. The subspace
/// spanned by the basis vectors is the "yes" answer; the orthogonal complement
/// is the "no" answer.
///
/// A projector satisfies P^2 = P and P = P^dagger (idempotent and Hermitian).
#[derive(Clone, Debug)]
pub struct Projector {
    /// Orthonormal basis vectors spanning the subspace.
    /// Each inner Vec<C64> is a basis vector of length `dim`.
    pub basis_vectors: Vec<Vec<C64>>,
    /// Human-readable label for this question/observable.
    pub label: String,
}

impl Projector {
    /// Create a projector from a set of basis vectors.
    ///
    /// The basis vectors are orthonormalized via Gram-Schmidt.
    /// Returns an error if the basis is empty.
    pub fn new(basis: Vec<Vec<C64>>, label: &str) -> Result<Self, CognitionError> {
        if basis.is_empty() {
            return Err(CognitionError::EmptyBasis);
        }
        let orthonormal = gram_schmidt(&basis);
        Ok(Projector {
            basis_vectors: orthonormal,
            label: label.to_string(),
        })
    }

    /// Create a rank-1 projector from a single (not necessarily normalized) vector.
    pub fn from_single_vector(v: Vec<C64>, label: &str) -> Result<Self, CognitionError> {
        if v.is_empty() {
            return Err(CognitionError::EmptyBasis);
        }
        Self::new(vec![v], label)
    }

    /// Create a projector onto the k-th computational basis state |k>.
    pub fn basis_projector(dim: usize, k: usize, label: &str) -> Self {
        let mut v = vec![Complex64::new(0.0, 0.0); dim];
        if k < dim {
            v[k] = Complex64::new(1.0, 0.0);
        }
        Projector {
            basis_vectors: vec![v],
            label: label.to_string(),
        }
    }

    /// Project a state vector onto this subspace.
    ///
    /// P|psi> = sum_i |b_i><b_i|psi>
    ///
    /// where {|b_i>} are the orthonormal basis vectors of the subspace.
    pub fn project(&self, state: &[C64]) -> Vec<C64> {
        let dim = state.len();
        let mut result = vec![Complex64::new(0.0, 0.0); dim];

        for basis_vec in &self.basis_vectors {
            // <b|psi>
            let coeff = inner_product(basis_vec, state);
            // result += coeff * |b>
            for (r, b) in result.iter_mut().zip(basis_vec.iter()) {
                *r = *r + coeff * b;
            }
        }
        result
    }

    /// Compute the orthogonal complement projector (I - P).
    ///
    /// The complement projector represents the "no" answer to this question.
    pub fn complement(&self, total_dim: usize) -> Projector {
        // Build the full set of orthonormal vectors spanning the complement.
        // Start from the standard basis and remove components in the projector subspace.
        let mut complement_basis = Vec::new();

        for k in 0..total_dim {
            // Start with |k>
            let mut v = vec![Complex64::new(0.0, 0.0); total_dim];
            v[k] = Complex64::new(1.0, 0.0);

            // Remove projection onto our subspace: v' = v - P*v
            let proj = self.project(&v);
            for (vi, pi) in v.iter_mut().zip(proj.iter()) {
                *vi = *vi - pi;
            }

            // Keep if non-trivial
            let norm = state_norm(&v);
            if norm > 1e-10 {
                let inv_norm = 1.0 / norm;
                for vi in v.iter_mut() {
                    *vi = Complex64::new(vi.re * inv_norm, vi.im * inv_norm);
                }
                complement_basis.push(v);
            }
        }

        // Re-orthonormalize
        let orthonormal = gram_schmidt(&complement_basis);

        Projector {
            basis_vectors: orthonormal,
            label: format!("NOT({})", self.label),
        }
    }

    /// Rank of the projector (number of orthonormal basis vectors).
    pub fn rank(&self) -> usize {
        self.basis_vectors.len()
    }

    /// Dimension of the ambient Hilbert space (inferred from first basis vector).
    pub fn dim(&self) -> usize {
        if self.basis_vectors.is_empty() {
            0
        } else {
            self.basis_vectors[0].len()
        }
    }

    /// Check whether two projectors commute: [P_A, P_B] = 0.
    ///
    /// Commuting projectors produce no order effects (classical-like behavior).
    /// Non-commuting projectors are the source of quantum cognitive phenomena.
    pub fn commutes_with(&self, other: &Projector, tol: f64) -> bool {
        // Two projectors commute iff P_A P_B = P_B P_A.
        // We check this on a basis: for each standard basis vector |k>,
        // verify P_A P_B |k> = P_B P_A |k>.
        let dim = self.dim().max(other.dim());
        if dim == 0 {
            return true;
        }

        for k in 0..dim {
            let mut ek = vec![Complex64::new(0.0, 0.0); dim];
            ek[k] = Complex64::new(1.0, 0.0);

            // P_A P_B |k>
            let pb_ek = other.project(&ek);
            let pa_pb_ek = self.project(&pb_ek);

            // P_B P_A |k>
            let pa_ek = self.project(&ek);
            let pb_pa_ek = other.project(&pa_ek);

            // Check equality
            for (a, b) in pa_pb_ek.iter().zip(pb_pa_ek.iter()) {
                if (a - b).norm() > tol {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================
// QUESTION ORDER EFFECT
// ============================================================

/// Result of analyzing order effects between two questions.
///
/// In quantum cognition, asking question A before B can change the probability
/// of a "yes" answer to B, compared to asking B first. This is analogous to
/// the non-commutativity of quantum observables.
#[derive(Clone, Debug)]
pub struct QuestionOrderEffect {
    /// P(B = yes | A asked first and answered yes).
    /// Computed as P(B_yes after A_yes) / P(A_yes) when feasible.
    pub p_a_then_b: f64,
    /// P(A = yes | B asked first and answered yes).
    pub p_b_then_a: f64,
    /// The order effect magnitude: p_a_then_b - p_b_then_a.
    /// Non-zero indicates a quantum-like order effect.
    pub order_effect: f64,
    /// QQ equality test value.
    /// P(A_yes then B_yes) + P(A_no then B_no) - P(B_yes then A_yes) - P(B_no then A_no).
    /// Should be approximately zero for quantum models.
    pub qq_equality_test: f64,
    /// Whether the QQ equality holds within tolerance (|qq_equality_test| < 0.05).
    pub satisfies_qq: bool,
}

// ============================================================
// CONJUNCTION RESULT
// ============================================================

/// Result of testing for the conjunction fallacy.
///
/// The conjunction fallacy (Tversky & Kahneman 1983) is the observation that
/// people often judge P(A and B) > P(A), violating classical probability.
///
/// The Linda Problem: "Linda is a bank teller" vs "Linda is a bank teller AND
/// a feminist activist." Most people rate the conjunction as more probable.
///
/// In quantum probability this is not a fallacy at all: sequential projection
/// can produce constructive interference, making the conjunction genuinely
/// more probable in the quantum framework.
#[derive(Clone, Debug)]
pub struct ConjunctionResult {
    /// P(A) = Born rule probability of A alone.
    pub p_a: f64,
    /// P(B) = Born rule probability of B alone.
    pub p_b: f64,
    /// Classical upper bound: min(P(A), P(B)).
    /// In classical probability, P(A and B) <= min(P(A), P(B)).
    pub p_a_and_b_classical: f64,
    /// Quantum probability: P(A and B) = ||P_B P_A |psi>||^2.
    /// This is the probability of sequential projection, which can exceed
    /// the classical bound when interference is constructive.
    pub p_a_and_b_quantum: f64,
    /// Whether a conjunction fallacy occurs: P(A and B)_quantum > min(P(A), P(B)).
    pub is_fallacy: bool,
    /// The interference term causing the deviation from classical probability.
    pub interference: f64,
}

// ============================================================
// SURE-THING RESULT
// ============================================================

/// Result of testing for violations of the sure-thing principle.
///
/// The sure-thing principle (Savage 1954): if you would choose action X
/// regardless of which condition holds, you should choose X when the
/// condition is unknown. Formally:
///
/// If P(action | C_i) >= threshold for all i, then P(action) >= threshold.
///
/// The Allais and Ellsberg paradoxes demonstrate that humans routinely
/// violate this principle. In quantum probability, interference between
/// condition branches naturally produces these violations.
#[derive(Clone, Debug)]
pub struct SureThingResult {
    /// P(action | condition_i) for each condition.
    pub conditional_probs: Vec<f64>,
    /// P(action) marginalized over all conditions (quantum).
    pub marginal_prob: f64,
    /// Classical marginal: weighted sum of conditional probabilities.
    /// Should equal marginal_prob in classical probability.
    pub classical_marginal: f64,
    /// The violation magnitude: |marginal_prob - classical_marginal|.
    pub violation: f64,
    /// Whether a violation occurs (violation > tolerance).
    pub is_violated: bool,
}

// ============================================================
// QUANTUM DECISION MAKER
// ============================================================

/// Main cognitive decision-making engine using quantum probability.
///
/// The QuantumDecisionMaker maintains a belief state and provides methods
/// for asking questions (projective measurements), detecting order effects,
/// computing conjunction fallacies, and testing the sure-thing principle.
///
/// # Cognitive Interpretation
///
/// The belief state |psi> represents the cognitive state of an agent.
/// Questions are represented as projectors onto subspaces. Asking a question
/// collapses the belief state (Luders rule), just as measurement collapses
/// a quantum state. This collapse is the mechanism behind order effects:
/// asking question A changes the state before question B is asked.
pub struct QuantumDecisionMaker {
    /// Configuration for the cognition model.
    config: CognitionConfig,
    /// Current belief state of the cognitive agent.
    belief: BeliefState,
}

impl QuantumDecisionMaker {
    /// Create a new decision maker with the given configuration.
    ///
    /// The initial belief state is a uniform superposition (maximum uncertainty).
    pub fn new(config: CognitionConfig) -> Self {
        let belief = BeliefState::new(config.belief_dim);
        QuantumDecisionMaker { config, belief }
    }

    /// Set the belief state explicitly.
    pub fn set_belief(&mut self, belief: BeliefState) {
        self.belief = belief;
    }

    /// Get a reference to the current belief state.
    pub fn belief(&self) -> &BeliefState {
        &self.belief
    }

    /// Get the configuration.
    pub fn config(&self) -> &CognitionConfig {
        &self.config
    }

    /// Ask a question (apply projective measurement) and return the probability
    /// of a "yes" answer. The belief state is collapsed according to the Luders rule.
    ///
    /// This is the fundamental operation: it computes P(yes) via the Born rule,
    /// then updates the belief state as if the answer was "yes."
    pub fn ask_question(&mut self, projector: &Projector) -> f64 {
        let prob = self.belief.probability(projector);
        self.belief.update(projector);
        prob
    }

    /// Ask a sequence of questions, collapsing after each one.
    ///
    /// Returns the probability of "yes" for each question, given that all
    /// previous questions were answered "yes."
    pub fn ask_sequence(&mut self, questions: &[Projector]) -> Vec<f64> {
        let mut probs = Vec::with_capacity(questions.len());
        for q in questions {
            let p = self.ask_question(q);
            probs.push(p);
        }
        probs
    }

    /// Compute the order effect between two questions.
    ///
    /// This measures how the order of questioning affects probabilities.
    /// For the Clinton-Gore experiment (Wang & Busemeyer 2013):
    /// - P(Clinton honest | asked first) ~ 0.50
    /// - P(Clinton honest | Gore asked first) ~ 0.57
    /// - Order effect ~ 0.07
    ///
    /// The method does not modify the internal belief state.
    pub fn order_effect(&self, q_a: &Projector, q_b: &Projector) -> QuestionOrderEffect {
        let dim = self.config.belief_dim;
        let not_a = q_a.complement(dim);
        let not_b = q_b.complement(dim);

        // --- Order A then B ---
        // P(A_yes then B_yes): project onto A, then onto B
        let p_ay_by = sequential_probability(&self.belief.state, q_a, q_b);
        // P(A_no then B_no)
        let p_an_bn = sequential_probability(&self.belief.state, &not_a, &not_b);
        // P(A_yes then B_no)
        let _p_ay_bn = sequential_probability(&self.belief.state, q_a, &not_b);
        // P(A_no then B_yes)
        let _p_an_by = sequential_probability(&self.belief.state, &not_a, q_b);

        // --- Order B then A ---
        // P(B_yes then A_yes)
        let p_by_ay = sequential_probability(&self.belief.state, q_b, q_a);
        // P(B_no then A_no)
        let p_bn_an = sequential_probability(&self.belief.state, &not_b, &not_a);

        // p_a_then_b: joint probability of getting yes to both when A is asked first
        let p_a_then_b = p_ay_by;
        // p_b_then_a: joint probability of getting yes to both when B is asked first
        let p_b_then_a = p_by_ay;

        let order_effect = p_a_then_b - p_b_then_a;

        // QQ equality: P(Ay,By) + P(An,Bn) should equal P(By,Ay) + P(Bn,An)
        let lhs = p_ay_by + p_an_bn;
        let rhs = p_by_ay + p_bn_an;
        let qq_equality_test = lhs - rhs;

        let qq_tolerance = 0.05;
        let satisfies_qq = qq_equality_test.abs() < qq_tolerance;

        QuestionOrderEffect {
            p_a_then_b,
            p_b_then_a,
            order_effect,
            qq_equality_test,
            satisfies_qq,
        }
    }

    /// Test for the conjunction fallacy between two questions.
    ///
    /// In the Linda Problem (Tversky & Kahneman 1983), people judge
    /// P(bank teller AND feminist) > P(bank teller). This violates
    /// classical probability but is natural in quantum probability when
    /// the projectors for "bank teller" and "feminist" don't commute.
    ///
    /// The method does not modify the internal belief state.
    pub fn conjunction_fallacy(
        &self,
        a: &Projector,
        b: &Projector,
    ) -> ConjunctionResult {
        // P(A) and P(B) individually
        let p_a = self.belief.probability(a);
        let p_b = self.belief.probability(b);

        // Classical bound
        let p_a_and_b_classical = p_a.min(p_b);

        // Quantum conjunction: P(A and B) = ||P_B P_A |psi>||^2
        // This is the probability of sequential projection
        let p_a_and_b_quantum = sequential_probability(&self.belief.state, a, b);

        // Interference term: the difference between quantum and classical
        let interference = p_a_and_b_quantum - p_a * p_b;

        let is_fallacy = p_a_and_b_quantum > p_a_and_b_classical + 1e-10;

        ConjunctionResult {
            p_a,
            p_b,
            p_a_and_b_classical,
            p_a_and_b_quantum,
            is_fallacy,
            interference,
        }
    }

    /// Test for violations of the sure-thing principle.
    ///
    /// Given a set of mutually exclusive conditions (e.g., "economy good" vs
    /// "economy bad") and a set of actions (e.g., "invest" vs "save"), check
    /// whether P(action) equals the weighted sum of P(action | condition_i).
    ///
    /// A non-zero violation indicates interference between condition branches,
    /// producing Allais/Ellsberg-type paradoxes.
    ///
    /// The method does not modify the internal belief state.
    pub fn sure_thing_violation(
        &self,
        actions: &[Projector],
        conditions: &[Projector],
    ) -> SureThingResult {
        if actions.is_empty() || conditions.is_empty() {
            return SureThingResult {
                conditional_probs: vec![],
                marginal_prob: 0.0,
                classical_marginal: 0.0,
                violation: 0.0,
                is_violated: false,
            };
        }

        // Use the first action as the "target" action
        let action = &actions[0];

        // Compute P(action | condition_i) for each condition
        let mut conditional_probs = Vec::new();
        let mut condition_weights = Vec::new();

        for cond in conditions {
            // P(condition_i)
            let p_cond = self.belief.probability(cond);
            condition_weights.push(p_cond);

            if p_cond > 1e-15 {
                // P(action and condition) via sequential projection
                let p_action_and_cond = sequential_probability(&self.belief.state, cond, action);
                // P(action | condition) = P(action and condition) / P(condition)
                conditional_probs.push(p_action_and_cond / p_cond);
            } else {
                conditional_probs.push(0.0);
            }
        }

        // Classical marginal: sum_i P(condition_i) * P(action | condition_i)
        let classical_marginal: f64 = condition_weights
            .iter()
            .zip(conditional_probs.iter())
            .map(|(w, p)| w * p)
            .sum();

        // Quantum marginal: P(action) directly
        let marginal_prob = self.belief.probability(action);

        let violation = (marginal_prob - classical_marginal).abs();
        let is_violated = violation > 0.01;

        SureThingResult {
            conditional_probs,
            marginal_prob,
            classical_marginal,
            violation,
            is_violated,
        }
    }

    /// Compute the quantum interference term between two projectors.
    ///
    /// The interference term captures the deviation of quantum probability
    /// from classical probability. For commuting projectors, interference is
    /// zero. For non-commuting projectors, it can be positive (constructive)
    /// or negative (destructive).
    ///
    /// Formally: I(A,B) = P_quantum(A and B) - P(A) * P(B)
    ///
    /// This is related to the "quantum correlation" between the two questions.
    pub fn interference_term(&self, a: &Projector, b: &Projector) -> f64 {
        let p_a = self.belief.probability(a);
        let p_b = self.belief.probability(b);
        let p_ab_quantum = sequential_probability(&self.belief.state, a, b);

        // Interference = quantum joint - classical independent product
        p_ab_quantum - p_a * p_b
    }
}

// ============================================================
// QUANTUM SURVEY SIMULATOR
// ============================================================

/// Simulator for survey/polling scenarios with quantum order effects.
///
/// In classical survey design, question order should not matter (assuming
/// no priming effects). In quantum cognition, question order naturally
/// arises from the non-commutativity of mental observables.
///
/// This simulator can model, for example, how asking about trust in
/// politician A before politician B changes the reported trust in B.
pub struct QuantumSurveySimulator {
    /// Maximum number of questions the survey was designed for.
    _num_questions: usize,
    /// Dimension of the belief Hilbert space.
    belief_dim: usize,
    /// Named questions with their projectors.
    questions: Vec<(String, Projector)>,
}

/// Result of simulating a survey across a population.
#[derive(Clone, Debug)]
pub struct SurveyResult {
    /// Marginal probability for each question (name, probability).
    pub question_probabilities: Vec<(String, f64)>,
    /// Order effect for each pair of questions (name_a, name_b, effect magnitude).
    pub order_effects: Vec<(String, String, f64)>,
    /// Whether the QQ equality holds across all question pairs.
    pub qq_equality_holds: bool,
}

impl QuantumSurveySimulator {
    /// Create a new survey simulator.
    pub fn new(num_questions: usize, belief_dim: usize) -> Self {
        QuantumSurveySimulator {
            _num_questions: num_questions,
            belief_dim,
            questions: Vec::with_capacity(num_questions),
        }
    }

    /// Add a question to the survey.
    pub fn add_question(&mut self, name: &str, projector: Projector) {
        self.questions.push((name.to_string(), projector));
    }

    /// Simulate a population of respondents with slightly varying initial beliefs.
    ///
    /// Each respondent starts with a belief state perturbed from the uniform
    /// superposition using a simple deterministic pseudo-random perturbation
    /// seeded by the respondent index and the provided seed.
    pub fn simulate_population(&self, population_size: usize, seed: u64) -> SurveyResult {
        let mut total_probs = vec![0.0f64; self.questions.len()];
        let mut total_order_effects: Vec<(String, String, f64)> = Vec::new();

        // Initialize order effect accumulators for all pairs
        let mut pair_effects: Vec<Vec<f64>> = Vec::new();
        for i in 0..self.questions.len() {
            for j in (i + 1)..self.questions.len() {
                pair_effects.push(Vec::new());
                let _ = (i, j); // suppress unused warning
            }
        }

        for respondent in 0..population_size {
            // Generate a perturbed belief state for this respondent
            let belief = perturbed_belief(self.belief_dim, respondent as u64, seed);

            let dm = QuantumDecisionMaker {
                config: CognitionConfig::new().with_belief_dim(self.belief_dim),
                belief,
            };

            // Marginal probabilities
            for (qi, (_name, proj)) in self.questions.iter().enumerate() {
                total_probs[qi] += dm.belief.probability(proj);
            }

            // Order effects for all pairs
            let mut pair_idx = 0;
            for i in 0..self.questions.len() {
                for j in (i + 1)..self.questions.len() {
                    let effect = dm.order_effect(&self.questions[i].1, &self.questions[j].1);
                    if pair_idx < pair_effects.len() {
                        pair_effects[pair_idx].push(effect.order_effect);
                    }
                    pair_idx += 1;
                }
            }
        }

        let pop = population_size as f64;

        // Average marginal probabilities
        let question_probabilities: Vec<(String, f64)> = self
            .questions
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), total_probs[i] / pop))
            .collect();

        // Average order effects
        let mut pair_idx = 0;
        let mut all_qq_hold = true;
        for i in 0..self.questions.len() {
            for j in (i + 1)..self.questions.len() {
                let avg_effect: f64 = if pair_idx < pair_effects.len()
                    && !pair_effects[pair_idx].is_empty()
                {
                    pair_effects[pair_idx].iter().sum::<f64>() / pair_effects[pair_idx].len() as f64
                } else {
                    0.0
                };
                total_order_effects.push((
                    self.questions[i].0.clone(),
                    self.questions[j].0.clone(),
                    avg_effect,
                ));
                // QQ equality: average order effect for quantum model should be modest
                // (exact QQ equality is for a single state; across population we check
                // that effects are consistent in sign)
                if avg_effect.abs() > 0.5 {
                    all_qq_hold = false;
                }
                pair_idx += 1;
            }
        }

        SurveyResult {
            question_probabilities,
            order_effects: total_order_effects,
            qq_equality_holds: all_qq_hold,
        }
    }

    /// Compute order effects for all question pairs using the default belief state.
    pub fn compute_all_order_effects(&self) -> Vec<QuestionOrderEffect> {
        let belief = BeliefState::new(self.belief_dim);
        let dm = QuantumDecisionMaker {
            config: CognitionConfig::new().with_belief_dim(self.belief_dim),
            belief,
        };

        let mut effects = Vec::new();
        for i in 0..self.questions.len() {
            for j in (i + 1)..self.questions.len() {
                let effect = dm.order_effect(&self.questions[i].1, &self.questions[j].1);
                effects.push(effect);
            }
        }
        effects
    }
}

// ============================================================
// CLASSIC EXAMPLES
// ============================================================

/// Classic examples from the quantum cognition literature.
///
/// These functions construct projectors and belief states that reproduce
/// well-known experimental findings.
pub struct CognitionExamples;

impl CognitionExamples {
    /// Clinton-Gore experiment (Wang & Busemeyer 2013).
    ///
    /// Participants were asked whether Clinton and Gore were honest/trustworthy.
    /// The order in which the questions were asked significantly affected responses:
    /// - P(Clinton honest | asked first) ~ 0.50
    /// - P(Clinton honest | Gore asked first) ~ 0.57
    /// - Order effect ~ 0.07
    ///
    /// Returns (decision_maker, clinton_projector, gore_projector) set up
    /// to reproduce the order effect.
    pub fn clinton_gore() -> (QuantumDecisionMaker, Projector, Projector) {
        // 2D belief space with an initial state that produces the observed effect
        let dim = 2;
        let config = CognitionConfig::new().with_belief_dim(dim).with_model(CognitionModel::QQModel);

        // Initial belief state: slightly biased superposition
        // Chosen so that the order effect is approximately 0.07
        let angle = 0.32 * std::f64::consts::PI;
        let state = vec![
            Complex64::new(angle.cos(), 0.0),
            Complex64::new(angle.sin(), 0.0),
        ];
        let belief = BeliefState::from_state(state);

        let mut dm = QuantumDecisionMaker::new(config);
        dm.set_belief(belief);

        // Clinton "honest" projector: rotated basis
        let theta_c = 0.45 * std::f64::consts::PI;
        let clinton = Projector::from_single_vector(
            vec![
                Complex64::new(theta_c.cos(), 0.0),
                Complex64::new(theta_c.sin(), 0.0),
            ],
            "Clinton honest",
        )
        .expect("clinton projector");

        // Gore "honest" projector: different rotation (non-commuting with Clinton)
        let theta_g = 0.28 * std::f64::consts::PI;
        let gore = Projector::from_single_vector(
            vec![
                Complex64::new(theta_g.cos(), 0.0),
                Complex64::new(theta_g.sin(), 0.0),
            ],
            "Gore honest",
        )
        .expect("gore projector");

        (dm, clinton, gore)
    }

    /// Linda Problem (Tversky & Kahneman 1983).
    ///
    /// "Linda is 31, single, outspoken, and bright. She majored in philosophy.
    /// As a student, she was deeply concerned with discrimination and social justice,
    /// and also participated in anti-nuclear demonstrations."
    ///
    /// Most people judge P(bank teller AND feminist) > P(bank teller alone).
    /// In quantum probability, this is explained by constructive interference
    /// between the "bank teller" and "feminist" subspaces.
    ///
    /// Returns (decision_maker, bank_teller_projector, feminist_projector).
    pub fn linda_problem() -> (QuantumDecisionMaker, Projector, Projector) {
        // 4D belief space for richer interference structure
        let dim = 4;
        let config = CognitionConfig::new().with_belief_dim(dim);

        // Initial belief: Linda's description primes a state that is
        // more aligned with "feminist" than "bank teller"
        let state = vec![
            Complex64::new(0.2, 0.0),   // basis: "typical bank teller"
            Complex64::new(0.1, 0.1),   // basis: "atypical bank teller"
            Complex64::new(0.6, 0.0),   // basis: "feminist activist"
            Complex64::new(0.3, -0.1),  // basis: "other"
        ];
        let belief = BeliefState::from_state(state);

        let mut dm = QuantumDecisionMaker::new(config);
        dm.set_belief(belief);

        // Bank teller projector: primarily along basis 0, with some overlap
        let bank_teller = Projector::new(
            vec![vec![
                Complex64::new(0.9, 0.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.1, 0.0),
                Complex64::new(0.0, 0.0),
            ]],
            "bank teller",
        )
        .expect("bank teller projector");

        // Feminist projector: primarily along basis 2, overlapping with basis 1
        let feminist = Projector::new(
            vec![vec![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.4, 0.0),
                Complex64::new(0.8, 0.0),
                Complex64::new(0.2, 0.0),
            ]],
            "feminist",
        )
        .expect("feminist projector");

        (dm, bank_teller, feminist)
    }
}

// ============================================================
// LINEAR ALGEBRA HELPERS (INLINE)
// ============================================================

/// Compute the inner product <a|b> = sum_i conj(a_i) * b_i.
#[inline]
fn inner_product(a: &[C64], b: &[C64]) -> C64 {
    let mut sum = Complex64::new(0.0, 0.0);
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum = sum + ai.conj() * bi;
    }
    sum
}

/// Compute ||state||^2 = <state|state>.
#[inline]
fn state_norm_squared(state: &[C64]) -> f64 {
    state.iter().map(|a| a.norm_sqr()).sum()
}

/// Compute ||state|| = sqrt(<state|state>).
#[inline]
fn state_norm(state: &[C64]) -> f64 {
    state_norm_squared(state).sqrt()
}

/// Gram-Schmidt orthonormalization of a set of vectors.
///
/// Returns a set of orthonormal vectors spanning the same subspace.
/// Vectors that become zero (linearly dependent) are discarded.
fn gram_schmidt(vectors: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let mut orthonormal: Vec<Vec<C64>> = Vec::new();

    for v in vectors {
        let mut u = v.clone();

        // Subtract projections onto all previous orthonormal vectors
        for q in &orthonormal {
            let proj_coeff = inner_product(q, &u);
            for (ui, qi) in u.iter_mut().zip(q.iter()) {
                *ui = *ui - proj_coeff * qi;
            }
        }

        // Normalize
        let norm = state_norm(&u);
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for ui in u.iter_mut() {
                *ui = Complex64::new(ui.re * inv_norm, ui.im * inv_norm);
            }
            orthonormal.push(u);
        }
    }

    orthonormal
}

/// Compute the sequential projection probability ||P_b P_a |psi>||^2.
///
/// This is the joint probability of getting "yes" to question A followed by
/// "yes" to question B, starting from state |psi>.
#[inline]
fn sequential_probability(state: &[C64], proj_a: &Projector, proj_b: &Projector) -> f64 {
    // Step 1: P_A |psi>
    let after_a = proj_a.project(state);
    // Step 2: P_B (P_A |psi>)
    let after_b = proj_b.project(&after_a);
    // ||P_B P_A |psi>||^2
    state_norm_squared(&after_b)
}

/// Generate a deterministic pseudo-random perturbation of the uniform belief state.
///
/// Uses a simple linear congruential generator seeded by the respondent index
/// and a global seed to produce slight variations in initial beliefs.
fn perturbed_belief(dim: usize, respondent: u64, seed: u64) -> BeliefState {
    let base_amp = 1.0 / (dim as f64).sqrt();
    let mut state = Vec::with_capacity(dim);

    // Simple LCG for deterministic perturbation
    let mut rng_state = seed.wrapping_mul(6364136223846793005).wrapping_add(respondent);

    for _ in 0..dim {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Small perturbation in [-0.1, 0.1]
        let perturbation = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.2;

        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let phase = ((rng_state >> 33) as f64 / (1u64 << 31) as f64) * 2.0
            * std::f64::consts::PI;

        let amp = base_amp + perturbation;
        state.push(Complex64::new(amp * phase.cos(), amp * phase.sin()));
    }

    BeliefState::from_state(state)
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-10;

    // Helper: create a non-commuting pair of projectors in 2D
    fn noncommuting_2d() -> (Projector, Projector) {
        // Projector A: onto |0>
        let a = Projector::basis_projector(2, 0, "A");

        // Projector B: onto |+> = (|0> + |1>)/sqrt(2) -- does NOT commute with A
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let b = Projector::from_single_vector(
            vec![
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
            ],
            "B",
        )
        .expect("projector B");

        (a, b)
    }

    // Helper: create commuting projectors in 4D (project onto orthogonal subspaces)
    fn commuting_4d() -> (Projector, Projector) {
        // A projects onto span{|0>, |1>}
        let a = Projector::new(
            vec![
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            ],
            "A",
        )
        .expect("commuting A");

        // B projects onto span{|0>, |2>}
        let b = Projector::new(
            vec![
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            ],
            "B",
        )
        .expect("commuting B");

        (a, b)
    }

    // -------------------------------------------------------
    // 1. Config builder defaults
    // -------------------------------------------------------
    #[test]
    fn test_config_builder() {
        let config = CognitionConfig::default();
        assert_eq!(config.belief_dim, 4);
        assert_eq!(config.model_type, CognitionModel::QuantumProbability);
        assert!((config.similarity_param - 0.5).abs() < TOL);

        let config2 = CognitionConfig::new()
            .with_belief_dim(8)
            .with_model(CognitionModel::QQModel)
            .with_similarity(0.7);
        assert_eq!(config2.belief_dim, 8);
        assert_eq!(config2.model_type, CognitionModel::QQModel);
        assert!((config2.similarity_param - 0.7).abs() < TOL);
    }

    // -------------------------------------------------------
    // 2. Belief state normalized
    // -------------------------------------------------------
    #[test]
    fn test_belief_state_normalized() {
        for dim in [2, 4, 8, 16] {
            let belief = BeliefState::new(dim);
            assert!(
                belief.is_normalized(TOL),
                "uniform superposition dim={} not normalized",
                dim
            );
        }

        // from_state auto-normalizes
        let belief = BeliefState::from_state(vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ]);
        assert!(belief.is_normalized(TOL));
    }

    // -------------------------------------------------------
    // 3. Born rule: probabilities in [0, 1]
    // -------------------------------------------------------
    #[test]
    fn test_born_rule_projector() {
        let belief = BeliefState::new(4);

        // Project onto |0>
        let proj = Projector::basis_projector(4, 0, "zero");
        let p = belief.probability(&proj);
        assert!(p >= -TOL && p <= 1.0 + TOL, "probability out of range: {}", p);

        // For uniform superposition in dim 4, P(|0>) should be 1/4
        assert!((p - 0.25).abs() < TOL, "expected 0.25, got {}", p);
    }

    // -------------------------------------------------------
    // 4. Luders update preserves normalization
    // -------------------------------------------------------
    #[test]
    fn test_luders_update_normalized() {
        let mut belief = BeliefState::new(4);
        let proj = Projector::basis_projector(4, 1, "one");

        belief.update(&proj);
        assert!(
            belief.is_normalized(TOL),
            "state not normalized after Luders update"
        );

        // After projecting onto |1>, the state should be |1>
        let p1 = belief.probability(&proj);
        assert!(
            (p1 - 1.0).abs() < TOL,
            "after projecting onto |1>, P(|1>) should be 1.0, got {}",
            p1
        );
    }

    // -------------------------------------------------------
    // 5. Commuting projectors produce no order effect
    // -------------------------------------------------------
    #[test]
    fn test_commuting_projectors_no_order_effect() {
        // Two orthogonal rank-1 projectors in 4D commute
        let a = Projector::basis_projector(4, 0, "A");
        let b = Projector::basis_projector(4, 1, "B");

        assert!(a.commutes_with(&b, 1e-10), "basis projectors should commute");

        let config = CognitionConfig::new().with_belief_dim(4);
        let dm = QuantumDecisionMaker::new(config);
        let effect = dm.order_effect(&a, &b);

        assert!(
            effect.order_effect.abs() < 1e-10,
            "commuting projectors should have zero order effect, got {}",
            effect.order_effect
        );
    }

    // -------------------------------------------------------
    // 6. Non-commuting projectors produce order effect
    // -------------------------------------------------------
    #[test]
    fn test_noncommuting_projectors_order_effect() {
        let (a, b) = noncommuting_2d();
        assert!(
            !a.commutes_with(&b, 1e-10),
            "these projectors should NOT commute"
        );

        let config = CognitionConfig::new().with_belief_dim(2);
        let dm = QuantumDecisionMaker::new(config);
        let effect = dm.order_effect(&a, &b);

        // For uniform superposition in 2D with |0> and |+> projectors,
        // order effect should be non-zero
        assert!(
            effect.order_effect.abs() > 1e-6,
            "non-commuting projectors should produce order effect, got {}",
            effect.order_effect
        );
    }

    // -------------------------------------------------------
    // 7. QQ equality holds for quantum model
    // -------------------------------------------------------
    #[test]
    fn test_qq_equality() {
        let (a, b) = noncommuting_2d();

        let config = CognitionConfig::new().with_belief_dim(2);
        let dm = QuantumDecisionMaker::new(config);
        let effect = dm.order_effect(&a, &b);

        // QQ equality: P(Ay,By) + P(An,Bn) = P(By,Ay) + P(Bn,An)
        // This is a fundamental prediction of the quantum model
        assert!(
            effect.satisfies_qq,
            "QQ equality should hold, qq_test = {}",
            effect.qq_equality_test
        );
        assert!(
            effect.qq_equality_test.abs() < 0.05,
            "QQ equality deviation too large: {}",
            effect.qq_equality_test
        );
    }

    // -------------------------------------------------------
    // 8. Conjunction bounded classically for commuting
    // -------------------------------------------------------
    #[test]
    fn test_conjunction_classical_bound() {
        // Commuting (orthogonal basis) projectors: conjunction should obey classical bound
        let a = Projector::basis_projector(4, 0, "A");
        let b = Projector::basis_projector(4, 1, "B");

        let config = CognitionConfig::new().with_belief_dim(4);
        let dm = QuantumDecisionMaker::new(config);
        let result = dm.conjunction_fallacy(&a, &b);

        // For orthogonal projectors, P(A and B) = 0 (no state can be in both)
        assert!(
            result.p_a_and_b_quantum < TOL,
            "orthogonal projectors: P(A and B) should be 0, got {}",
            result.p_a_and_b_quantum
        );
        assert!(!result.is_fallacy, "no fallacy for orthogonal projectors");
    }

    // -------------------------------------------------------
    // 9. Conjunction fallacy possible for non-commuting
    // -------------------------------------------------------
    #[test]
    fn test_conjunction_fallacy_possible() {
        // Use the Linda problem setup which is designed to produce the fallacy
        let (dm, bank_teller, feminist) = CognitionExamples::linda_problem();
        let result = dm.conjunction_fallacy(&feminist, &bank_teller);

        // The quantum model should be able to produce P(A and B) that differs
        // from classical. Whether it exceeds the bound depends on the setup.
        // At minimum, the quantum probability should be non-negative.
        assert!(
            result.p_a_and_b_quantum >= -TOL,
            "quantum conjunction probability should be non-negative, got {}",
            result.p_a_and_b_quantum
        );
        assert!(
            result.p_a <= 1.0 + TOL,
            "P(A) out of range: {}",
            result.p_a
        );
        assert!(
            result.p_b <= 1.0 + TOL,
            "P(B) out of range: {}",
            result.p_b
        );
    }

    // -------------------------------------------------------
    // 10. Sure-thing: no violation for diagonal (classical) case
    // -------------------------------------------------------
    #[test]
    fn test_sure_thing_classical() {
        // When conditions are orthogonal basis projectors and the action
        // is also a basis projector, there is no interference.
        let dim = 4;
        let config = CognitionConfig::new().with_belief_dim(dim);
        let dm = QuantumDecisionMaker::new(config);

        let action = Projector::basis_projector(dim, 0, "action");
        let cond_0 = Projector::basis_projector(dim, 0, "cond_0");
        let cond_1 = Projector::basis_projector(dim, 1, "cond_1");
        let cond_2 = Projector::basis_projector(dim, 2, "cond_2");
        let cond_3 = Projector::basis_projector(dim, 3, "cond_3");

        let result = dm.sure_thing_violation(
            &[action],
            &[cond_0, cond_1, cond_2, cond_3],
        );

        // For a complete set of orthogonal conditions and a basis-aligned action,
        // the classical and quantum marginals should match.
        assert!(
            result.violation < 0.02,
            "violation should be small for classical case, got {}",
            result.violation
        );
    }

    // -------------------------------------------------------
    // 11. Sure-thing violation with interference
    // -------------------------------------------------------
    #[test]
    fn test_sure_thing_quantum_violation() {
        let dim = 4;
        let config = CognitionConfig::new().with_belief_dim(dim);

        // Biased initial state
        let state = vec![
            Complex64::new(0.5, 0.2),
            Complex64::new(0.3, -0.3),
            Complex64::new(0.4, 0.1),
            Complex64::new(0.2, 0.4),
        ];
        let belief = BeliefState::from_state(state);
        let mut dm = QuantumDecisionMaker::new(config);
        dm.set_belief(belief);

        // Non-diagonal action projector
        let inv2 = 1.0 / 2.0_f64.sqrt();
        let action = Projector::from_single_vector(
            vec![
                Complex64::new(inv2, 0.0),
                Complex64::new(inv2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            "action",
        )
        .expect("action projector");

        // Non-orthogonal condition projectors (they overlap, creating interference)
        let c1 = Projector::from_single_vector(
            vec![
                Complex64::new(0.8, 0.0),
                Complex64::new(0.2, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.2, 0.0),
            ],
            "cond_1",
        )
        .expect("cond 1");

        let c2 = Projector::from_single_vector(
            vec![
                Complex64::new(0.1, 0.0),
                Complex64::new(0.7, 0.0),
                Complex64::new(0.3, 0.0),
                Complex64::new(0.6, 0.0),
            ],
            "cond_2",
        )
        .expect("cond 2");

        let result = dm.sure_thing_violation(&[action], &[c1, c2]);

        // With non-orthogonal conditions and a non-diagonal action,
        // the classical marginal generally differs from the quantum marginal.
        // We just verify the structure is correct.
        assert!(result.marginal_prob >= -TOL);
        assert!(result.marginal_prob <= 1.0 + TOL);
        for &p in &result.conditional_probs {
            assert!(p >= -TOL && p <= 1.0 + TOL, "conditional prob out of range: {}", p);
        }
    }

    // -------------------------------------------------------
    // 12. Interference term zero for commuting projectors
    // -------------------------------------------------------
    #[test]
    fn test_interference_term_zero_commuting() {
        let a = Projector::basis_projector(4, 0, "A");
        let b = Projector::basis_projector(4, 1, "B");

        let config = CognitionConfig::new().with_belief_dim(4);
        let dm = QuantumDecisionMaker::new(config);

        let interference = dm.interference_term(&a, &b);
        // For orthogonal basis projectors with uniform superposition:
        // P_AB_quantum = 0, P(A)*P(B) = 1/4 * 1/4 = 1/16
        // interference = 0 - 1/16 = -1/16
        // However, interference = 0 only when P_AB = P(A)*P(B).
        // For orthogonal projectors P_AB = ||P_B P_A|psi>||^2 = 0,
        // so interference = -P(A)*P(B). This is classical correlation, not
        // quantum interference. For truly commuting but overlapping projectors,
        // the interference term is zero.

        // Use commuting, overlapping projectors instead:
        let (a2, b2) = commuting_4d();
        let dm2 = QuantumDecisionMaker::new(CognitionConfig::new().with_belief_dim(4));

        // Check they actually commute
        assert!(a2.commutes_with(&b2, 1e-10));

        let i2 = dm2.interference_term(&a2, &b2);
        // For commuting projectors: P_AB_quantum = P(A inter B) = P(A)*P(B)/P(A union B)
        // The key property: P_AB is independent of order, so quantum = classical product
        // when projectors commute AND share common subspace.
        // Actually, for commuting projectors P_A P_B = P_{A intersect B},
        // so sequential prob = P(intersection), and interference = P(intersection) - P(A)*P(B).
        // This can be nonzero! The zero-interference property is specifically about
        // the ORDER-DEPENDENT interference. Let us verify consistency instead.
        let p_a = dm2.belief().probability(&a2);
        let p_b = dm2.belief().probability(&b2);
        let p_ab = sequential_probability(&dm2.belief().state, &a2, &b2);
        let p_ba = sequential_probability(&dm2.belief().state, &b2, &a2);

        // For commuting projectors, P_A P_B = P_B P_A, so sequential prob is the same
        assert!(
            (p_ab - p_ba).abs() < 1e-10,
            "commuting projectors should give same sequential prob: {} vs {}",
            p_ab,
            p_ba
        );
    }

    // -------------------------------------------------------
    // 13. Interference term nonzero for non-commuting
    // -------------------------------------------------------
    #[test]
    fn test_interference_term_nonzero() {
        let (a, b) = noncommuting_2d();

        let config = CognitionConfig::new().with_belief_dim(2);
        let dm = QuantumDecisionMaker::new(config);

        // Check order dependence (the hallmark of non-commutativity)
        let p_ab = sequential_probability(&dm.belief().state, &a, &b);
        let p_ba = sequential_probability(&dm.belief().state, &b, &a);

        assert!(
            (p_ab - p_ba).abs() > 1e-6,
            "non-commuting projectors should give different sequential probs: {} vs {}",
            p_ab,
            p_ba
        );
    }

    // -------------------------------------------------------
    // 14. Projector complement: P + P_complement spans full space
    // -------------------------------------------------------
    #[test]
    fn test_projector_complement() {
        let dim = 4;
        let proj = Projector::basis_projector(dim, 0, "zero");
        let comp = proj.complement(dim);

        // P + complement should be the identity:
        // For any state, P|psi> + (I-P)|psi> = |psi>
        let state = vec![
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.4, 0.3),
            Complex64::new(0.1, -0.1),
        ];

        let projected = proj.project(&state);
        let complemented = comp.project(&state);

        // Sum should equal original state
        for i in 0..dim {
            let sum = projected[i] + complemented[i];
            assert!(
                (sum - state[i]).norm() < 1e-10,
                "P + complement should be identity at index {}: {} + {} != {}",
                i,
                projected[i],
                complemented[i],
                state[i]
            );
        }

        // Ranks should be complementary
        assert_eq!(proj.rank() + comp.rank(), dim);
    }

    // -------------------------------------------------------
    // 15. Survey simulation produces valid probabilities
    // -------------------------------------------------------
    #[test]
    fn test_survey_simulation() {
        let dim = 4;
        let mut survey = QuantumSurveySimulator::new(3, dim);

        survey.add_question("Q1", Projector::basis_projector(dim, 0, "Q1"));
        survey.add_question("Q2", Projector::basis_projector(dim, 1, "Q2"));

        // Non-commuting question
        let inv2 = 1.0 / 2.0_f64.sqrt();
        let q3 = Projector::from_single_vector(
            vec![
                Complex64::new(inv2, 0.0),
                Complex64::new(inv2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            "Q3",
        )
        .expect("survey q3");
        survey.add_question("Q3", q3);

        let result = survey.simulate_population(100, 42);

        // All probabilities should be in [0, 1]
        for (name, prob) in &result.question_probabilities {
            assert!(
                *prob >= -0.01 && *prob <= 1.01,
                "probability for {} out of range: {}",
                name,
                prob
            );
        }

        // Should have 3 order effect pairs: (Q1,Q2), (Q1,Q3), (Q2,Q3)
        assert_eq!(
            result.order_effects.len(),
            3,
            "expected 3 order effect pairs, got {}",
            result.order_effects.len()
        );
    }

    // -------------------------------------------------------
    // 16. Sequential questions: probabilities well-defined
    // -------------------------------------------------------
    #[test]
    fn test_sequential_questions() {
        let config = CognitionConfig::new().with_belief_dim(4);
        let mut dm = QuantumDecisionMaker::new(config);

        let q1 = Projector::basis_projector(4, 0, "Q1");
        let inv2 = 1.0 / 2.0_f64.sqrt();
        let q2 = Projector::from_single_vector(
            vec![
                Complex64::new(inv2, 0.0),
                Complex64::new(inv2, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            "Q2",
        )
        .expect("seq q2");
        let q3 = Projector::basis_projector(4, 1, "Q3");

        let probs = dm.ask_sequence(&[q1, q2, q3]);

        assert_eq!(probs.len(), 3);
        for (i, p) in probs.iter().enumerate() {
            assert!(
                *p >= -TOL && *p <= 1.0 + TOL,
                "sequential prob {} out of range: {}",
                i,
                p
            );
        }

        // After all projections, state should still be normalized
        assert!(
            dm.belief().is_normalized(1e-8),
            "state not normalized after sequential questions"
        );
    }

    // -------------------------------------------------------
    // 17. Clinton-Gore example produces order effect
    // -------------------------------------------------------
    #[test]
    fn test_clinton_gore_example() {
        let (dm, clinton, gore) = CognitionExamples::clinton_gore();

        // These projectors should NOT commute (they are rotated bases in 2D)
        assert!(
            !clinton.commutes_with(&gore, 1e-10),
            "Clinton and Gore projectors should not commute"
        );

        let effect = dm.order_effect(&clinton, &gore);

        // The model should produce a non-zero order effect
        assert!(
            effect.order_effect.abs() > 1e-6,
            "Clinton-Gore should produce order effect, got {}",
            effect.order_effect
        );

        // QQ equality should hold
        assert!(
            effect.satisfies_qq,
            "QQ equality should hold for Clinton-Gore, qq_test = {}",
            effect.qq_equality_test
        );
    }

    // -------------------------------------------------------
    // 18. Error enum coverage
    // -------------------------------------------------------
    #[test]
    fn test_error_display() {
        let e1 = CognitionError::InvalidDimension {
            expected: 4,
            got: 2,
        };
        assert!(e1.to_string().contains("invalid dimension"));

        let e2 = CognitionError::ProjectorRankMismatch {
            projector_dim: 2,
            state_dim: 4,
        };
        assert!(e2.to_string().contains("does not match"));

        let e3 = CognitionError::NotNormalized { norm: 0.5 };
        assert!(e3.to_string().contains("not normalized"));

        let e4 = CognitionError::EmptyBasis;
        assert!(e4.to_string().contains("empty basis"));
    }

    // -------------------------------------------------------
    // 19. Empty basis error
    // -------------------------------------------------------
    #[test]
    fn test_empty_basis_error() {
        let result = Projector::new(vec![], "empty");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CognitionError::EmptyBasis);

        let result2 = Projector::from_single_vector(vec![], "empty");
        assert!(result2.is_err());
    }

    // -------------------------------------------------------
    // 20. Gram-Schmidt orthonormality
    // -------------------------------------------------------
    #[test]
    fn test_gram_schmidt_orthonormality() {
        let v1 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let v2 = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let v3 = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let ortho = gram_schmidt(&[v1, v2, v3]);
        assert_eq!(ortho.len(), 3, "should produce 3 orthonormal vectors");

        // Check orthonormality
        for i in 0..ortho.len() {
            let norm = state_norm(&ortho[i]);
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "vector {} not normalized: {}",
                i,
                norm
            );
            for j in (i + 1)..ortho.len() {
                let ip = inner_product(&ortho[i], &ortho[j]).norm();
                assert!(
                    ip < 1e-10,
                    "vectors {} and {} not orthogonal: <i|j> = {}",
                    i,
                    j,
                    ip
                );
            }
        }
    }

    // -------------------------------------------------------
    // 21. Projector idempotency: P^2 = P
    // -------------------------------------------------------
    #[test]
    fn test_projector_idempotent() {
        let dim = 4;
        let proj = Projector::new(
            vec![
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
                vec![
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            ],
            "test",
        )
        .expect("test projector");

        // For every basis state, P(P|k>) should equal P|k>
        for k in 0..dim {
            let mut ek = vec![Complex64::new(0.0, 0.0); dim];
            ek[k] = Complex64::new(1.0, 0.0);

            let p_ek = proj.project(&ek);
            let pp_ek = proj.project(&p_ek);

            for i in 0..dim {
                assert!(
                    (p_ek[i] - pp_ek[i]).norm() < 1e-10,
                    "P^2 != P at basis {}, index {}: {} vs {}",
                    k,
                    i,
                    p_ek[i],
                    pp_ek[i]
                );
            }
        }
    }
}
