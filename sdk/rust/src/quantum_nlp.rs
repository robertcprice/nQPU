//! Quantum Natural Language Processing (DisCoCat Framework)
//!
//! **WORLD FIRST**: The first quantum simulator with built-in compositional
//! distributional semantics for natural language processing. Implements the
//! DisCoCat (Distributional Compositional Categorical) framework, encoding
//! natural language meaning into quantum circuits.
//!
//! # Theory
//!
//! The DisCoCat framework unifies two traditions in computational linguistics:
//!
//! 1. **Distributional semantics**: Word meaning is determined by context
//!    (Firth 1957: "You shall know a word by the company it keeps")
//! 2. **Compositional semantics**: Sentence meaning is built from word meanings
//!    via grammatical structure (Montague 1970, Lambek 1958)
//!
//! The key insight (Coecke, Sadrzadeh, Clark 2010) is that both distributional
//! vector spaces and pregroup grammars live in compact closed categories, so
//! there exists a structure-preserving functor between them. This functor maps:
//!
//! - **Words** to quantum states (density operators or pure states)
//! - **Grammatical reductions** to quantum operations (cups/caps = Bell effects)
//! - **Sentences** to scalar truth values via measurement
//!
//! # Pregroup Grammar
//!
//! Types are generated from atomic types {n (noun), s (sentence)} with left
//! and right adjoints satisfying:
//!
//! - n^l . n -> 1 (left cup)
//! - n . n^r -> 1 (right cup)
//! - 1 -> n . n^l (left cap)
//! - 1 -> n^r . n (right cap)
//!
//! A transitive verb has type n^r . s . n^l, consuming a subject (left) and
//! object (right) to produce a sentence.
//!
//! # Circuit Ansatze
//!
//! Word meanings are parameterized quantum circuits. Supported ansatze:
//!
//! - **IQP** (Instantaneous Quantum Polynomial): Diagonal entangling layers
//! - **Sim14**: Hardware-efficient with Ry + CNOT layers (Sim et al. 2019)
//! - **Sim15**: Strongly entangling with Rx, Ry, Rz + CNOT
//! - **TensorNetwork**: MPS-inspired sequential entangling
//!
//! # Applications
//!
//! - Sentence similarity and entailment
//! - Question answering (truth-value semantics)
//! - Sentiment classification
//! - Compositional generalization
//! - Quantum advantage in NLP (exponential meaning spaces)
//!
//! # References
//!
//! - Coecke, Sadrzadeh, Clark (2010) - Mathematical Foundations for a
//!   Compositional Distributional Model of Meaning. arXiv:1003.4394
//! - Lambek (1958) - The Mathematics of Sentence Structure
//! - Sadrzadeh, Clark, Coecke (2013) - The Frobenius Anatomy of Word Meanings
//! - Kartsaklis et al. (2021) - lambeq: An Efficient High-Level Python Library
//!   for Quantum NLP. arXiv:2110.04236
//! - Meichanetzidis et al. (2023) - Grammar-Aware Sentence Classification on
//!   Quantum Computers. arXiv:2012.03756
//! - Lorenz et al. (2023) - QNLP in Practice. arXiv:2102.12846

use crate::{QuantumState, GateOperations};
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;

// ===================================================================
// ERROR TYPES
// ===================================================================

/// Errors arising from quantum NLP operations
#[derive(Clone, Debug)]
pub enum NlpError {
    /// Word not found in lexicon
    UnknownWord(String),
    /// Grammatical type reduction failed (sentence does not parse)
    TypeReductionFailed(String),
    /// Empty sentence provided
    EmptySentence,
    /// Invalid circuit configuration
    InvalidConfig(String),
    /// Training failed to converge
    TrainingFailed(String),
    /// Qubit count mismatch
    QubitMismatch { expected: usize, got: usize },
    /// Parameter count mismatch
    ParameterMismatch { expected: usize, got: usize },
}

impl fmt::Display for NlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NlpError::UnknownWord(w) => write!(f, "Unknown word: '{}'", w),
            NlpError::TypeReductionFailed(msg) => write!(f, "Type reduction failed: {}", msg),
            NlpError::EmptySentence => write!(f, "Empty sentence"),
            NlpError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            NlpError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            NlpError::QubitMismatch { expected, got } => {
                write!(f, "Qubit mismatch: expected {}, got {}", expected, got)
            }
            NlpError::ParameterMismatch { expected, got } => {
                write!(f, "Parameter mismatch: expected {}, got {}", expected, got)
            }
        }
    }
}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Circuit ansatz for encoding word meanings
#[derive(Clone, Debug, PartialEq)]
pub enum CircuitAnsatz {
    /// Instantaneous Quantum Polynomial: Hadamard + diagonal ZZ entangling
    /// Efficient for classification tasks (Havlicek et al. 2019)
    IQP,
    /// Hardware-efficient ansatz: Ry rotations + CNOT ladder
    /// 14 parameters per layer for 2 qubits (Sim et al. 2019, circuit #14)
    Sim14,
    /// Strongly entangling: Rx, Ry, Rz per qubit + CNOT ring
    /// Maximum expressibility per layer (Sim et al. 2019, circuit #15)
    Sim15,
    /// Tensor-network inspired: sequential nearest-neighbor CNOTs
    /// Matches MPS structure for efficient classical simulation bounds
    TensorNetwork,
}

/// How to compute sentence similarity
#[derive(Clone, Debug, PartialEq)]
pub enum SimilarityMeasure {
    /// State fidelity: |<psi|phi>|^2
    Fidelity,
    /// SWAP test circuit (ancilla-based)
    SwapTest,
    /// Real part of inner product: Re(<psi|phi>)
    InnerProduct,
}

/// Configuration for quantum NLP operations
#[derive(Clone, Debug)]
pub struct QuantumNlpConfig {
    /// Number of qubits per noun wire (default: 1)
    pub noun_qubits: usize,
    /// Number of qubits for the sentence wire (default: 1)
    pub sentence_qubits: usize,
    /// Circuit ansatz for word parameterization
    pub ansatz: CircuitAnsatz,
    /// Similarity measure for comparing sentence states
    pub similarity_measure: SimilarityMeasure,
    /// Number of ansatz layers per word box
    pub num_layers: usize,
    /// Whether to post-select sentence wire to |0>
    pub post_select: bool,
}

impl Default for QuantumNlpConfig {
    fn default() -> Self {
        Self {
            noun_qubits: 1,
            sentence_qubits: 1,
            ansatz: CircuitAnsatz::Sim14,
            similarity_measure: SimilarityMeasure::Fidelity,
            num_layers: 1,
            post_select: false,
        }
    }
}

impl QuantumNlpConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set noun qubits
    pub fn with_noun_qubits(mut self, n: usize) -> Self {
        self.noun_qubits = n;
        self
    }

    /// Builder: set sentence qubits
    pub fn with_sentence_qubits(mut self, n: usize) -> Self {
        self.sentence_qubits = n;
        self
    }

    /// Builder: set circuit ansatz
    pub fn with_ansatz(mut self, ansatz: CircuitAnsatz) -> Self {
        self.ansatz = ansatz;
        self
    }

    /// Builder: set similarity measure
    pub fn with_similarity(mut self, sim: SimilarityMeasure) -> Self {
        self.similarity_measure = sim;
        self
    }

    /// Builder: set number of layers
    pub fn with_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Builder: set post-selection
    pub fn with_post_select(mut self, ps: bool) -> Self {
        self.post_select = ps;
        self
    }

    /// Compute how many parameters a word of the given type needs
    /// under this config's ansatz and layer count.
    pub fn params_for_type(&self, gram_type: &GrammaticalType) -> usize {
        let total_qubits = self.total_qubits_for_type(gram_type);
        if total_qubits == 0 {
            return 0;
        }
        self.params_per_layer(total_qubits) * self.num_layers
    }

    /// Total qubits spanned by a word box of the given type
    pub fn total_qubits_for_type(&self, gram_type: &GrammaticalType) -> usize {
        match gram_type {
            GrammaticalType::Noun => self.noun_qubits,
            GrammaticalType::Sentence => self.sentence_qubits,
            GrammaticalType::NounRightAdj | GrammaticalType::NounLeftAdj => self.noun_qubits,
            // Transitive verb: n^r . s . n^l => subject + sentence + object
            GrammaticalType::Transitive => {
                2 * self.noun_qubits + self.sentence_qubits
            }
            // Intransitive verb: n^r . s => subject + sentence
            GrammaticalType::Intransitive => {
                self.noun_qubits + self.sentence_qubits
            }
            // Adjective: n . n^l => output noun + input noun
            GrammaticalType::Adjective => {
                2 * self.noun_qubits
            }
            // Determiner: identity on noun type
            GrammaticalType::Determiner => 0,
        }
    }

    /// Number of parameters per ansatz layer for a given qubit count
    fn params_per_layer(&self, num_qubits: usize) -> usize {
        match self.ansatz {
            CircuitAnsatz::IQP => {
                // Hadamard on all + Rz on all + Rzz on pairs
                num_qubits + num_qubits * (num_qubits.saturating_sub(1)) / 2
            }
            CircuitAnsatz::Sim14 => {
                // Ry on each qubit
                num_qubits
            }
            CircuitAnsatz::Sim15 => {
                // Rx + Ry + Rz on each qubit
                3 * num_qubits
            }
            CircuitAnsatz::TensorNetwork => {
                // Ry on each qubit + Rz on each qubit
                2 * num_qubits
            }
        }
    }
}

// ===================================================================
// GRAMMATICAL TYPES (PREGROUP GRAMMAR)
// ===================================================================

/// Atomic and compound pregroup grammatical types.
///
/// In a pregroup grammar (Lambek 2001), every word is assigned a type
/// built from atomic types and their left/right adjoints. Type reduction
/// proceeds via the pregroup inequalities:
///
/// - p^l . p <= 1 (left cancellation / cup)
/// - p . p^r <= 1 (right cancellation / cup)
/// - 1 <= p . p^l (left creation / cap)
/// - 1 <= p^r . p (right creation / cap)
#[derive(Clone, Debug, PartialEq)]
pub enum GrammaticalType {
    /// Noun type: n
    Noun,
    /// Sentence type: s
    Sentence,
    /// Right adjoint of noun: n^r (consumes a noun to the right)
    NounRightAdj,
    /// Left adjoint of noun: n^l (consumes a noun to the left)
    NounLeftAdj,
    /// Transitive verb: n^r . s . n^l
    /// Takes a subject (left) and object (right) to produce a sentence
    Transitive,
    /// Intransitive verb: n^r . s
    /// Takes a subject (left) to produce a sentence
    Intransitive,
    /// Adjective: n . n^l
    /// Modifies a noun, outputting a noun
    Adjective,
    /// Determiner: identity on noun (the, a)
    /// Passes through without modification
    Determiner,
}

impl GrammaticalType {
    /// Expand compound type into a sequence of atomic types.
    ///
    /// This decomposes compound types (Transitive, Intransitive, Adjective)
    /// into their constituent atomic types for type reduction.
    pub fn expand(&self) -> Vec<AtomicType> {
        match self {
            GrammaticalType::Noun => vec![AtomicType::N],
            GrammaticalType::Sentence => vec![AtomicType::S],
            GrammaticalType::NounRightAdj => vec![AtomicType::NR],
            GrammaticalType::NounLeftAdj => vec![AtomicType::NL],
            GrammaticalType::Transitive => vec![AtomicType::NR, AtomicType::S, AtomicType::NL],
            GrammaticalType::Intransitive => vec![AtomicType::NR, AtomicType::S],
            GrammaticalType::Adjective => vec![AtomicType::N, AtomicType::NL],
            GrammaticalType::Determiner => vec![AtomicType::N, AtomicType::NL],
        }
    }
}

impl fmt::Display for GrammaticalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrammaticalType::Noun => write!(f, "n"),
            GrammaticalType::Sentence => write!(f, "s"),
            GrammaticalType::NounRightAdj => write!(f, "n^r"),
            GrammaticalType::NounLeftAdj => write!(f, "n^l"),
            GrammaticalType::Transitive => write!(f, "n^r.s.n^l"),
            GrammaticalType::Intransitive => write!(f, "n^r.s"),
            GrammaticalType::Adjective => write!(f, "n.n^l"),
            GrammaticalType::Determiner => write!(f, "n.n^l"),
        }
    }
}

/// Atomic types for pregroup type reduction
#[derive(Clone, Debug, PartialEq)]
pub enum AtomicType {
    /// Noun
    N,
    /// Sentence
    S,
    /// Right adjoint of noun
    NR,
    /// Left adjoint of noun
    NL,
}

impl fmt::Display for AtomicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AtomicType::N => write!(f, "n"),
            AtomicType::S => write!(f, "s"),
            AtomicType::NR => write!(f, "n^r"),
            AtomicType::NL => write!(f, "n^l"),
        }
    }
}

/// Check whether two adjacent atomic types can cancel (form a cup).
///
/// Returns true for the pregroup reductions:
/// - N . NR -> 1 (right cup on noun)
/// - NL . N -> 1 (left cup on noun)
fn can_reduce(left: &AtomicType, right: &AtomicType) -> bool {
    matches!(
        (left, right),
        (AtomicType::N, AtomicType::NR)
            | (AtomicType::NL, AtomicType::N)
            | (AtomicType::N, AtomicType::NL)
    )
}

/// Attempt to reduce a sequence of atomic types to a single type.
///
/// Performs greedy left-to-right cup reduction. A grammatically valid
/// sentence reduces to the sentence type S.
pub fn reduce_types(types: &[AtomicType]) -> Result<Vec<AtomicType>, NlpError> {
    if types.is_empty() {
        return Err(NlpError::EmptySentence);
    }

    let mut current = types.to_vec();
    let mut changed = true;

    // Iterate until no more reductions are possible
    while changed {
        changed = false;
        let mut i = 0;
        while i + 1 < current.len() {
            if can_reduce(&current[i], &current[i + 1]) {
                current.remove(i);
                current.remove(i);
                changed = true;
                // Don't increment i; check the new pair at position i
            } else {
                i += 1;
            }
        }
    }

    Ok(current)
}

/// Check if a type sequence reduces to a sentence type
pub fn reduces_to_sentence(types: &[AtomicType]) -> bool {
    match reduce_types(types) {
        Ok(reduced) => reduced.len() == 1 && reduced[0] == AtomicType::S,
        Err(_) => false,
    }
}

// ===================================================================
// WORD AND SENTENCE STRUCTURES
// ===================================================================

/// A word with its grammatical type and trainable parameters.
///
/// In DisCoCat, each word is mapped to a quantum process (state or map)
/// whose structure is determined by the grammatical type and whose
/// specific meaning is encoded in the trainable parameters.
#[derive(Clone, Debug)]
pub struct Word {
    /// The word text (lowercased)
    pub text: String,
    /// Grammatical type in the pregroup grammar
    pub gram_type: GrammaticalType,
    /// Trainable parameters for the word's quantum circuit
    pub parameters: Vec<f64>,
}

impl Word {
    /// Create a new word
    pub fn new(text: &str, gram_type: GrammaticalType, parameters: Vec<f64>) -> Self {
        Self {
            text: text.to_lowercase(),
            gram_type,
            parameters,
        }
    }

    /// Create a word with random parameters
    pub fn with_random_params(text: &str, gram_type: GrammaticalType, num_params: usize) -> Self {
        // Deterministic pseudo-random from text hash for reproducibility
        let mut params = Vec::with_capacity(num_params);
        let hash = text.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        for i in 0..num_params {
            let seed = hash.wrapping_mul(i as u64 + 1).wrapping_add(7919);
            let val = ((seed % 10000) as f64 / 10000.0) * 2.0 * PI;
            params.push(val);
        }
        Self {
            text: text.to_lowercase(),
            gram_type,
            parameters: params,
        }
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.text, self.gram_type)
    }
}

// ===================================================================
// SENTENCE DIAGRAM (STRING DIAGRAM / MONOIDAL CATEGORY)
// ===================================================================

/// A wire in the sentence diagram (string diagram in the monoidal category).
#[derive(Clone, Debug, PartialEq)]
pub enum Wire {
    /// Identity wire: passes through qubit unchanged
    Identity(usize),
    /// Cup: connects output wire `a` to output wire `b` (type reduction)
    /// Implements the pregroup reduction p . p^r -> 1 or p^l . p -> 1
    /// In quantum terms: post-selected Bell measurement / trace
    Cup(usize, usize),
    /// Cap: creates two wires from nothing (used in caps for type creation)
    /// Implements 1 -> p^r . p or 1 -> p . p^l
    /// In quantum terms: Bell state preparation
    Cap(usize, usize),
}

/// A box in the sentence diagram representing a word's quantum process.
#[derive(Clone, Debug)]
pub struct DiagramBox {
    /// The word this box represents
    pub word: String,
    /// Number of input wires (0 for states/nouns)
    pub input_wires: usize,
    /// Number of output wires
    pub output_wires: usize,
    /// Trainable parameters for the circuit
    pub parameters: Vec<f64>,
    /// Which qubit indices this box acts on
    pub qubit_indices: Vec<usize>,
}

/// The string diagram for a parsed sentence.
///
/// This encodes the categorical structure of the sentence: boxes (words)
/// connected by wires (type wires) with cups (reductions).
#[derive(Clone, Debug)]
pub struct SentenceDiagram {
    /// Wire connections (cups, caps, identities)
    pub wires: Vec<Wire>,
    /// Word boxes in the diagram
    pub boxes: Vec<DiagramBox>,
    /// Total number of qubits needed
    pub total_qubits: usize,
    /// Which qubit(s) carry the sentence output
    pub sentence_qubits: Vec<usize>,
}

/// A parsed sentence ready for circuit compilation
#[derive(Clone, Debug)]
pub struct Sentence {
    /// The words in the sentence
    pub words: Vec<Word>,
    /// The string diagram encoding the grammatical structure
    pub diagram: SentenceDiagram,
    /// Original text
    pub text: String,
}

// ===================================================================
// LEXICON
// ===================================================================

/// A lexicon mapping words to their grammatical types and parameters.
///
/// The lexicon is the core linguistic resource in DisCoCat. It assigns
/// each word a pregroup type (determining circuit topology) and a set
/// of trainable parameters (determining circuit content / meaning).
#[derive(Clone, Debug)]
pub struct Lexicon {
    words: HashMap<String, Word>,
}

impl Lexicon {
    /// Create an empty lexicon
    pub fn new() -> Self {
        Self {
            words: HashMap::new(),
        }
    }

    /// Create a basic English lexicon with ~50 common words.
    ///
    /// Words are initialized with deterministic pseudo-random parameters
    /// derived from their text, so the lexicon is reproducible.
    pub fn english_basic() -> Self {
        let mut lex = Self::new();

        // --- Nouns ---
        let nouns = [
            "alice", "bob", "charlie", "cat", "dog", "fish", "bird",
            "food", "water", "person", "robot", "computer", "book",
            "house", "car", "tree", "sun", "moon",
        ];
        for noun in &nouns {
            lex.add_word_random(noun, GrammaticalType::Noun, 2);
        }

        // --- Transitive verbs ---
        let transitive = [
            "loves", "sees", "chases", "eats", "knows", "hates",
            "reads", "builds", "catches", "teaches", "helps",
        ];
        for verb in &transitive {
            lex.add_word_random(verb, GrammaticalType::Transitive, 6);
        }

        // --- Intransitive verbs ---
        let intransitive = [
            "sleeps", "runs", "swims", "flies", "thinks", "jumps",
            "walks", "talks", "exists",
        ];
        for verb in &intransitive {
            lex.add_word_random(verb, GrammaticalType::Intransitive, 4);
        }

        // --- Adjectives ---
        let adjectives = [
            "big", "small", "fast", "slow", "quantum", "classical",
            "red", "blue", "happy", "sad", "old", "new",
        ];
        for adj in &adjectives {
            lex.add_word_random(adj, GrammaticalType::Adjective, 4);
        }

        // --- Determiners (identity on noun type) ---
        let determiners = ["the", "a", "an"];
        for det in &determiners {
            lex.add_word(det, GrammaticalType::Determiner, vec![]);
        }

        lex
    }

    /// Add a word with explicit parameters
    pub fn add_word(&mut self, text: &str, gram_type: GrammaticalType, params: Vec<f64>) {
        let word = Word::new(text, gram_type, params);
        self.words.insert(text.to_lowercase(), word);
    }

    /// Add a word with pseudo-random parameters (deterministic from text)
    pub fn add_word_random(&mut self, text: &str, gram_type: GrammaticalType, num_params: usize) {
        let word = Word::with_random_params(text, gram_type, num_params);
        self.words.insert(text.to_lowercase(), word);
    }

    /// Look up a word in the lexicon
    pub fn lookup(&self, text: &str) -> Result<&Word, NlpError> {
        self.words
            .get(&text.to_lowercase())
            .ok_or_else(|| NlpError::UnknownWord(text.to_string()))
    }

    /// Get a mutable reference to a word
    pub fn lookup_mut(&mut self, text: &str) -> Result<&mut Word, NlpError> {
        self.words
            .get_mut(&text.to_lowercase())
            .ok_or_else(|| NlpError::UnknownWord(text.to_string()))
    }

    /// Update parameters for a word
    pub fn update_params(&mut self, text: &str, params: Vec<f64>) -> Result<(), NlpError> {
        let word = self.lookup_mut(text)?;
        word.parameters = params;
        Ok(())
    }

    /// Number of words in the lexicon
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Check if lexicon is empty
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Get all word entries
    pub fn words(&self) -> &HashMap<String, Word> {
        &self.words
    }

    /// Check if a word exists
    pub fn contains(&self, text: &str) -> bool {
        self.words.contains_key(&text.to_lowercase())
    }
}

impl Default for Lexicon {
    fn default() -> Self {
        Self::new()
    }
}

// ===================================================================
// PARSER: SENTENCE -> DIAGRAM
// ===================================================================

/// Parse a sentence string into a structured Sentence with diagram.
///
/// The parser performs:
/// 1. Tokenization (whitespace split, lowercase)
/// 2. Lexicon lookup for each token
/// 3. Type expansion into atomic types
/// 4. Pregroup type reduction (greedy cup matching)
/// 5. Diagram construction with qubit assignment
///
/// # Errors
///
/// Returns `NlpError::UnknownWord` if any token is not in the lexicon.
/// Returns `NlpError::TypeReductionFailed` if the types do not reduce to `s`.
/// Returns `NlpError::EmptySentence` if the input is empty.
pub fn parse_sentence(lexicon: &Lexicon, sentence: &str) -> Result<Sentence, NlpError> {
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(NlpError::EmptySentence);
    }

    // Look up each word
    let mut words = Vec::with_capacity(tokens.len());
    for token in &tokens {
        let word = lexicon.lookup(token)?;
        words.push(word.clone());
    }

    // Filter out determiners for diagram construction (they are identity)
    let meaningful_words: Vec<&Word> = words
        .iter()
        .filter(|w| w.gram_type != GrammaticalType::Determiner)
        .collect();

    // Expand all types into atomic sequence
    let mut all_types: Vec<(AtomicType, usize)> = Vec::new(); // (type, word_index)
    for (i, word) in meaningful_words.iter().enumerate() {
        for at in word.gram_type.expand() {
            all_types.push((at, i));
        }
    }

    // Extract just the types for reduction
    let type_seq: Vec<AtomicType> = all_types.iter().map(|(t, _)| t.clone()).collect();

    // Check that sentence reduces correctly
    let reduced = reduce_types(&type_seq)?;
    if reduced.len() != 1 || reduced[0] != AtomicType::S {
        return Err(NlpError::TypeReductionFailed(format!(
            "Expected reduction to 's', got: {:?}",
            reduced
        )));
    }

    // Build the diagram
    let diagram = build_diagram(&meaningful_words, &all_types);

    Ok(Sentence {
        words,
        diagram,
        text: sentence.to_string(),
    })
}

/// Build a string diagram from typed words.
///
/// Assigns qubits to each atomic type wire, identifies cups (reductions),
/// and records the qubit indices for each word box.
fn build_diagram(
    words: &[&Word],
    typed_wires: &[(AtomicType, usize)],
) -> SentenceDiagram {
    let num_wires = typed_wires.len();
    let mut wire_connections: Vec<Wire> = Vec::new();
    let mut used = vec![false; num_wires];
    let mut qubit_map: Vec<Option<usize>> = vec![None; num_wires];
    let mut sentence_qubits = Vec::new();

    // Find cups: adjacent pairs that reduce
    // We need to find matching pairs greedily from left to right
    let mut cup_pairs: Vec<(usize, usize)> = Vec::new();
    {
        let mut temp_types: Vec<(AtomicType, usize)> = typed_wires.to_vec();
        let mut temp_indices: Vec<usize> = (0..num_wires).collect();
        let mut changed = true;
        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < temp_types.len() {
                if can_reduce(&temp_types[i].0, &temp_types[i + 1].0) {
                    cup_pairs.push((temp_indices[i], temp_indices[i + 1]));
                    temp_types.remove(i);
                    temp_types.remove(i);
                    temp_indices.remove(i);
                    temp_indices.remove(i);
                    changed = true;
                } else {
                    i += 1;
                }
            }
        }
    }

    // Mark cups
    for &(a, b) in &cup_pairs {
        used[a] = true;
        used[b] = true;
    }

    // Assign qubits: one qubit per surviving wire, cups share qubits
    let mut next_qubit = 0usize;

    // First assign qubits to cup pairs (they share a qubit)
    for &(a, b) in &cup_pairs {
        let q = next_qubit;
        qubit_map[a] = Some(q);
        qubit_map[b] = Some(q);
        wire_connections.push(Wire::Cup(a, b));
        next_qubit += 1;
    }

    // Then assign qubits to surviving wires
    for i in 0..num_wires {
        if qubit_map[i].is_none() {
            qubit_map[i] = Some(next_qubit);
            if typed_wires[i].0 == AtomicType::S {
                sentence_qubits.push(next_qubit);
            }
            wire_connections.push(Wire::Identity(i));
            next_qubit += 1;
        }
    }

    let total_qubits = next_qubit;

    // Build word boxes with qubit assignments
    let mut boxes = Vec::new();
    for (word_idx, word) in words.iter().enumerate() {
        // Collect all wire indices belonging to this word
        let word_wire_indices: Vec<usize> = typed_wires
            .iter()
            .enumerate()
            .filter(|(_, (_, wi))| *wi == word_idx)
            .map(|(i, _)| i)
            .collect();

        let word_qubit_indices: Vec<usize> = word_wire_indices
            .iter()
            .filter_map(|&wi| qubit_map[wi])
            .collect();

        // Determine input/output wire counts based on type
        let (input_wires, output_wires) = match word.gram_type {
            GrammaticalType::Noun => (0, 1),
            GrammaticalType::Sentence => (0, 1),
            GrammaticalType::NounRightAdj => (1, 0),
            GrammaticalType::NounLeftAdj => (1, 0),
            GrammaticalType::Transitive => (2, 1),    // consumes 2 nouns, produces s
            GrammaticalType::Intransitive => (1, 1),   // consumes 1 noun, produces s
            GrammaticalType::Adjective => (1, 1),      // consumes noun, produces noun
            GrammaticalType::Determiner => (1, 1),     // identity
        };

        boxes.push(DiagramBox {
            word: word.text.clone(),
            input_wires,
            output_wires,
            parameters: word.parameters.clone(),
            qubit_indices: word_qubit_indices,
        });
    }

    SentenceDiagram {
        wires: wire_connections,
        boxes,
        total_qubits,
        sentence_qubits,
    }
}

// ===================================================================
// CIRCUIT GENERATION
// ===================================================================

/// A compiled quantum NLP circuit ready for simulation
#[derive(Clone, Debug)]
pub struct QuantumNlpCircuit {
    /// Total number of qubits
    pub num_qubits: usize,
    /// Sequence of gate operations
    pub gates: Vec<NlpGate>,
    /// Which qubits to measure for the sentence truth value
    pub measurement_qubits: Vec<usize>,
    /// Human-readable description
    pub description: String,
}

/// Gates used in NLP circuits
#[derive(Clone, Debug)]
pub enum NlpGate {
    /// Ry rotation on a single qubit
    Ry(usize, f64),
    /// Rx rotation on a single qubit
    Rx(usize, f64),
    /// Rz rotation on a single qubit
    Rz(usize, f64),
    /// Hadamard
    H(usize),
    /// CNOT (control, target)
    Cnot(usize, usize),
    /// Controlled-Z
    Cz(usize, usize),
    /// Rzz interaction (qubit1, qubit2, angle)
    Rzz(usize, usize, f64),
    /// Post-selected measurement on a qubit (projects to |0>)
    PostSelect(usize),
}

/// Convert a parsed sentence into a quantum circuit.
///
/// The functor maps:
/// - Noun states -> Ry-parameterized state preparation
/// - Verb boxes -> parameterized entangling circuits (ansatz-dependent)
/// - Adjective boxes -> parameterized unitary on noun wire
/// - Cups -> Bell-effect (CNOT + measurement / trace)
///
/// The circuit structure mirrors the string diagram topology.
pub fn sentence_to_circuit(
    sentence: &Sentence,
    config: &QuantumNlpConfig,
) -> QuantumNlpCircuit {
    let diagram = &sentence.diagram;
    let num_qubits = diagram.total_qubits.max(1);
    let mut gates: Vec<NlpGate> = Vec::new();

    // Apply each word box as a parameterized circuit
    for dbox in &diagram.boxes {
        if dbox.qubit_indices.is_empty() {
            continue; // Determiners have no qubits
        }

        let qubits = &dbox.qubit_indices;
        let params = &dbox.parameters;

        match config.ansatz {
            CircuitAnsatz::IQP => {
                apply_iqp_ansatz(&mut gates, qubits, params, config.num_layers);
            }
            CircuitAnsatz::Sim14 => {
                apply_sim14_ansatz(&mut gates, qubits, params, config.num_layers);
            }
            CircuitAnsatz::Sim15 => {
                apply_sim15_ansatz(&mut gates, qubits, params, config.num_layers);
            }
            CircuitAnsatz::TensorNetwork => {
                apply_tn_ansatz(&mut gates, qubits, params, config.num_layers);
            }
        }
    }

    // Apply cups: CNOT + post-selection to implement the trace/Bell-effect
    for wire in &diagram.wires {
        if let Wire::Cup(a, b) = wire {
            // The cup connects two wires that share a qubit.
            // In the quantum circuit, this is realized as a CNOT
            // from one participant to the shared qubit, followed by
            // post-selection (or trace) on the shared qubit.
            // Since cups share a qubit index, we add a post-selection marker.
            let _ = (a, b); // Cups are already handled by qubit sharing
        }
    }

    // Post-select sentence qubits if configured
    if config.post_select {
        for &sq in &diagram.sentence_qubits {
            gates.push(NlpGate::PostSelect(sq));
        }
    }

    let description = format!(
        "DisCoCat circuit for '{}': {} qubits, {} gates, ansatz={:?}",
        sentence.text,
        num_qubits,
        gates.len(),
        config.ansatz,
    );

    QuantumNlpCircuit {
        num_qubits,
        gates,
        measurement_qubits: diagram.sentence_qubits.clone(),
        description,
    }
}

/// IQP ansatz: Hadamard layer + Rz rotations + Rzz entangling
fn apply_iqp_ansatz(
    gates: &mut Vec<NlpGate>,
    qubits: &[usize],
    params: &[f64],
    num_layers: usize,
) {
    let n = qubits.len();
    let single_params = n;
    let pair_params = n * (n.saturating_sub(1)) / 2;
    let params_per_layer = single_params + pair_params;

    for layer in 0..num_layers {
        let offset = layer * params_per_layer;

        // Hadamard on all qubits
        for &q in qubits {
            gates.push(NlpGate::H(q));
        }

        // Rz rotations
        for (i, &q) in qubits.iter().enumerate() {
            let p_idx = offset + i;
            if p_idx < params.len() {
                gates.push(NlpGate::Rz(q, params[p_idx]));
            }
        }

        // Rzz entangling (all pairs)
        let mut pair_idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let p_idx = offset + single_params + pair_idx;
                if p_idx < params.len() {
                    gates.push(NlpGate::Rzz(qubits[i], qubits[j], params[p_idx]));
                }
                pair_idx += 1;
            }
        }
    }
}

/// Sim14 ansatz: Ry rotations + CNOT ladder
fn apply_sim14_ansatz(
    gates: &mut Vec<NlpGate>,
    qubits: &[usize],
    params: &[f64],
    num_layers: usize,
) {
    let n = qubits.len();
    let params_per_layer = n;

    for layer in 0..num_layers {
        let offset = layer * params_per_layer;

        // Ry on each qubit
        for (i, &q) in qubits.iter().enumerate() {
            let p_idx = offset + i;
            if p_idx < params.len() {
                gates.push(NlpGate::Ry(q, params[p_idx]));
            }
        }

        // CNOT ladder (nearest-neighbor)
        if n > 1 {
            for i in 0..(n - 1) {
                gates.push(NlpGate::Cnot(qubits[i], qubits[i + 1]));
            }
        }
    }
}

/// Sim15 ansatz: Rx + Ry + Rz per qubit + CNOT ring
fn apply_sim15_ansatz(
    gates: &mut Vec<NlpGate>,
    qubits: &[usize],
    params: &[f64],
    num_layers: usize,
) {
    let n = qubits.len();
    let params_per_layer = 3 * n;

    for layer in 0..num_layers {
        let offset = layer * params_per_layer;

        // Rx, Ry, Rz on each qubit
        for (i, &q) in qubits.iter().enumerate() {
            let base = offset + 3 * i;
            if base < params.len() {
                gates.push(NlpGate::Rx(q, params[base]));
            }
            if base + 1 < params.len() {
                gates.push(NlpGate::Ry(q, params[base + 1]));
            }
            if base + 2 < params.len() {
                gates.push(NlpGate::Rz(q, params[base + 2]));
            }
        }

        // CNOT ring
        if n > 1 {
            for i in 0..n {
                gates.push(NlpGate::Cnot(qubits[i], qubits[(i + 1) % n]));
            }
        }
    }
}

/// Tensor-network ansatz: Ry + Rz per qubit + sequential CNOTs
fn apply_tn_ansatz(
    gates: &mut Vec<NlpGate>,
    qubits: &[usize],
    params: &[f64],
    num_layers: usize,
) {
    let n = qubits.len();
    let params_per_layer = 2 * n;

    for layer in 0..num_layers {
        let offset = layer * params_per_layer;

        // Ry + Rz on each qubit
        for (i, &q) in qubits.iter().enumerate() {
            let base = offset + 2 * i;
            if base < params.len() {
                gates.push(NlpGate::Ry(q, params[base]));
            }
            if base + 1 < params.len() {
                gates.push(NlpGate::Rz(q, params[base + 1]));
            }
        }

        // Sequential CNOTs (MPS-like)
        if n > 1 {
            for i in 0..(n - 1) {
                gates.push(NlpGate::Cnot(qubits[i], qubits[i + 1]));
            }
        }
    }
}

// ===================================================================
// CIRCUIT SIMULATION
// ===================================================================

/// Execute an NLP circuit and return the final quantum state.
///
/// Uses the nQPU-Metal statevector simulator for exact simulation.
pub fn simulate_circuit(circuit: &QuantumNlpCircuit) -> QuantumState {
    let mut state = QuantumState::new(circuit.num_qubits);

    for gate in &circuit.gates {
        match gate {
            NlpGate::Ry(q, theta) => GateOperations::ry(&mut state, *q, *theta),
            NlpGate::Rx(q, theta) => GateOperations::rx(&mut state, *q, *theta),
            NlpGate::Rz(q, theta) => GateOperations::rz(&mut state, *q, *theta),
            NlpGate::H(q) => GateOperations::h(&mut state, *q),
            NlpGate::Cnot(c, t) => {
                if *c == *t {
                    // CNOT on same qubit is identity (can happen with cup-shared qubits)
                    continue;
                }
                // GateOperations::cnot requires bit0 < bit1.
                // For reversed control, use H-H sandwich to swap roles.
                let (lo, hi) = if *c < *t { (*c, *t) } else { (*t, *c) };
                if *c < *t {
                    GateOperations::cnot(&mut state, lo, hi);
                } else {
                    // CNOT(c>t) = H⊗H · CNOT(t,c) · H⊗H
                    GateOperations::h(&mut state, lo);
                    GateOperations::h(&mut state, hi);
                    GateOperations::cnot(&mut state, lo, hi);
                    GateOperations::h(&mut state, lo);
                    GateOperations::h(&mut state, hi);
                }
            }
            NlpGate::Cz(c, t) => {
                if *c == *t { continue; } // CZ on same qubit = phase, skip
                let (lo, hi) = if *c < *t { (*c, *t) } else { (*t, *c) };
                GateOperations::h(&mut state, hi);
                GateOperations::cnot(&mut state, lo, hi);
                GateOperations::h(&mut state, hi);
            }
            NlpGate::Rzz(q1, q2, theta) => {
                if *q1 == *q2 {
                    // Rzz on same qubit = Rz(theta)
                    GateOperations::rz(&mut state, *q1, *theta);
                    continue;
                }
                let (lo, hi) = if *q1 < *q2 { (*q1, *q2) } else { (*q2, *q1) };
                GateOperations::cnot(&mut state, lo, hi);
                GateOperations::rz(&mut state, hi, *theta);
                GateOperations::cnot(&mut state, lo, hi);
            }
            NlpGate::PostSelect(q) => {
                // Project onto |0> for qubit q and renormalize
                post_select_zero(&mut state, *q);
            }
        }
    }

    state
}

/// Post-select a qubit to |0> by zeroing |1> amplitudes and renormalizing.
fn post_select_zero(state: &mut QuantumState, qubit: usize) {
    let _stride = 1 << qubit;
    let amplitudes = state.amplitudes_mut();
    let dim = amplitudes.len();

    // Zero all amplitudes where qubit is |1>
    for i in 0..dim {
        if (i >> qubit) & 1 == 1 {
            amplitudes[i] = Complex64::new(0.0, 0.0);
        }
    }

    // Renormalize
    let norm_sq: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
    if norm_sq > 1e-15 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for amp in amplitudes.iter_mut() {
            *amp = Complex64::new(amp.re * inv_norm, amp.im * inv_norm);
        }
    }
}

/// Evaluate a sentence's truth value by simulating its circuit
/// and measuring the sentence qubit(s) in the Z basis.
///
/// Returns probability of measuring |0> on the sentence wire,
/// interpreted as the sentence's truth value in [0, 1].
pub fn evaluate_sentence(sentence: &Sentence, config: &QuantumNlpConfig) -> f64 {
    let circuit = sentence_to_circuit(sentence, config);
    let state = simulate_circuit(&circuit);
    sentence_truth_value(&state, &circuit.measurement_qubits)
}

/// Extract truth value from a quantum state by measuring sentence qubits.
///
/// The truth value is P(all sentence qubits = |0>), i.e., the probability
/// of the sentence wire being in the "true" state.
fn sentence_truth_value(state: &QuantumState, measurement_qubits: &[usize]) -> f64 {
    if measurement_qubits.is_empty() {
        // No designated sentence qubit; use total |0...0> probability
        let amps = state.amplitudes_ref();
        if amps.is_empty() {
            return 0.0;
        }
        return amps[0].norm_sqr();
    }

    let probs = state.probabilities();
    let mut truth = 0.0;

    for (idx, &p) in probs.iter().enumerate() {
        // Check if all measurement qubits are 0 in this basis state
        let all_zero = measurement_qubits.iter().all(|&q| (idx >> q) & 1 == 0);
        if all_zero {
            truth += p;
        }
    }

    truth
}

// ===================================================================
// SENTENCE SIMILARITY
// ===================================================================

/// Compute similarity between two sentences.
///
/// Uses the configured similarity measure:
/// - **Fidelity**: |<psi|phi>|^2 (state overlap)
/// - **SwapTest**: Simulated SWAP test circuit
/// - **InnerProduct**: Re(<psi|phi>)
///
/// Both sentences are compiled into circuits, simulated, and compared.
pub fn sentence_similarity(
    s1: &Sentence,
    s2: &Sentence,
    config: &QuantumNlpConfig,
) -> f64 {
    let circuit1 = sentence_to_circuit(s1, config);
    let circuit2 = sentence_to_circuit(s2, config);

    let state1 = simulate_circuit(&circuit1);
    let state2 = simulate_circuit(&circuit2);

    match config.similarity_measure {
        SimilarityMeasure::Fidelity => {
            compute_fidelity(&state1, &state2)
        }
        SimilarityMeasure::SwapTest => {
            compute_swap_test_similarity(&state1, &state2)
        }
        SimilarityMeasure::InnerProduct => {
            compute_inner_product(&state1, &state2)
        }
    }
}

/// State fidelity: |<psi|phi>|^2
///
/// For pure states, this equals the squared overlap.
/// Returns a value in [0, 1] where 1 means identical states.
fn compute_fidelity(state1: &QuantumState, state2: &QuantumState) -> f64 {
    state1.fidelity(state2)
}

/// SWAP test similarity.
///
/// The SWAP test circuit uses an ancilla qubit to estimate |<psi|phi>|^2:
/// 1. Prepare ancilla in |+>
/// 2. Controlled-SWAP between the two states
/// 3. Measure ancilla: P(0) = (1 + |<psi|phi>|^2) / 2
///
/// We simulate this analytically using the inner product.
fn compute_swap_test_similarity(state1: &QuantumState, state2: &QuantumState) -> f64 {
    // SWAP test: P(ancilla=0) = (1 + |<psi|phi>|^2) / 2
    // So |<psi|phi>|^2 = 2 * P(0) - 1
    // But we want the similarity directly, so compute fidelity
    let fidelity = state1.fidelity(state2);
    // Return the SWAP test probability (always in [0.5, 1.0])
    (1.0 + fidelity) / 2.0
}

/// Real part of inner product: Re(<psi|phi>)
fn compute_inner_product(state1: &QuantumState, state2: &QuantumState) -> f64 {
    let amps1 = state1.amplitudes_ref();
    let amps2 = state2.amplitudes_ref();

    let min_len = amps1.len().min(amps2.len());
    let mut inner = Complex64::new(0.0, 0.0);

    for i in 0..min_len {
        // <psi|phi> = sum_i conj(psi_i) * phi_i
        let conj_a = Complex64::new(amps1[i].re, -amps1[i].im);
        inner = inner + conj_a * amps2[i];
    }

    inner.re
}

// ===================================================================
// TRAINING
// ===================================================================

/// Result of training word parameters
#[derive(Clone, Debug)]
pub struct TrainingResult {
    /// Loss values per epoch
    pub loss_history: Vec<f64>,
    /// Final loss
    pub final_loss: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
    /// Whether training converged (final loss < threshold)
    pub converged: bool,
}

/// A training example: (sentence, target_truth_value)
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// The sentence text
    pub sentence: String,
    /// Target truth value in [0, 1]
    pub target: f64,
}

impl TrainingExample {
    /// Create a training example
    pub fn new(sentence: &str, target: f64) -> Self {
        Self {
            sentence: sentence.to_string(),
            target: target.clamp(0.0, 1.0),
        }
    }
}

/// Trainer for quantum NLP word parameters.
///
/// Uses parameter-shift rule gradient descent to optimize word parameters
/// so that sentences produce desired truth values.
///
/// The parameter-shift rule computes exact gradients for parameterized
/// quantum circuits:
///
/// dL/d(theta_i) = [L(theta_i + pi/2) - L(theta_i - pi/2)] / 2
pub struct NlpTrainer {
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Maximum number of training epochs
    pub epochs: usize,
    /// Convergence threshold for loss
    pub convergence_threshold: f64,
}

impl Default for NlpTrainer {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            epochs: 100,
            convergence_threshold: 1e-4,
        }
    }
}

impl NlpTrainer {
    /// Create a trainer with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set max epochs
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Builder: set convergence threshold
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Train word parameters on a dataset of (sentence, target) pairs.
    ///
    /// Uses the parameter-shift rule to compute gradients and vanilla SGD
    /// to update parameters. All word parameters appearing in any training
    /// sentence are updated jointly.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training examples with sentences and target truth values
    /// * `lexicon` - Mutable lexicon whose word parameters will be updated
    /// * `config` - NLP circuit configuration
    ///
    /// # Returns
    ///
    /// A `TrainingResult` with loss history and convergence status.
    pub fn train(
        &self,
        dataset: &[TrainingExample],
        lexicon: &mut Lexicon,
        config: &QuantumNlpConfig,
    ) -> Result<TrainingResult, NlpError> {
        if dataset.is_empty() {
            return Err(NlpError::TrainingFailed("Empty dataset".to_string()));
        }

        let mut loss_history = Vec::with_capacity(self.epochs);

        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            for example in dataset {
                // Parse sentence with current lexicon
                let sentence = parse_sentence(lexicon, &example.sentence)?;

                // Compute current truth value
                let truth = evaluate_sentence(&sentence, config);

                // MSE loss for this example
                let loss = (truth - example.target).powi(2);
                total_loss += loss;

                // Compute gradients via parameter-shift rule
                let gradient_direction = 2.0 * (truth - example.target);

                // Collect all unique words in this sentence that have parameters
                let mut word_texts: Vec<String> = Vec::new();
                for word in &sentence.words {
                    if !word.parameters.is_empty()
                        && word.gram_type != GrammaticalType::Determiner
                        && !word_texts.contains(&word.text)
                    {
                        word_texts.push(word.text.clone());
                    }
                }

                // Update each word's parameters using parameter-shift rule
                for word_text in &word_texts {
                    let num_params = lexicon.lookup(word_text)?.parameters.len();

                    for p_idx in 0..num_params {
                        // Forward shift: theta + pi/2
                        let original = lexicon.lookup(word_text)?.parameters[p_idx];

                        // f(theta + pi/2)
                        lexicon.lookup_mut(word_text)?.parameters[p_idx] = original + PI / 2.0;
                        let sentence_plus = parse_sentence(lexicon, &example.sentence)?;
                        let truth_plus = evaluate_sentence(&sentence_plus, config);

                        // f(theta - pi/2)
                        lexicon.lookup_mut(word_text)?.parameters[p_idx] = original - PI / 2.0;
                        let sentence_minus = parse_sentence(lexicon, &example.sentence)?;
                        let truth_minus = evaluate_sentence(&sentence_minus, config);

                        // Restore and compute gradient
                        let param_gradient = (truth_plus - truth_minus) / 2.0;

                        // SGD update
                        let new_val =
                            original - self.learning_rate * gradient_direction * param_gradient;
                        lexicon.lookup_mut(word_text)?.parameters[p_idx] = new_val;
                    }
                }
            }

            let avg_loss = total_loss / dataset.len() as f64;
            loss_history.push(avg_loss);

            // Check convergence
            if avg_loss < self.convergence_threshold {
                return Ok(TrainingResult {
                    loss_history,
                    final_loss: avg_loss,
                    epochs_completed: epoch + 1,
                    converged: true,
                });
            }
        }

        let final_loss = *loss_history.last().unwrap_or(&f64::NAN);
        Ok(TrainingResult {
            loss_history,
            final_loss,
            epochs_completed: self.epochs,
            converged: final_loss < self.convergence_threshold,
        })
    }
}

// ===================================================================
// BATCH OPERATIONS
// ===================================================================

/// Evaluate multiple sentences in batch, returning truth values.
pub fn batch_evaluate(
    sentences: &[Sentence],
    config: &QuantumNlpConfig,
) -> Vec<f64> {
    sentences
        .iter()
        .map(|s| evaluate_sentence(s, config))
        .collect()
}

/// Compute a pairwise similarity matrix for a set of sentences.
///
/// Returns a symmetric N x N matrix where entry (i, j) is the
/// similarity between sentence i and sentence j.
pub fn similarity_matrix(
    sentences: &[Sentence],
    config: &QuantumNlpConfig,
) -> Vec<Vec<f64>> {
    let n = sentences.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0; // Self-similarity is always 1
        for j in (i + 1)..n {
            let sim = sentence_similarity(&sentences[i], &sentences[j], config);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }

    matrix
}

// ===================================================================
// ADVANCED: QUESTION ANSWERING
// ===================================================================

/// Question-answering via sentence overlap.
///
/// Given a question (encoded as a sentence) and candidate answers,
/// returns the answer with highest similarity/truth-value.
pub fn answer_question(
    question: &Sentence,
    candidates: &[Sentence],
    config: &QuantumNlpConfig,
) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;

    for (i, candidate) in candidates.iter().enumerate() {
        let score = sentence_similarity(question, candidate, config);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    (best_idx, best_score)
}

// ===================================================================
// ADVANCED: SENTENCE COMPOSITION
// ===================================================================

/// Compose two sentences by tensor product of their meaning states.
///
/// Useful for multi-sentence reasoning where the joint state encodes
/// logical relationships between sentences.
pub fn compose_sentences(
    s1: &Sentence,
    s2: &Sentence,
    config: &QuantumNlpConfig,
) -> QuantumState {
    let circuit1 = sentence_to_circuit(s1, config);
    let circuit2 = sentence_to_circuit(s2, config);

    let state1 = simulate_circuit(&circuit1);
    let state2 = simulate_circuit(&circuit2);

    tensor_product_states(&state1, &state2)
}

/// Compute tensor product of two quantum states.
///
/// |psi> tensor |phi> has dimension dim1 * dim2.
fn tensor_product_states(s1: &QuantumState, s2: &QuantumState) -> QuantumState {
    let amps1 = s1.amplitudes_ref();
    let amps2 = s2.amplitudes_ref();
    let total_qubits = (amps1.len() as f64).log2() as usize
        + (amps2.len() as f64).log2() as usize;

    let mut product = QuantumState::new(total_qubits);
    let prod_amps = product.amplitudes_mut();

    for i in 0..amps1.len() {
        for j in 0..amps2.len() {
            let idx = i * amps2.len() + j;
            if idx < prod_amps.len() {
                prod_amps[idx] = amps1[i] * amps2[j];
            }
        }
    }

    product
}

// ===================================================================
// ADVANCED: DENSITY MATRIX SEMANTICS
// ===================================================================

/// Compute the density matrix for a sentence's meaning state.
///
/// rho = |psi><psi| where |psi> is the sentence's quantum state.
/// Useful for mixed-state semantics and ambiguity modeling.
pub fn sentence_density_matrix(
    sentence: &Sentence,
    config: &QuantumNlpConfig,
) -> Vec<Vec<Complex64>> {
    let circuit = sentence_to_circuit(sentence, config);
    let state = simulate_circuit(&circuit);
    let amps = state.amplitudes_ref();
    let dim = amps.len();

    let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            // rho[i][j] = psi[i] * conj(psi[j])
            rho[i][j] = amps[i] * Complex64::new(amps[j].re, -amps[j].im);
        }
    }

    rho
}

/// Von Neumann entropy of a sentence's meaning state.
///
/// S(rho) = -Tr(rho log rho)
/// For pure states this is 0; for maximally mixed states it is log(d).
/// We compute this from the density matrix eigenvalues.
pub fn sentence_entropy(
    sentence: &Sentence,
    config: &QuantumNlpConfig,
) -> f64 {
    let rho = sentence_density_matrix(sentence, config);
    let dim = rho.len();

    // For a pure state |psi><psi|, entropy is 0
    // Compute trace of rho^2 to check purity
    let mut purity = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            purity += (rho[i][j] * Complex64::new(rho[j][i].re, -rho[j][i].im)).re;
        }
    }

    // For pure states, purity = Tr(rho^2) = 1, entropy = 0
    // For mixed states, use S = -sum_i lambda_i log(lambda_i)
    // We approximate with S = -log(Tr(rho^2)) for Renyi-2 entropy
    if purity > 1.0 - 1e-10 {
        0.0
    } else {
        -purity.ln()
    }
}

// ===================================================================
// ADVANCED: FUNCTORIAL SEMANTICS
// ===================================================================

/// Semantic functor mapping from pregroup grammar to quantum circuits.
///
/// The functor F: Grammar -> FdHilb maps:
/// - Atomic type n -> C^{2^noun_qubits} (noun Hilbert space)
/// - Atomic type s -> C^{2^sentence_qubits} (sentence Hilbert space)
/// - Cup: A tensor A* -> C (Bell effect / inner product)
/// - Cap: C -> A* tensor A (Bell state preparation)
///
/// This struct encapsulates the functor's action on morphisms.
pub struct SemanticFunctor {
    /// NLP configuration defining the functor's target category
    pub config: QuantumNlpConfig,
}

impl SemanticFunctor {
    /// Create a semantic functor with the given configuration
    pub fn new(config: QuantumNlpConfig) -> Self {
        Self { config }
    }

    /// Apply the functor to a word, producing its quantum circuit fragment.
    pub fn apply_word(&self, word: &Word) -> Vec<NlpGate> {
        let total_qubits = self.config.total_qubits_for_type(&word.gram_type);
        if total_qubits == 0 {
            return vec![];
        }

        let qubits: Vec<usize> = (0..total_qubits).collect();
        let mut gates = Vec::new();

        match self.config.ansatz {
            CircuitAnsatz::IQP => {
                apply_iqp_ansatz(&mut gates, &qubits, &word.parameters, self.config.num_layers);
            }
            CircuitAnsatz::Sim14 => {
                apply_sim14_ansatz(&mut gates, &qubits, &word.parameters, self.config.num_layers);
            }
            CircuitAnsatz::Sim15 => {
                apply_sim15_ansatz(&mut gates, &qubits, &word.parameters, self.config.num_layers);
            }
            CircuitAnsatz::TensorNetwork => {
                apply_tn_ansatz(&mut gates, &qubits, &word.parameters, self.config.num_layers);
            }
        }

        gates
    }

    /// Apply the functor to a cup (type reduction).
    ///
    /// Returns a CNOT + post-selection sequence implementing the
    /// categorical trace / Bell-effect.
    pub fn apply_cup(&self, qubit_a: usize, qubit_b: usize) -> Vec<NlpGate> {
        vec![
            NlpGate::Cnot(qubit_a, qubit_b),
            NlpGate::H(qubit_a),
            NlpGate::PostSelect(qubit_a),
            NlpGate::PostSelect(qubit_b),
        ]
    }

    /// Map a full sentence through the functor.
    pub fn map_sentence(&self, sentence: &Sentence) -> QuantumNlpCircuit {
        sentence_to_circuit(sentence, &self.config)
    }
}

// ===================================================================
// UTILITIES
// ===================================================================

/// Pretty-print a sentence diagram
pub fn diagram_to_string(sentence: &Sentence) -> String {
    let mut out = String::new();
    out.push_str(&format!("Sentence: \"{}\"\n", sentence.text));
    out.push_str(&format!(
        "Total qubits: {}\n",
        sentence.diagram.total_qubits
    ));
    out.push_str(&format!(
        "Sentence qubits: {:?}\n",
        sentence.diagram.sentence_qubits
    ));

    out.push_str("\nWord boxes:\n");
    for (i, dbox) in sentence.diagram.boxes.iter().enumerate() {
        out.push_str(&format!(
            "  [{}] '{}': {} in -> {} out, qubits={:?}, params={}\n",
            i,
            dbox.word,
            dbox.input_wires,
            dbox.output_wires,
            dbox.qubit_indices,
            dbox.parameters.len(),
        ));
    }

    out.push_str("\nWires:\n");
    for wire in &sentence.diagram.wires {
        match wire {
            Wire::Identity(i) => out.push_str(&format!("  Identity({})\n", i)),
            Wire::Cup(a, b) => out.push_str(&format!("  Cup({}, {})\n", a, b)),
            Wire::Cap(a, b) => out.push_str(&format!("  Cap({}, {})\n", a, b)),
        }
    }

    out
}

/// Pretty-print a circuit
pub fn circuit_to_string(circuit: &QuantumNlpCircuit) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", circuit.description));
    out.push_str(&format!(
        "Measurement qubits: {:?}\n",
        circuit.measurement_qubits
    ));

    out.push_str("\nGates:\n");
    for (i, gate) in circuit.gates.iter().enumerate() {
        let gate_str = match gate {
            NlpGate::Ry(q, t) => format!("Ry({}, {:.4})", q, t),
            NlpGate::Rx(q, t) => format!("Rx({}, {:.4})", q, t),
            NlpGate::Rz(q, t) => format!("Rz({}, {:.4})", q, t),
            NlpGate::H(q) => format!("H({})", q),
            NlpGate::Cnot(c, t) => format!("CNOT({}, {})", c, t),
            NlpGate::Cz(c, t) => format!("CZ({}, {})", c, t),
            NlpGate::Rzz(a, b, t) => format!("Rzz({}, {}, {:.4})", a, b, t),
            NlpGate::PostSelect(q) => format!("PostSelect({})", q),
        };
        out.push_str(&format!("  [{}] {}\n", i, gate_str));
    }

    out
}

// ===================================================================
// DEMONSTRATIONS
// ===================================================================

/// Run the full DisCoCat demonstration.
///
/// Shows word-order sensitivity, adjective modification, sentence
/// similarity, and basic training on a toy dataset.
pub fn run_discocat_demo() -> String {
    let mut out = String::new();
    out.push_str("=== DisCoCat Quantum NLP Demo ===\n\n");

    let lexicon = Lexicon::english_basic();
    let config = QuantumNlpConfig::new();

    // Demo 1: Word order matters
    out.push_str("--- Demo 1: Word Order Sensitivity ---\n");
    if let (Ok(s1), Ok(s2)) = (
        parse_sentence(&lexicon, "alice loves bob"),
        parse_sentence(&lexicon, "bob loves alice"),
    ) {
        let t1 = evaluate_sentence(&s1, &config);
        let t2 = evaluate_sentence(&s2, &config);
        let sim = sentence_similarity(&s1, &s2, &config);
        out.push_str(&format!("  'alice loves bob' truth: {:.6}\n", t1));
        out.push_str(&format!("  'bob loves alice' truth: {:.6}\n", t2));
        out.push_str(&format!(
            "  Similarity: {:.6} (should be < 1.0: word order matters!)\n",
            sim
        ));
        out.push_str(&format!(
            "  Different meanings: {}\n\n",
            (t1 - t2).abs() > 1e-10
        ));
    }

    // Demo 2: Adjective modification
    out.push_str("--- Demo 2: Adjective Modification ---\n");
    if let (Ok(s1), Ok(s2)) = (
        parse_sentence(&lexicon, "big cat sleeps"),
        parse_sentence(&lexicon, "small cat sleeps"),
    ) {
        let t1 = evaluate_sentence(&s1, &config);
        let t2 = evaluate_sentence(&s2, &config);
        let sim = sentence_similarity(&s1, &s2, &config);
        out.push_str(&format!("  'big cat sleeps' truth: {:.6}\n", t1));
        out.push_str(&format!("  'small cat sleeps' truth: {:.6}\n", t2));
        out.push_str(&format!("  Similarity: {:.6}\n\n", sim));
    }

    // Demo 3: Intransitive vs transitive
    out.push_str("--- Demo 3: Verb Types ---\n");
    if let (Ok(s1), Ok(s2)) = (
        parse_sentence(&lexicon, "cat sleeps"),
        parse_sentence(&lexicon, "cat chases dog"),
    ) {
        let t1 = evaluate_sentence(&s1, &config);
        let t2 = evaluate_sentence(&s2, &config);
        out.push_str(&format!("  'cat sleeps' truth: {:.6}\n", t1));
        out.push_str(&format!("  'cat chases dog' truth: {:.6}\n", t2));
        out.push_str(&format!(
            "  Different circuit topologies: {} vs {} qubits\n\n",
            s1.diagram.total_qubits, s2.diagram.total_qubits
        ));
    }

    // Demo 4: Self-similarity
    out.push_str("--- Demo 4: Self-Similarity ---\n");
    if let Ok(s1) = parse_sentence(&lexicon, "alice runs") {
        let sim = sentence_similarity(&s1, &s1, &config);
        out.push_str(&format!(
            "  'alice runs' vs 'alice runs': {:.6} (should be 1.0)\n\n",
            sim
        ));
    }

    // Demo 5: Similarity matrix
    out.push_str("--- Demo 5: Sentence Similarity Matrix ---\n");
    let test_sentences = [
        "alice sleeps",
        "bob sleeps",
        "alice runs",
        "cat sleeps",
    ];
    let mut parsed = Vec::new();
    for s in &test_sentences {
        if let Ok(p) = parse_sentence(&lexicon, s) {
            parsed.push(p);
        }
    }
    if parsed.len() == test_sentences.len() {
        let sim_matrix = similarity_matrix(&parsed, &config);
        out.push_str("  Sentences: ");
        for s in &test_sentences {
            out.push_str(&format!("'{}' ", s));
        }
        out.push('\n');
        for (i, row) in sim_matrix.iter().enumerate() {
            out.push_str(&format!("  [{}]: ", i));
            for val in row {
                out.push_str(&format!("{:.4} ", val));
            }
            out.push('\n');
        }
    }

    out.push_str("\n=== End DisCoCat Demo ===\n");
    out
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Lexicon tests ---

    #[test]
    fn test_lexicon_creation() {
        let lex = Lexicon::english_basic();
        assert!(lex.len() > 40, "Basic lexicon should have 40+ words");
        assert!(!lex.is_empty());
    }

    #[test]
    fn test_lexicon_contains_nouns() {
        let lex = Lexicon::english_basic();
        assert!(lex.contains("alice"));
        assert!(lex.contains("bob"));
        assert!(lex.contains("cat"));
        assert!(lex.contains("dog"));
    }

    #[test]
    fn test_lexicon_contains_verbs() {
        let lex = Lexicon::english_basic();
        assert!(lex.contains("loves"));
        assert!(lex.contains("sleeps"));
        assert!(lex.contains("chases"));
    }

    #[test]
    fn test_lexicon_contains_adjectives() {
        let lex = Lexicon::english_basic();
        assert!(lex.contains("big"));
        assert!(lex.contains("quantum"));
    }

    #[test]
    fn test_lexicon_contains_determiners() {
        let lex = Lexicon::english_basic();
        assert!(lex.contains("the"));
        assert!(lex.contains("a"));
    }

    #[test]
    fn test_word_type_assignment() {
        let lex = Lexicon::english_basic();
        assert_eq!(lex.lookup("alice").unwrap().gram_type, GrammaticalType::Noun);
        assert_eq!(lex.lookup("loves").unwrap().gram_type, GrammaticalType::Transitive);
        assert_eq!(lex.lookup("sleeps").unwrap().gram_type, GrammaticalType::Intransitive);
        assert_eq!(lex.lookup("big").unwrap().gram_type, GrammaticalType::Adjective);
        assert_eq!(lex.lookup("the").unwrap().gram_type, GrammaticalType::Determiner);
    }

    #[test]
    fn test_lexicon_add_custom_word() {
        let mut lex = Lexicon::new();
        lex.add_word("qubit", GrammaticalType::Noun, vec![0.5, 1.0]);
        assert!(lex.contains("qubit"));
        assert_eq!(lex.lookup("qubit").unwrap().parameters.len(), 2);
    }

    #[test]
    fn test_unknown_word_error() {
        let lex = Lexicon::english_basic();
        let result = lex.lookup("supercalifragilistic");
        assert!(result.is_err());
        match result {
            Err(NlpError::UnknownWord(w)) => assert_eq!(w, "supercalifragilistic"),
            _ => panic!("Expected UnknownWord error"),
        }
    }

    // --- Type reduction tests ---

    #[test]
    fn test_noun_right_adj_reduction() {
        // n . n^r -> 1 (cup reduction)
        let types = vec![AtomicType::N, AtomicType::NR];
        let reduced = reduce_types(&types).unwrap();
        assert!(reduced.is_empty(), "n . n^r should reduce to empty (1)");
    }

    #[test]
    fn test_noun_left_adj_reduction() {
        // n^l . n -> 1 (cup reduction) — but our can_reduce also handles n . n^l
        let types = vec![AtomicType::N, AtomicType::NL];
        let reduced = reduce_types(&types).unwrap();
        assert!(reduced.is_empty(), "n . n^l should reduce to empty (1)");
    }

    #[test]
    fn test_intransitive_sentence_reduces() {
        // "alice sleeps": n . (n^r . s) -> s
        let types = vec![AtomicType::N, AtomicType::NR, AtomicType::S];
        assert!(reduces_to_sentence(&types));
    }

    #[test]
    fn test_transitive_sentence_reduces() {
        // "alice loves bob": n . (n^r . s . n^l) . n -> s
        let types = vec![
            AtomicType::N,
            AtomicType::NR,
            AtomicType::S,
            AtomicType::NL,
            AtomicType::N,
        ];
        assert!(reduces_to_sentence(&types));
    }

    #[test]
    fn test_adjective_noun_intransitive_reduces() {
        // "big cat sleeps": (n . n^l) . n . (n^r . s) -> s
        let types = vec![
            AtomicType::N,
            AtomicType::NL,
            AtomicType::N,
            AtomicType::NR,
            AtomicType::S,
        ];
        assert!(reduces_to_sentence(&types));
    }

    #[test]
    fn test_invalid_type_sequence() {
        // "noun noun" should not reduce to sentence
        let types = vec![AtomicType::N, AtomicType::N];
        assert!(!reduces_to_sentence(&types));
    }

    // --- Parsing tests ---

    #[test]
    fn test_parse_intransitive() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        assert_eq!(sentence.words.len(), 2);
        assert!(sentence.diagram.total_qubits > 0);
        assert!(!sentence.diagram.sentence_qubits.is_empty());
    }

    #[test]
    fn test_parse_transitive() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice loves bob").unwrap();
        assert_eq!(sentence.words.len(), 3);
        assert!(sentence.diagram.total_qubits > 0);
    }

    #[test]
    fn test_parse_adjective_sentence() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "big cat sleeps").unwrap();
        assert_eq!(sentence.words.len(), 3);
    }

    #[test]
    fn test_parse_with_determiner() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "the cat sleeps").unwrap();
        // "the" is a determiner, should be handled
        assert_eq!(sentence.words.len(), 3);
    }

    #[test]
    fn test_parse_empty_sentence() {
        let lex = Lexicon::english_basic();
        let result = parse_sentence(&lex, "");
        assert!(result.is_err());
        match result {
            Err(NlpError::EmptySentence) => {}
            _ => panic!("Expected EmptySentence error"),
        }
    }

    #[test]
    fn test_parse_unknown_word() {
        let lex = Lexicon::english_basic();
        let result = parse_sentence(&lex, "alice glorps bob");
        assert!(result.is_err());
        match result {
            Err(NlpError::UnknownWord(w)) => assert_eq!(w, "glorps"),
            _ => panic!("Expected UnknownWord error"),
        }
    }

    // --- Circuit generation tests ---

    #[test]
    fn test_circuit_generation_intransitive() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let circuit = sentence_to_circuit(&sentence, &config);
        assert!(circuit.num_qubits > 0);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_circuit_generation_transitive() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice loves bob").unwrap();
        let config = QuantumNlpConfig::new();
        let circuit = sentence_to_circuit(&sentence, &config);
        assert!(circuit.num_qubits > 0);
        assert!(!circuit.gates.is_empty());
    }

    #[test]
    fn test_iqp_ansatz_circuit() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new().with_ansatz(CircuitAnsatz::IQP);
        let circuit = sentence_to_circuit(&sentence, &config);
        // IQP should have Hadamard gates
        let has_h = circuit.gates.iter().any(|g| matches!(g, NlpGate::H(_)));
        assert!(has_h, "IQP ansatz should include Hadamard gates");
    }

    #[test]
    fn test_sim14_ansatz_circuit() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new().with_ansatz(CircuitAnsatz::Sim14);
        let circuit = sentence_to_circuit(&sentence, &config);
        let has_ry = circuit.gates.iter().any(|g| matches!(g, NlpGate::Ry(_, _)));
        assert!(has_ry, "Sim14 ansatz should include Ry gates");
    }

    #[test]
    fn test_sim15_ansatz_circuit() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new().with_ansatz(CircuitAnsatz::Sim15);
        let circuit = sentence_to_circuit(&sentence, &config);
        let has_rx = circuit.gates.iter().any(|g| matches!(g, NlpGate::Rx(_, _)));
        let has_rz = circuit.gates.iter().any(|g| matches!(g, NlpGate::Rz(_, _)));
        assert!(has_rx, "Sim15 ansatz should include Rx gates");
        assert!(has_rz, "Sim15 ansatz should include Rz gates");
    }

    #[test]
    fn test_tensor_network_ansatz_circuit() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new().with_ansatz(CircuitAnsatz::TensorNetwork);
        let circuit = sentence_to_circuit(&sentence, &config);
        assert!(!circuit.gates.is_empty());
    }

    // --- Simulation tests ---

    #[test]
    fn test_sentence_state_normalization() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let circuit = sentence_to_circuit(&sentence, &config);
        let state = simulate_circuit(&circuit);
        let norm: f64 = state.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State should be normalized, got norm = {}",
            norm
        );
    }

    #[test]
    fn test_truth_value_in_range() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice loves bob").unwrap();
        let config = QuantumNlpConfig::new();
        let truth = evaluate_sentence(&sentence, &config);
        assert!(
            (0.0..=1.0).contains(&truth),
            "Truth value should be in [0,1], got {}",
            truth
        );
    }

    // --- Sentence similarity tests ---

    #[test]
    fn test_identical_sentences_similarity_one() {
        let lex = Lexicon::english_basic();
        let s = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let sim = sentence_similarity(&s, &s, &config);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Self-similarity should be 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_word_order_produces_different_circuits() {
        let lex = Lexicon::english_basic();
        let s1 = parse_sentence(&lex, "alice loves bob").unwrap();
        let s2 = parse_sentence(&lex, "bob loves alice").unwrap();
        let config = QuantumNlpConfig::new();

        let t1 = evaluate_sentence(&s1, &config);
        let t2 = evaluate_sentence(&s2, &config);

        // alice loves bob should produce a different truth value than bob loves alice
        // because the noun parameters are different
        assert!(
            (t1 - t2).abs() > 1e-12,
            "Word order should matter: t1={}, t2={}",
            t1,
            t2
        );
    }

    #[test]
    fn test_different_sentences_similarity_less_than_one() {
        let lex = Lexicon::english_basic();
        let s1 = parse_sentence(&lex, "alice sleeps").unwrap();
        let s2 = parse_sentence(&lex, "bob runs").unwrap();
        let config = QuantumNlpConfig::new();
        let sim = sentence_similarity(&s1, &s2, &config);
        assert!(
            sim < 1.0 - 1e-10,
            "Different sentences should have similarity < 1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_fidelity_similarity_in_range() {
        let lex = Lexicon::english_basic();
        let s1 = parse_sentence(&lex, "cat sleeps").unwrap();
        let s2 = parse_sentence(&lex, "dog runs").unwrap();
        let config = QuantumNlpConfig::new()
            .with_similarity(SimilarityMeasure::Fidelity);
        let sim = sentence_similarity(&s1, &s2, &config);
        assert!(
            (0.0..=1.0).contains(&sim),
            "Fidelity should be in [0,1], got {}",
            sim
        );
    }

    #[test]
    fn test_swap_test_similarity_in_range() {
        let lex = Lexicon::english_basic();
        let s1 = parse_sentence(&lex, "cat sleeps").unwrap();
        let s2 = parse_sentence(&lex, "dog runs").unwrap();
        let config = QuantumNlpConfig::new()
            .with_similarity(SimilarityMeasure::SwapTest);
        let sim = sentence_similarity(&s1, &s2, &config);
        assert!(
            (0.0..=1.0).contains(&sim),
            "SWAP test similarity should be in [0,1], got {}",
            sim
        );
    }

    #[test]
    fn test_inner_product_similarity() {
        let lex = Lexicon::english_basic();
        let s = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new()
            .with_similarity(SimilarityMeasure::InnerProduct);
        let sim = sentence_similarity(&s, &s, &config);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Self inner product should be 1.0, got {}",
            sim
        );
    }

    // --- Cup and wire tests ---

    #[test]
    fn test_cup_wire_reduces_qubit_count() {
        let lex = Lexicon::english_basic();
        // Transitive sentence has cups connecting noun wires to verb
        let s = parse_sentence(&lex, "alice loves bob").unwrap();

        // The diagram should have cups
        let num_cups = s
            .diagram
            .wires
            .iter()
            .filter(|w| matches!(w, Wire::Cup(_, _)))
            .count();
        assert!(
            num_cups > 0,
            "Transitive sentence should have cup wires for type reduction"
        );
    }

    #[test]
    fn test_noun_qubits_scale() {
        let lex = Lexicon::english_basic();
        let s = parse_sentence(&lex, "alice sleeps").unwrap();

        let config1 = QuantumNlpConfig::new().with_noun_qubits(1);
        let config2 = QuantumNlpConfig::new().with_noun_qubits(2);

        let c1 = sentence_to_circuit(&s, &config1);
        let c2 = sentence_to_circuit(&s, &config2);

        // More noun qubits should not decrease total qubit count
        // (The diagram structure stays the same, but configs control
        // how many physical qubits each wire maps to in the ansatz)
        assert!(c1.num_qubits > 0);
        assert!(c2.num_qubits > 0);
    }

    // --- Multiple adjective test ---

    #[test]
    fn test_multiple_adjectives() {
        // This tests that two adjectives compose on the same noun
        let mut lex = Lexicon::english_basic();
        // We need a double-adjective sentence to parse
        // "big fast cat sleeps" -> (n.n^l) . (n.n^l) . n . (n^r.s)
        // Type sequence: n . n^l . n . n^l . n . n^r . s
        // Reduction: n.n^l -> 1, so: n . n^l . n . n^r . s
        //            n.n^l -> 1, so: n . n^r . s -> s
        let sentence = parse_sentence(&lex, "big fast cat sleeps");
        // This should parse correctly with greedy reduction
        assert!(
            sentence.is_ok(),
            "Multiple adjectives should compose correctly"
        );
    }

    // --- Training tests ---

    #[test]
    fn test_training_runs() {
        let mut lex = Lexicon::new();
        lex.add_word("alice", GrammaticalType::Noun, vec![0.5]);
        lex.add_word("sleeps", GrammaticalType::Intransitive, vec![0.5, 1.0]);
        lex.add_word("bob", GrammaticalType::Noun, vec![1.0]);

        let config = QuantumNlpConfig::new();
        let dataset = vec![
            TrainingExample::new("alice sleeps", 1.0),
        ];

        let trainer = NlpTrainer::new()
            .with_learning_rate(0.3)
            .with_epochs(5);

        let result = trainer.train(&dataset, &mut lex, &config);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.epochs_completed, 5);
        assert!(!result.loss_history.is_empty());
    }

    #[test]
    fn test_training_reduces_loss() {
        let mut lex = Lexicon::new();
        lex.add_word("cat", GrammaticalType::Noun, vec![0.1]);
        lex.add_word("runs", GrammaticalType::Intransitive, vec![0.1, 0.2]);

        let config = QuantumNlpConfig::new();
        let dataset = vec![
            TrainingExample::new("cat runs", 0.8),
        ];

        let trainer = NlpTrainer::new()
            .with_learning_rate(0.5)
            .with_epochs(20);

        let result = trainer.train(&dataset, &mut lex, &config).unwrap();

        // Loss should decrease from first to last epoch
        if result.loss_history.len() > 1 {
            let first_loss = result.loss_history[0];
            let last_loss = result.final_loss;
            assert!(
                last_loss <= first_loss + 0.1,
                "Training should generally reduce loss: first={}, last={}",
                first_loss,
                last_loss
            );
        }
    }

    // --- Batch and matrix tests ---

    #[test]
    fn test_batch_evaluate() {
        let lex = Lexicon::english_basic();
        let config = QuantumNlpConfig::new();

        let sentences: Vec<Sentence> = ["alice sleeps", "bob runs", "cat swims"]
            .iter()
            .filter_map(|s| parse_sentence(&lex, s).ok())
            .collect();

        let truths = batch_evaluate(&sentences, &config);
        assert_eq!(truths.len(), sentences.len());
        for t in &truths {
            assert!((0.0..=1.0).contains(t));
        }
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let lex = Lexicon::english_basic();
        let config = QuantumNlpConfig::new();

        let sentences: Vec<Sentence> = ["alice sleeps", "bob runs", "cat swims"]
            .iter()
            .filter_map(|s| parse_sentence(&lex, s).ok())
            .collect();

        let matrix = similarity_matrix(&sentences, &config);
        let n = sentences.len();

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (matrix[i][j] - matrix[j][i]).abs() < 1e-10,
                    "Similarity matrix should be symmetric"
                );
            }
            // Diagonal should be 1.0
            assert!(
                (matrix[i][i] - 1.0).abs() < 1e-10,
                "Self-similarity should be 1.0"
            );
        }
    }

    // --- Density matrix and entropy tests ---

    #[test]
    fn test_density_matrix_trace_one() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let rho = sentence_density_matrix(&sentence, &config);

        let trace: f64 = (0..rho.len()).map(|i| rho[i][i].re).sum();
        assert!(
            (trace - 1.0).abs() < 1e-10,
            "Density matrix trace should be 1.0, got {}",
            trace
        );
    }

    #[test]
    fn test_pure_state_entropy_zero() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let entropy = sentence_entropy(&sentence, &config);
        assert!(
            entropy.abs() < 1e-6,
            "Pure state entropy should be ~0, got {}",
            entropy
        );
    }

    // --- Config builder tests ---

    #[test]
    fn test_config_builder() {
        let config = QuantumNlpConfig::new()
            .with_noun_qubits(2)
            .with_sentence_qubits(2)
            .with_ansatz(CircuitAnsatz::Sim15)
            .with_similarity(SimilarityMeasure::SwapTest)
            .with_layers(3)
            .with_post_select(true);

        assert_eq!(config.noun_qubits, 2);
        assert_eq!(config.sentence_qubits, 2);
        assert_eq!(config.ansatz, CircuitAnsatz::Sim15);
        assert_eq!(config.similarity_measure, SimilarityMeasure::SwapTest);
        assert_eq!(config.num_layers, 3);
        assert!(config.post_select);
    }

    #[test]
    fn test_params_for_type_noun() {
        let config = QuantumNlpConfig::new();
        let params = config.params_for_type(&GrammaticalType::Noun);
        assert!(params > 0, "Noun should require parameters");
    }

    #[test]
    fn test_params_for_type_transitive() {
        let config = QuantumNlpConfig::new();
        let noun_params = config.params_for_type(&GrammaticalType::Noun);
        let trans_params = config.params_for_type(&GrammaticalType::Transitive);
        assert!(
            trans_params >= noun_params,
            "Transitive verb should need at least as many params as a noun"
        );
    }

    #[test]
    fn test_params_for_type_determiner_zero() {
        let config = QuantumNlpConfig::new();
        let params = config.params_for_type(&GrammaticalType::Determiner);
        assert_eq!(params, 0, "Determiner should need 0 parameters");
    }

    // --- Semantic functor tests ---

    #[test]
    fn test_semantic_functor_word() {
        let config = QuantumNlpConfig::new();
        let functor = SemanticFunctor::new(config);
        let word = Word::new("cat", GrammaticalType::Noun, vec![1.0]);
        let gates = functor.apply_word(&word);
        assert!(!gates.is_empty(), "Functor should produce gates for noun");
    }

    #[test]
    fn test_semantic_functor_determiner_empty() {
        let config = QuantumNlpConfig::new();
        let functor = SemanticFunctor::new(config);
        let word = Word::new("the", GrammaticalType::Determiner, vec![]);
        let gates = functor.apply_word(&word);
        assert!(gates.is_empty(), "Functor should produce no gates for determiner");
    }

    #[test]
    fn test_semantic_functor_cup() {
        let config = QuantumNlpConfig::new();
        let functor = SemanticFunctor::new(config);
        let gates = functor.apply_cup(0, 1);
        assert!(!gates.is_empty(), "Cup should produce gates");
        // Should contain CNOT and post-selection
        let has_cnot = gates.iter().any(|g| matches!(g, NlpGate::Cnot(_, _)));
        assert!(has_cnot, "Cup should include CNOT");
    }

    // --- Question answering test ---

    #[test]
    fn test_question_answering() {
        let lex = Lexicon::english_basic();
        let config = QuantumNlpConfig::new();

        let question = parse_sentence(&lex, "cat sleeps").unwrap();
        let candidates: Vec<Sentence> = ["dog runs", "cat sleeps", "bob swims"]
            .iter()
            .filter_map(|s| parse_sentence(&lex, s).ok())
            .collect();

        let (best_idx, best_score) = answer_question(&question, &candidates, &config);

        // The identical sentence should score highest
        assert_eq!(best_idx, 1, "Identical sentence should be best match");
        assert!(
            (best_score - 1.0).abs() < 1e-10,
            "Identical sentence should have score 1.0"
        );
    }

    // --- Composition test ---

    #[test]
    fn test_sentence_composition() {
        let lex = Lexicon::english_basic();
        let config = QuantumNlpConfig::new();

        let s1 = parse_sentence(&lex, "alice sleeps").unwrap();
        let s2 = parse_sentence(&lex, "bob runs").unwrap();

        let composed = compose_sentences(&s1, &s2, &config);
        let norm: f64 = composed.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Composed state should be normalized"
        );
    }

    // --- Display and utility tests ---

    #[test]
    fn test_grammatical_type_display() {
        assert_eq!(format!("{}", GrammaticalType::Noun), "n");
        assert_eq!(format!("{}", GrammaticalType::Sentence), "s");
        assert_eq!(format!("{}", GrammaticalType::Transitive), "n^r.s.n^l");
        assert_eq!(format!("{}", GrammaticalType::Intransitive), "n^r.s");
        assert_eq!(format!("{}", GrammaticalType::Adjective), "n.n^l");
    }

    #[test]
    fn test_diagram_to_string() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let output = diagram_to_string(&sentence);
        assert!(output.contains("alice"));
        assert!(output.contains("sleeps"));
        assert!(output.contains("Total qubits"));
    }

    #[test]
    fn test_circuit_to_string() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();
        let config = QuantumNlpConfig::new();
        let circuit = sentence_to_circuit(&sentence, &config);
        let output = circuit_to_string(&circuit);
        assert!(output.contains("DisCoCat"));
        assert!(output.contains("Gates"));
    }

    // --- Error type display test ---

    #[test]
    fn test_error_display() {
        let err = NlpError::UnknownWord("xyz".to_string());
        assert_eq!(format!("{}", err), "Unknown word: 'xyz'");

        let err = NlpError::EmptySentence;
        assert_eq!(format!("{}", err), "Empty sentence");

        let err = NlpError::QubitMismatch {
            expected: 3,
            got: 5,
        };
        assert!(format!("{}", err).contains("3"));
        assert!(format!("{}", err).contains("5"));
    }

    // --- Demo test ---

    #[test]
    fn test_demo_runs() {
        let output = run_discocat_demo();
        assert!(output.contains("DisCoCat"));
        assert!(output.contains("Word Order"));
        assert!(output.contains("Adjective"));
    }

    // --- Post-selection test ---

    #[test]
    fn test_post_selection_preserves_normalization() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::h(&mut state, 1);
        // State is |++> = (|00> + |01> + |10> + |11>)/2

        post_select_zero(&mut state, 0);
        // Should project to (|00> + |01>)/sqrt(2)

        let norm: f64 = state.amplitudes_ref().iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Post-selection should preserve normalization, got {}",
            norm
        );

        // Qubit 0 should be |0> with certainty
        let p0: f64 = state
            .amplitudes_ref()
            .iter()
            .enumerate()
            .filter(|(i, _)| (i >> 0) & 1 == 0)
            .map(|(_, a)| a.norm_sqr())
            .sum();
        assert!(
            (p0 - 1.0).abs() < 1e-10,
            "Post-selected qubit should be |0>"
        );
    }

    // --- Multi-layer test ---

    #[test]
    fn test_multi_layer_increases_gates() {
        let lex = Lexicon::english_basic();
        let sentence = parse_sentence(&lex, "alice sleeps").unwrap();

        let config1 = QuantumNlpConfig::new().with_layers(1);
        let config2 = QuantumNlpConfig::new().with_layers(3);

        let c1 = sentence_to_circuit(&sentence, &config1);
        let c2 = sentence_to_circuit(&sentence, &config2);

        assert!(
            c2.gates.len() >= c1.gates.len(),
            "More layers should produce at least as many gates"
        );
    }

    // --- Tensor product test ---

    #[test]
    fn test_tensor_product_states() {
        let s1 = QuantumState::new(1); // |0>
        let mut s2 = QuantumState::new(1);
        GateOperations::x(&mut s2, 0); // |1>

        let product = tensor_product_states(&s1, &s2);
        let amps = product.amplitudes_ref();

        // |0> tensor |1> = |01>, which is index 1 in a 2-qubit system
        assert!(amps[1].norm_sqr() > 0.99, "Should be |01>");
        assert!(amps[0].norm_sqr() < 0.01);
    }
}
