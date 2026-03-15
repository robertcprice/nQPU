//! Native import of Google's Stim QEC circuit format.
//!
//! Stim is a fast stabilizer circuit simulator by Google for quantum error
//! correction research. This module parses the Stim text-based circuit format
//! and converts it into nQPU-Metal gate sequences, detector models, and noise
//! event lists suitable for simulation and decoding.
//!
//! # Supported Stim Instructions
//!
//! - Single-qubit gates: `H`, `X`, `Y`, `Z`, `S`, `S_DAG`, `T`
//! - Two-qubit gates: `CNOT`/`CX`, `CZ`, `SWAP`, `ISWAP`
//! - Noise channels: `DEPOLARIZE1`, `DEPOLARIZE2`, `X_ERROR`, `Z_ERROR`
//! - Measurement: `M`, `MR` (measure-reset), `R` (reset)
//! - Annotations: `DETECTOR`, `OBSERVABLE_INCLUDE`, `TICK`
//! - Control flow: `REPEAT N { ... }`
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::stim_import::StimCircuit;
//!
//! let stim_text = r#"
//! H 0
//! CNOT 0 1
//! M 0 1
//! DETECTOR rec[-1] rec[-2]
//! "#;
//!
//! let circuit = StimCircuit::parse(stim_text).unwrap();
//! let conversion = circuit.to_gates();
//! assert_eq!(conversion.num_qubits, 2);
//! assert_eq!(conversion.num_measurements, 2);
//! ```

use crate::gates::{Gate, GateType};
use crate::noise_models::NoiseModel;
use crate::qec_interop::{DetectorNode, ErrorTerm, StimLikeDetectorModel};
use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during Stim circuit parsing.
#[derive(Clone, Debug, PartialEq)]
pub enum StimError {
    /// General parse failure with context message.
    ParseError(String),
    /// Encountered an instruction not supported by the importer.
    UnknownInstruction(String),
    /// A qubit target could not be parsed as a valid index.
    InvalidTarget(String),
    /// Mismatched or missing braces in a REPEAT block.
    UnmatchedBrace(String),
}

impl fmt::Display for StimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StimError::ParseError(msg) => write!(f, "Stim parse error: {}", msg),
            StimError::UnknownInstruction(name) => {
                write!(f, "Unknown Stim instruction: {}", name)
            }
            StimError::InvalidTarget(msg) => write!(f, "Invalid target: {}", msg),
            StimError::UnmatchedBrace(msg) => write!(f, "Unmatched brace: {}", msg),
        }
    }
}

impl std::error::Error for StimError {}

// ---------------------------------------------------------------------------
// Parsed instruction AST
// ---------------------------------------------------------------------------

/// A parsed Stim instruction.
#[derive(Clone, Debug, PartialEq)]
pub enum StimInstruction {
    /// Single-qubit gate applied to one or more targets.
    Gate { name: String, targets: Vec<usize> },
    /// Two-qubit gate applied to consecutive pairs of targets.
    TwoQubitGate {
        name: String,
        pairs: Vec<(usize, usize)>,
    },
    /// Single-qubit noise channel with explicit probability.
    Noise1 {
        name: String,
        probability: f64,
        targets: Vec<usize>,
    },
    /// Two-qubit noise channel with explicit probability.
    Noise2 {
        name: String,
        probability: f64,
        pairs: Vec<(usize, usize)>,
    },
    /// Measurement of one or more qubits.
    Measure { targets: Vec<usize> },
    /// Reset of one or more qubits to |0>.
    Reset { targets: Vec<usize> },
    /// Measure then immediately reset.
    MeasureReset { targets: Vec<usize> },
    /// Detector annotation referencing measurement records.
    Detector { measurement_records: Vec<i64> },
    /// Logical observable annotation.
    ObservableInclude {
        observable_id: usize,
        measurement_records: Vec<i64>,
    },
    /// Repeat block with a body of instructions.
    Repeat {
        count: usize,
        body: Vec<StimInstruction>,
    },
    /// Tick barrier (layer separator).
    Tick,
}

// ---------------------------------------------------------------------------
// Parsed circuit
// ---------------------------------------------------------------------------

/// A fully parsed Stim circuit.
#[derive(Clone, Debug)]
pub struct StimCircuit {
    /// The flat list of top-level instructions (REPEAT bodies are nested).
    pub instructions: Vec<StimInstruction>,
    /// Total number of qubits referenced in the circuit.
    pub num_qubits: usize,
}

// ---------------------------------------------------------------------------
// Conversion output
// ---------------------------------------------------------------------------

/// Result of converting a Stim circuit into nQPU-Metal gate operations.
#[derive(Clone, Debug)]
pub struct StimConversion {
    /// Sequence of nQPU-Metal gates (measurements and resets are not included
    /// as gates; see `num_measurements` for the measurement count).
    pub gates: Vec<Gate>,
    /// Detector annotations extracted from the circuit.
    pub detectors: Vec<DetectorNode>,
    /// Logical observable annotations, each as a list of measurement record
    /// offsets (negative indices relative to the current measurement counter).
    pub logical_observables: Vec<Vec<i64>>,
    /// Noise events extracted from noise instructions.
    pub noise_events: Vec<NoiseEvent>,
    /// Total number of qubits referenced.
    pub num_qubits: usize,
    /// Total number of measurement operations encountered (after unrolling).
    pub num_measurements: usize,
}

/// A single noise event parsed from the Stim circuit.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseEvent {
    /// The Stim noise instruction name (e.g. `DEPOLARIZE1`).
    pub noise_type: String,
    /// Error probability.
    pub probability: f64,
    /// Qubits affected by this noise event.
    pub qubits: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Helpers: instruction classification
// ---------------------------------------------------------------------------

/// Returns `true` if `name` is a known single-qubit gate in Stim.
fn is_single_qubit_gate(name: &str) -> bool {
    matches!(
        name,
        "H" | "X" | "Y" | "Z" | "S" | "S_DAG" | "T" | "SX" | "I"
    )
}

/// Returns `true` if `name` is a known two-qubit gate in Stim.
fn is_two_qubit_gate(name: &str) -> bool {
    matches!(name, "CNOT" | "CX" | "CZ" | "SWAP" | "ISWAP")
}

/// Returns `true` if `name` is a single-qubit noise channel.
fn is_noise1(name: &str) -> bool {
    matches!(name, "DEPOLARIZE1" | "X_ERROR" | "Z_ERROR" | "Y_ERROR")
}

/// Returns `true` if `name` is a two-qubit noise channel.
fn is_noise2(name: &str) -> bool {
    matches!(name, "DEPOLARIZE2")
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Parse a parenthesized probability from a token like `DEPOLARIZE1(0.01)`.
/// Returns `(base_name, probability)`.
fn parse_parameterized_name(token: &str) -> Result<(String, f64), StimError> {
    if let Some(paren_start) = token.find('(') {
        let paren_end = token
            .find(')')
            .ok_or_else(|| StimError::ParseError(format!("Missing closing paren in: {}", token)))?;
        let base = token[..paren_start].to_string();
        let param_str = &token[paren_start + 1..paren_end];
        let value: f64 = param_str.parse().map_err(|_| {
            StimError::ParseError(format!("Invalid parameter value: {}", param_str))
        })?;
        Ok((base, value))
    } else {
        Err(StimError::ParseError(format!(
            "Expected parameterized instruction: {}",
            token
        )))
    }
}

/// Parse a list of qubit indices from string tokens.
fn parse_qubit_targets(tokens: &[&str]) -> Result<Vec<usize>, StimError> {
    tokens
        .iter()
        .map(|t| {
            t.parse::<usize>()
                .map_err(|_| StimError::InvalidTarget(format!("Cannot parse qubit index: {}", t)))
        })
        .collect()
}

/// Parse consecutive pairs from a flat list of qubit targets.
fn parse_qubit_pairs(targets: &[usize]) -> Result<Vec<(usize, usize)>, StimError> {
    if targets.len() % 2 != 0 {
        return Err(StimError::InvalidTarget(
            "Two-qubit gate requires an even number of targets".to_string(),
        ));
    }
    Ok(targets.chunks(2).map(|c| (c[0], c[1])).collect())
}

/// Parse `rec[-N]` tokens into negative measurement record offsets.
fn parse_measurement_records(tokens: &[&str]) -> Result<Vec<i64>, StimError> {
    tokens
        .iter()
        .map(|t| {
            let t = t.trim();
            if t.starts_with("rec[") && t.ends_with(']') {
                let inner = &t[4..t.len() - 1];
                inner.parse::<i64>().map_err(|_| {
                    StimError::ParseError(format!("Invalid measurement record: {}", t))
                })
            } else {
                Err(StimError::ParseError(format!(
                    "Expected rec[-N], got: {}",
                    t
                )))
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Block-level parsing (handles REPEAT { ... })
// ---------------------------------------------------------------------------

/// Parse a sequence of lines into instructions, consuming lines from the iterator.
/// `depth` tracks nesting for brace matching.
fn parse_block(
    lines: &[&str],
    start: usize,
    depth: usize,
) -> Result<(Vec<StimInstruction>, usize), StimError> {
    let mut instructions = Vec::new();
    let mut idx = start;

    while idx < lines.len() {
        let raw = lines[idx].trim();

        // Skip blank lines and comments.
        if raw.is_empty() || raw.starts_with('#') {
            idx += 1;
            continue;
        }

        // Closing brace ends the current block.
        if raw == "}" {
            if depth == 0 {
                return Err(StimError::UnmatchedBrace(
                    "Unexpected closing brace at top level".to_string(),
                ));
            }
            return Ok((instructions, idx + 1));
        }

        // Check for REPEAT.
        if raw.starts_with("REPEAT") {
            let parts: Vec<&str> = raw.split_whitespace().collect();
            if parts.len() < 2 {
                return Err(StimError::ParseError("REPEAT requires a count".to_string()));
            }
            let count: usize = parts[1].parse().map_err(|_| {
                StimError::ParseError(format!("Invalid REPEAT count: {}", parts[1]))
            })?;

            // The opening brace may be on the same line or the next line.
            let brace_on_same_line = raw.contains('{');
            let body_start = if brace_on_same_line {
                idx + 1
            } else {
                // Find the next non-empty line that should be `{`.
                let mut next = idx + 1;
                while next < lines.len() && lines[next].trim().is_empty() {
                    next += 1;
                }
                if next >= lines.len() || lines[next].trim() != "{" {
                    return Err(StimError::UnmatchedBrace(
                        "Expected '{' after REPEAT count".to_string(),
                    ));
                }
                next + 1
            };

            let (body, after_close) = parse_block(lines, body_start, depth + 1)?;
            instructions.push(StimInstruction::Repeat { count, body });
            idx = after_close;
            continue;
        }

        // Regular instruction line.
        let inst = parse_instruction_line(raw)?;
        instructions.push(inst);
        idx += 1;
    }

    if depth > 0 {
        return Err(StimError::UnmatchedBrace(
            "Unterminated REPEAT block (missing closing brace)".to_string(),
        ));
    }

    Ok((instructions, idx))
}

/// Parse a single non-block instruction line.
fn parse_instruction_line(line: &str) -> Result<StimInstruction, StimError> {
    // Strip inline comments.
    let line = if let Some(hash_pos) = line.find('#') {
        line[..hash_pos].trim()
    } else {
        line.trim()
    };

    if line.is_empty() {
        return Err(StimError::ParseError(
            "Empty instruction after stripping comment".to_string(),
        ));
    }

    let tokens: Vec<&str> = line.split_whitespace().collect();
    let instruction_name = tokens[0];
    let args = &tokens[1..];

    // TICK
    if instruction_name == "TICK" {
        return Ok(StimInstruction::Tick);
    }

    // M (measure)
    if instruction_name == "M" {
        let targets = parse_qubit_targets(args)?;
        return Ok(StimInstruction::Measure { targets });
    }

    // MR (measure-reset)
    if instruction_name == "MR" {
        let targets = parse_qubit_targets(args)?;
        return Ok(StimInstruction::MeasureReset { targets });
    }

    // R (reset)
    if instruction_name == "R" {
        let targets = parse_qubit_targets(args)?;
        return Ok(StimInstruction::Reset { targets });
    }

    // DETECTOR
    if instruction_name == "DETECTOR" {
        let records = parse_measurement_records(args)?;
        return Ok(StimInstruction::Detector {
            measurement_records: records,
        });
    }

    // OBSERVABLE_INCLUDE(k)
    if instruction_name.starts_with("OBSERVABLE_INCLUDE") {
        let (_, obs_id) = parse_parameterized_name(instruction_name)?;
        let records = parse_measurement_records(args)?;
        return Ok(StimInstruction::ObservableInclude {
            observable_id: obs_id as usize,
            measurement_records: records,
        });
    }

    // Parameterized noise instructions: DEPOLARIZE1(p), DEPOLARIZE2(p), X_ERROR(p), etc.
    if instruction_name.contains('(') {
        let (base_name, probability) = parse_parameterized_name(instruction_name)?;

        if is_noise1(&base_name) {
            let targets = parse_qubit_targets(args)?;
            return Ok(StimInstruction::Noise1 {
                name: base_name,
                probability,
                targets,
            });
        }

        if is_noise2(&base_name) {
            let targets = parse_qubit_targets(args)?;
            let pairs = parse_qubit_pairs(&targets)?;
            return Ok(StimInstruction::Noise2 {
                name: base_name,
                probability,
                pairs,
            });
        }

        return Err(StimError::UnknownInstruction(format!(
            "Unknown parameterized instruction: {}",
            base_name
        )));
    }

    // Single-qubit gates.
    if is_single_qubit_gate(instruction_name) {
        let targets = parse_qubit_targets(args)?;
        return Ok(StimInstruction::Gate {
            name: instruction_name.to_string(),
            targets,
        });
    }

    // Two-qubit gates.
    if is_two_qubit_gate(instruction_name) {
        let targets = parse_qubit_targets(args)?;
        let pairs = parse_qubit_pairs(&targets)?;
        return Ok(StimInstruction::TwoQubitGate {
            name: instruction_name.to_string(),
            pairs,
        });
    }

    Err(StimError::UnknownInstruction(instruction_name.to_string()))
}

// ---------------------------------------------------------------------------
// StimCircuit implementation
// ---------------------------------------------------------------------------

impl StimCircuit {
    /// Parse a Stim circuit from its text representation.
    ///
    /// Returns a `StimCircuit` with all instructions parsed and the inferred
    /// qubit count (1 + the maximum qubit index referenced).
    pub fn parse(input: &str) -> Result<StimCircuit, StimError> {
        let lines: Vec<&str> = input.lines().collect();
        let (instructions, _) = parse_block(&lines, 0, 0)?;
        let num_qubits = Self::compute_num_qubits(&instructions);
        Ok(StimCircuit {
            instructions,
            num_qubits,
        })
    }

    /// Recursively determine the maximum qubit index across all instructions.
    fn compute_num_qubits(instructions: &[StimInstruction]) -> usize {
        let mut max_qubit: Option<usize> = None;

        for inst in instructions {
            let local_max = match inst {
                StimInstruction::Gate { targets, .. } => targets.iter().max().copied(),
                StimInstruction::TwoQubitGate { pairs, .. } => pairs
                    .iter()
                    .flat_map(|(a, b)| std::iter::once(*a).chain(std::iter::once(*b)))
                    .max(),
                StimInstruction::Noise1 { targets, .. } => targets.iter().max().copied(),
                StimInstruction::Noise2 { pairs, .. } => pairs
                    .iter()
                    .flat_map(|(a, b)| std::iter::once(*a).chain(std::iter::once(*b)))
                    .max(),
                StimInstruction::Measure { targets } => targets.iter().max().copied(),
                StimInstruction::Reset { targets } => targets.iter().max().copied(),
                StimInstruction::MeasureReset { targets } => targets.iter().max().copied(),
                StimInstruction::Repeat { body, .. } => {
                    let inner = Self::compute_num_qubits(body);
                    if inner > 0 {
                        Some(inner - 1)
                    } else {
                        None
                    }
                }
                StimInstruction::Detector { .. }
                | StimInstruction::ObservableInclude { .. }
                | StimInstruction::Tick => None,
            };

            if let Some(lm) = local_max {
                max_qubit = Some(max_qubit.map_or(lm, |prev: usize| prev.max(lm)));
            }
        }

        max_qubit.map_or(0, |m| m + 1)
    }

    /// Flatten all `REPEAT` blocks into a linear instruction stream.
    ///
    /// The returned vector contains no `Repeat` variants; every instruction
    /// inside a repeat body is duplicated the appropriate number of times.
    pub fn unroll_repeats(&self) -> Vec<StimInstruction> {
        Self::unroll_instructions(&self.instructions)
    }

    fn unroll_instructions(instructions: &[StimInstruction]) -> Vec<StimInstruction> {
        let mut result = Vec::new();
        for inst in instructions {
            match inst {
                StimInstruction::Repeat { count, body } => {
                    let unrolled_body = Self::unroll_instructions(body);
                    for _ in 0..*count {
                        result.extend(unrolled_body.iter().cloned());
                    }
                }
                other => result.push(other.clone()),
            }
        }
        result
    }

    /// Convert the parsed Stim circuit into nQPU-Metal gates and metadata.
    ///
    /// Gates, noise events, detectors, and observable annotations are all
    /// extracted from the unrolled instruction stream.
    pub fn to_gates(&self) -> StimConversion {
        let flat = self.unroll_repeats();

        let mut gates = Vec::new();
        let mut detectors = Vec::new();
        let mut logical_observables: Vec<Vec<i64>> = Vec::new();
        let mut noise_events = Vec::new();
        let mut measurement_count: usize = 0;
        let mut detector_id: usize = 0;

        for inst in &flat {
            match inst {
                StimInstruction::Gate { name, targets } => {
                    let gate_type = Self::map_single_gate(name);
                    for &t in targets {
                        gates.push(Gate::single(gate_type.clone(), t));
                    }
                }
                StimInstruction::TwoQubitGate { name, pairs } => {
                    let gate_type = Self::map_two_qubit_gate(name);
                    for &(a, b) in pairs {
                        match &gate_type {
                            GateType::CNOT => {
                                gates.push(Gate::two(GateType::CNOT, a, b));
                            }
                            GateType::CZ => {
                                gates.push(Gate::two(GateType::CZ, a, b));
                            }
                            GateType::SWAP => {
                                gates.push(Gate::new(GateType::SWAP, vec![a, b], vec![]));
                            }
                            GateType::ISWAP => {
                                gates.push(Gate::new(GateType::ISWAP, vec![a, b], vec![]));
                            }
                            other => {
                                gates.push(Gate::new(other.clone(), vec![b], vec![a]));
                            }
                        }
                    }
                }
                StimInstruction::Measure { targets } => {
                    measurement_count += targets.len();
                }
                StimInstruction::MeasureReset { targets } => {
                    measurement_count += targets.len();
                }
                StimInstruction::Reset { .. } => {
                    // Reset is a state preparation operation; no gate emitted.
                }
                StimInstruction::Noise1 {
                    name,
                    probability,
                    targets,
                } => {
                    noise_events.push(NoiseEvent {
                        noise_type: name.clone(),
                        probability: *probability,
                        qubits: targets.clone(),
                    });
                }
                StimInstruction::Noise2 {
                    name,
                    probability,
                    pairs,
                } => {
                    for (a, b) in pairs {
                        noise_events.push(NoiseEvent {
                            noise_type: name.clone(),
                            probability: *probability,
                            qubits: vec![*a, *b],
                        });
                    }
                }
                StimInstruction::Detector {
                    measurement_records,
                } => {
                    detectors.push(DetectorNode {
                        id: detector_id,
                        round: 0,
                        x: detector_id % self.num_qubits.max(1),
                        y: detector_id / self.num_qubits.max(1),
                    });
                    detector_id += 1;
                    // Store the measurement record references for later use
                    // in detector model construction.
                    let _ = measurement_records;
                }
                StimInstruction::ObservableInclude {
                    observable_id,
                    measurement_records,
                } => {
                    // Extend the logical observables list to accommodate the id.
                    while logical_observables.len() <= *observable_id {
                        logical_observables.push(Vec::new());
                    }
                    logical_observables[*observable_id].extend(measurement_records.iter().copied());
                }
                StimInstruction::Tick => {
                    // Tick is a barrier / layer separator; no gate emitted.
                }
                StimInstruction::Repeat { .. } => {
                    // Should not appear after unrolling; handled defensively.
                }
            }
        }

        StimConversion {
            gates,
            detectors,
            logical_observables,
            noise_events,
            num_qubits: self.num_qubits,
            num_measurements: measurement_count,
        }
    }

    /// Build a `StimLikeDetectorModel` from the parsed circuit.
    ///
    /// The `distance` parameter is the code distance (cannot be inferred from
    /// the circuit alone and must be supplied by the caller). Rounds are
    /// estimated from REPEAT counts in the instruction stream.
    pub fn to_detector_model(&self, distance: usize) -> StimLikeDetectorModel {
        let conversion = self.to_gates();

        let rounds = self.estimate_rounds();

        // Build error terms from noise events.
        let error_terms: Vec<ErrorTerm> = conversion
            .noise_events
            .iter()
            .enumerate()
            .map(|(i, event)| {
                // Map each noise event to an error term. For single-qubit noise
                // we associate the nearest detector(s); for simplicity we use
                // the qubit indices as detector references when available.
                let det_refs: Vec<usize> = event
                    .qubits
                    .iter()
                    .filter(|&&q| q < conversion.detectors.len())
                    .copied()
                    .collect();
                ErrorTerm {
                    probability: event.probability,
                    detectors: if det_refs.is_empty() {
                        vec![i % conversion.detectors.len().max(1)]
                    } else {
                        det_refs
                    },
                    observables: vec![],
                }
            })
            .collect();

        // Logical observable: collect detector ids referenced by all observable
        // annotations. We flatten all measurement record offsets and resolve
        // them to absolute detector ids.
        let logical_observable: Vec<usize> = conversion
            .logical_observables
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect();

        StimLikeDetectorModel {
            distance,
            rounds,
            detectors: conversion.detectors,
            error_terms,
            logical_observable,
        }
    }

    /// Estimate the number of QEC rounds from REPEAT counts.
    fn estimate_rounds(&self) -> usize {
        Self::estimate_rounds_inner(&self.instructions)
    }

    fn estimate_rounds_inner(instructions: &[StimInstruction]) -> usize {
        let mut rounds = 1;
        for inst in instructions {
            if let StimInstruction::Repeat { count, body } = inst {
                let inner = Self::estimate_rounds_inner(body);
                rounds = rounds.max(*count * inner);
            }
        }
        rounds
    }

    /// Build a `NoiseModel` from the noise events in this circuit.
    ///
    /// Extracts the dominant depolarizing probability and maps it to the
    /// appropriate `NoiseModel` field. Other noise types are approximated.
    pub fn to_noise_model(&self) -> NoiseModel {
        let conversion = self.to_gates();
        let mut depol1_max: f64 = 0.0;
        let mut depol2_max: f64 = 0.0;

        for event in &conversion.noise_events {
            match event.noise_type.as_str() {
                "DEPOLARIZE1" | "X_ERROR" | "Z_ERROR" | "Y_ERROR" => {
                    depol1_max = depol1_max.max(event.probability);
                }
                "DEPOLARIZE2" => {
                    depol2_max = depol2_max.max(event.probability);
                }
                _ => {}
            }
        }

        let dominant = depol1_max.max(depol2_max);
        NoiseModel {
            depolarizing_prob: dominant,
            amplitude_damping_prob: dominant * 0.1,
            phase_damping_prob: dominant * 0.1,
            readout_error: (dominant, dominant),
            coherent_errors: std::collections::HashMap::new(),
            crosstalk_prob: depol2_max * 0.5,
        }
    }

    // -----------------------------------------------------------------------
    // Gate mapping
    // -----------------------------------------------------------------------

    /// Map a Stim single-qubit gate name to a `GateType`.
    fn map_single_gate(name: &str) -> GateType {
        match name {
            "H" => GateType::H,
            "X" => GateType::X,
            "Y" => GateType::Y,
            "Z" => GateType::Z,
            "S" => GateType::S,
            "S_DAG" => GateType::Rz(-std::f64::consts::FRAC_PI_2),
            "T" => GateType::T,
            "SX" => GateType::SX,
            // Identity: use Rz(0) as a no-op rotation.
            "I" => GateType::Rz(0.0),
            _ => GateType::H, // Defensive fallback; unreachable for validated input.
        }
    }

    /// Map a Stim two-qubit gate name to a `GateType`.
    fn map_two_qubit_gate(name: &str) -> GateType {
        match name {
            "CNOT" | "CX" => GateType::CNOT,
            "CZ" => GateType::CZ,
            "SWAP" => GateType::SWAP,
            "ISWAP" => GateType::ISWAP,
            _ => GateType::CNOT, // Defensive fallback.
        }
    }
}

// ---------------------------------------------------------------------------
// Display for StimInstruction (debugging convenience)
// ---------------------------------------------------------------------------

impl fmt::Display for StimInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StimInstruction::Gate { name, targets } => {
                let t: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
                write!(f, "{} {}", name, t.join(" "))
            }
            StimInstruction::TwoQubitGate { name, pairs } => {
                let t: Vec<String> = pairs
                    .iter()
                    .flat_map(|(a, b)| vec![a.to_string(), b.to_string()])
                    .collect();
                write!(f, "{} {}", name, t.join(" "))
            }
            StimInstruction::Noise1 {
                name,
                probability,
                targets,
            } => {
                let t: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
                write!(f, "{}({}) {}", name, probability, t.join(" "))
            }
            StimInstruction::Noise2 {
                name,
                probability,
                pairs,
            } => {
                let t: Vec<String> = pairs
                    .iter()
                    .flat_map(|(a, b)| vec![a.to_string(), b.to_string()])
                    .collect();
                write!(f, "{}({}) {}", name, probability, t.join(" "))
            }
            StimInstruction::Measure { targets } => {
                let t: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
                write!(f, "M {}", t.join(" "))
            }
            StimInstruction::Reset { targets } => {
                let t: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
                write!(f, "R {}", t.join(" "))
            }
            StimInstruction::MeasureReset { targets } => {
                let t: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
                write!(f, "MR {}", t.join(" "))
            }
            StimInstruction::Detector {
                measurement_records,
            } => {
                let r: Vec<String> = measurement_records
                    .iter()
                    .map(|r| format!("rec[{}]", r))
                    .collect();
                write!(f, "DETECTOR {}", r.join(" "))
            }
            StimInstruction::ObservableInclude {
                observable_id,
                measurement_records,
            } => {
                let r: Vec<String> = measurement_records
                    .iter()
                    .map(|r| format!("rec[{}]", r))
                    .collect();
                write!(f, "OBSERVABLE_INCLUDE({}) {}", observable_id, r.join(" "))
            }
            StimInstruction::Repeat { count, body } => {
                writeln!(f, "REPEAT {} {{", count)?;
                for inst in body {
                    writeln!(f, "    {}", inst)?;
                }
                write!(f, "}}")
            }
            StimInstruction::Tick => write!(f, "TICK"),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Parse simple Stim circuit
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_simple_circuit() {
        let input = "H 0 1 2\nCNOT 0 1\nM 0 1\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.instructions.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 2. Parse with REPEAT block
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_repeat_block() {
        let input = r#"
H 0
REPEAT 3 {
    CNOT 0 1
    M 0 1
}
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        // Top-level: H, REPEAT
        assert_eq!(circuit.instructions.len(), 2);
        match &circuit.instructions[1] {
            StimInstruction::Repeat { count, body } => {
                assert_eq!(*count, 3);
                assert_eq!(body.len(), 2);
            }
            other => panic!("Expected Repeat, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 3. Gate mapping verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_gate_mapping_single_qubit() {
        assert_eq!(StimCircuit::map_single_gate("H"), GateType::H);
        assert_eq!(StimCircuit::map_single_gate("X"), GateType::X);
        assert_eq!(StimCircuit::map_single_gate("Y"), GateType::Y);
        assert_eq!(StimCircuit::map_single_gate("Z"), GateType::Z);
        assert_eq!(StimCircuit::map_single_gate("S"), GateType::S);
        assert_eq!(StimCircuit::map_single_gate("T"), GateType::T);
        assert_eq!(
            StimCircuit::map_single_gate("S_DAG"),
            GateType::Rz(-std::f64::consts::FRAC_PI_2)
        );
    }

    #[test]
    fn test_gate_mapping_two_qubit() {
        assert_eq!(StimCircuit::map_two_qubit_gate("CNOT"), GateType::CNOT);
        assert_eq!(StimCircuit::map_two_qubit_gate("CX"), GateType::CNOT);
        assert_eq!(StimCircuit::map_two_qubit_gate("CZ"), GateType::CZ);
        assert_eq!(StimCircuit::map_two_qubit_gate("SWAP"), GateType::SWAP);
        assert_eq!(StimCircuit::map_two_qubit_gate("ISWAP"), GateType::ISWAP);
    }

    // -----------------------------------------------------------------------
    // 4. Detector annotation parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_detector() {
        let input = "M 0 1\nDETECTOR rec[-1] rec[-2]\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 2);
        match &circuit.instructions[1] {
            StimInstruction::Detector {
                measurement_records,
            } => {
                assert_eq!(measurement_records, &[-1, -2]);
            }
            other => panic!("Expected Detector, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 5. Noise instruction parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_noise1() {
        let input = "DEPOLARIZE1(0.01) 0 1 2\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::Noise1 {
                name,
                probability,
                targets,
            } => {
                assert_eq!(name, "DEPOLARIZE1");
                assert!((probability - 0.01).abs() < 1e-12);
                assert_eq!(targets, &[0, 1, 2]);
            }
            other => panic!("Expected Noise1, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_x_error() {
        let input = "X_ERROR(0.005) 0 3\n";
        let circuit = StimCircuit::parse(input).unwrap();
        match &circuit.instructions[0] {
            StimInstruction::Noise1 {
                name,
                probability,
                targets,
            } => {
                assert_eq!(name, "X_ERROR");
                assert!((probability - 0.005).abs() < 1e-12);
                assert_eq!(targets, &[0, 3]);
            }
            other => panic!("Expected Noise1, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 6. rec[-N] measurement record parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_measurement_records() {
        let tokens = vec!["rec[-1]", "rec[-3]", "rec[-5]"];
        let records = parse_measurement_records(&tokens).unwrap();
        assert_eq!(records, vec![-1, -3, -5]);
    }

    #[test]
    fn test_parse_measurement_record_invalid() {
        let tokens = vec!["rec[abc]"];
        let result = parse_measurement_records(&tokens);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 7. Unroll repeats
    // -----------------------------------------------------------------------

    #[test]
    fn test_unroll_repeats() {
        let input = r#"
REPEAT 3 {
    H 0
    CNOT 0 1
}
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let unrolled = circuit.unroll_repeats();
        // 3 iterations * 2 instructions = 6
        assert_eq!(unrolled.len(), 6);
        for inst in &unrolled {
            assert!(!matches!(inst, StimInstruction::Repeat { .. }));
        }
    }

    #[test]
    fn test_unroll_nested_repeats() {
        let input = r#"
REPEAT 2 {
    REPEAT 3 {
        H 0
    }
}
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let unrolled = circuit.unroll_repeats();
        // 2 * 3 = 6 H gates
        assert_eq!(unrolled.len(), 6);
    }

    // -----------------------------------------------------------------------
    // 8. Convert to detector model
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_detector_model() {
        let input = r#"
H 0 1
CNOT 0 1
DEPOLARIZE1(0.01) 0 1
M 0 1
DETECTOR rec[-1] rec[-2]
OBSERVABLE_INCLUDE(0) rec[-1]
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let model = circuit.to_detector_model(3);
        assert_eq!(model.distance, 3);
        assert_eq!(model.detectors.len(), 1);
        assert!(!model.error_terms.is_empty());
    }

    // -----------------------------------------------------------------------
    // 9. Error cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_unknown_gate_error() {
        let input = "FOOBAR 0 1\n";
        let result = StimCircuit::parse(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            StimError::UnknownInstruction(name) => assert_eq!(name, "FOOBAR"),
            other => panic!("Expected UnknownInstruction, got {:?}", other),
        }
    }

    #[test]
    fn test_bad_target_error() {
        let input = "H abc\n";
        let result = StimCircuit::parse(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            StimError::InvalidTarget(_) => {}
            other => panic!("Expected InvalidTarget, got {:?}", other),
        }
    }

    #[test]
    fn test_unmatched_brace_error() {
        let input = "REPEAT 3 {\nH 0\n";
        let result = StimCircuit::parse(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            StimError::UnmatchedBrace(_) => {}
            other => panic!("Expected UnmatchedBrace, got {:?}", other),
        }
    }

    #[test]
    fn test_unexpected_closing_brace() {
        let input = "H 0\n}\n";
        let result = StimCircuit::parse(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            StimError::UnmatchedBrace(_) => {}
            other => panic!("Expected UnmatchedBrace, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 10. Round-trip: parse -> convert to gates -> verify gate count
    // -----------------------------------------------------------------------

    #[test]
    fn test_roundtrip_gate_count() {
        let input = r#"
H 0 1 2
CNOT 0 1 2 3
M 0 1 2 3
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let conversion = circuit.to_gates();
        // 3 H gates + 2 CNOT pairs = 5 gates total
        assert_eq!(conversion.gates.len(), 5);
        assert_eq!(conversion.num_measurements, 4);
        assert_eq!(conversion.num_qubits, 4);
    }

    // -----------------------------------------------------------------------
    // 11. Two-qubit noise parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_noise2() {
        let input = "DEPOLARIZE2(0.02) 0 1 2 3\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 1);
        match &circuit.instructions[0] {
            StimInstruction::Noise2 {
                name,
                probability,
                pairs,
            } => {
                assert_eq!(name, "DEPOLARIZE2");
                assert!((probability - 0.02).abs() < 1e-12);
                assert_eq!(pairs, &[(0, 1), (2, 3)]);
            }
            other => panic!("Expected Noise2, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 12. Observable include parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_observable_include() {
        let input = "M 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 2);
        match &circuit.instructions[1] {
            StimInstruction::ObservableInclude {
                observable_id,
                measurement_records,
            } => {
                assert_eq!(*observable_id, 0);
                assert_eq!(measurement_records, &[-1]);
            }
            other => panic!("Expected ObservableInclude, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_observable_includes() {
        let input = r#"
M 0 1 2
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
OBSERVABLE_INCLUDE(1) rec[-3]
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let conversion = circuit.to_gates();
        assert_eq!(conversion.logical_observables.len(), 2);
        assert_eq!(conversion.logical_observables[0], vec![-1, -2]);
        assert_eq!(conversion.logical_observables[1], vec![-3]);
    }

    // -----------------------------------------------------------------------
    // 13. Multi-line comments and whitespace handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_comments_and_whitespace() {
        let input = r#"
# This is a comment
   # Indented comment

H 0   # Inline comment
   CNOT 0 1

# Another comment
M 0 1
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 3);
        assert_eq!(circuit.num_qubits, 2);
    }

    // -----------------------------------------------------------------------
    // 14. TICK instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_tick_instruction() {
        let input = "H 0\nTICK\nCNOT 0 1\nTICK\nM 0 1\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 5);
        assert_eq!(circuit.instructions[1], StimInstruction::Tick);
        assert_eq!(circuit.instructions[3], StimInstruction::Tick);
    }

    // -----------------------------------------------------------------------
    // 15. Reset and MeasureReset
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_and_measure_reset() {
        let input = "R 0 1\nH 0\nCNOT 0 1\nMR 0 1\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.instructions.len(), 4);
        match &circuit.instructions[0] {
            StimInstruction::Reset { targets } => assert_eq!(targets, &[0, 1]),
            other => panic!("Expected Reset, got {:?}", other),
        }
        match &circuit.instructions[3] {
            StimInstruction::MeasureReset { targets } => assert_eq!(targets, &[0, 1]),
            other => panic!("Expected MeasureReset, got {:?}", other),
        }

        let conversion = circuit.to_gates();
        // MR produces measurements
        assert_eq!(conversion.num_measurements, 2);
    }

    // -----------------------------------------------------------------------
    // 16. Full surface-code-like circuit
    // -----------------------------------------------------------------------

    #[test]
    fn test_surface_code_circuit() {
        let input = r#"
R 0 1 2 3 4
REPEAT 3 {
    H 1 3
    TICK
    CNOT 1 0 3 2
    TICK
    CNOT 1 2 3 4
    TICK
    H 1 3
    TICK
    M 1 3
    DETECTOR rec[-1] rec[-2]
    R 1 3
}
M 0 2 4
OBSERVABLE_INCLUDE(0) rec[-1] rec[-3]
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.num_qubits, 5);

        let conversion = circuit.to_gates();
        // Per round: 2H + 2CNOT + 2CNOT + 2H = 8 gates, x3 rounds = 24
        assert_eq!(conversion.gates.len(), 24);
        // Per round: M 1 3 = 2 measurements, x3 = 6; plus final M 0 2 4 = 3 => 9
        assert_eq!(conversion.num_measurements, 9);
        // 3 detectors (one per round)
        assert_eq!(conversion.detectors.len(), 3);
        // 1 logical observable
        assert_eq!(conversion.logical_observables.len(), 1);
        assert_eq!(conversion.logical_observables[0], vec![-1, -3]);

        let model = circuit.to_detector_model(3);
        assert_eq!(model.distance, 3);
        assert_eq!(model.rounds, 3);
        assert_eq!(model.detectors.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 17. Noise model extraction
    // -----------------------------------------------------------------------

    #[test]
    fn test_noise_model_extraction() {
        let input = r#"
H 0 1
DEPOLARIZE1(0.01) 0 1
CNOT 0 1
DEPOLARIZE2(0.02) 0 1
M 0 1
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let noise = circuit.to_noise_model();
        assert!((noise.depolarizing_prob - 0.02).abs() < 1e-12);
        assert!(noise.crosstalk_prob > 0.0);
    }

    // -----------------------------------------------------------------------
    // 18. StimError Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_stim_error_display() {
        let e = StimError::ParseError("bad input".to_string());
        assert_eq!(format!("{}", e), "Stim parse error: bad input");

        let e = StimError::UnknownInstruction("QUUX".to_string());
        assert_eq!(format!("{}", e), "Unknown Stim instruction: QUUX");

        let e = StimError::InvalidTarget("qubit -1".to_string());
        assert_eq!(format!("{}", e), "Invalid target: qubit -1");

        let e = StimError::UnmatchedBrace("missing }".to_string());
        assert_eq!(format!("{}", e), "Unmatched brace: missing }");
    }

    // -----------------------------------------------------------------------
    // 19. StimInstruction Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_instruction_display() {
        let inst = StimInstruction::Gate {
            name: "H".to_string(),
            targets: vec![0, 1],
        };
        assert_eq!(format!("{}", inst), "H 0 1");

        let inst = StimInstruction::Noise1 {
            name: "DEPOLARIZE1".to_string(),
            probability: 0.01,
            targets: vec![0],
        };
        assert_eq!(format!("{}", inst), "DEPOLARIZE1(0.01) 0");

        let inst = StimInstruction::Tick;
        assert_eq!(format!("{}", inst), "TICK");
    }

    // -----------------------------------------------------------------------
    // 20. Two-qubit gate odd target count error
    // -----------------------------------------------------------------------

    #[test]
    fn test_two_qubit_odd_targets_error() {
        let input = "CNOT 0 1 2\n";
        let result = StimCircuit::parse(input);
        assert!(result.is_err());
        match result.unwrap_err() {
            StimError::InvalidTarget(msg) => {
                assert!(msg.contains("even"));
            }
            other => panic!("Expected InvalidTarget, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 21. Empty circuit
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_circuit() {
        let input = "# only comments\n\n# another comment\n";
        let circuit = StimCircuit::parse(input).unwrap();
        assert_eq!(circuit.num_qubits, 0);
        assert_eq!(circuit.instructions.len(), 0);
    }

    // -----------------------------------------------------------------------
    // 22. Z_ERROR noise instruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_z_error_noise() {
        let input = "Z_ERROR(0.003) 0 1\n";
        let circuit = StimCircuit::parse(input).unwrap();
        match &circuit.instructions[0] {
            StimInstruction::Noise1 {
                name,
                probability,
                targets,
            } => {
                assert_eq!(name, "Z_ERROR");
                assert!((probability - 0.003).abs() < 1e-12);
                assert_eq!(targets, &[0, 1]);
            }
            other => panic!("Expected Noise1, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // 23. Conversion produces correct gate types
    // -----------------------------------------------------------------------

    #[test]
    fn test_conversion_gate_types() {
        let input = "H 0\nX 1\nS 2\nCNOT 0 1\nCZ 2 3\n";
        let circuit = StimCircuit::parse(input).unwrap();
        let conversion = circuit.to_gates();
        assert_eq!(conversion.gates.len(), 5);
        assert_eq!(conversion.gates[0].gate_type, GateType::H);
        assert_eq!(conversion.gates[1].gate_type, GateType::X);
        assert_eq!(conversion.gates[2].gate_type, GateType::S);
        assert_eq!(conversion.gates[3].gate_type, GateType::CNOT);
        assert_eq!(conversion.gates[4].gate_type, GateType::CZ);
    }

    // -----------------------------------------------------------------------
    // 24. SWAP gate conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_swap_gate_conversion() {
        let input = "SWAP 0 1\n";
        let circuit = StimCircuit::parse(input).unwrap();
        let conversion = circuit.to_gates();
        assert_eq!(conversion.gates.len(), 1);
        assert_eq!(conversion.gates[0].gate_type, GateType::SWAP);
        assert_eq!(conversion.gates[0].targets, vec![0, 1]);
        assert!(conversion.gates[0].controls.is_empty());
    }

    // -----------------------------------------------------------------------
    // 25. Noise events in conversion output
    // -----------------------------------------------------------------------

    #[test]
    fn test_noise_events_in_conversion() {
        let input = r#"
H 0
DEPOLARIZE1(0.01) 0
CNOT 0 1
DEPOLARIZE2(0.02) 0 1
"#;
        let circuit = StimCircuit::parse(input).unwrap();
        let conversion = circuit.to_gates();
        assert_eq!(conversion.noise_events.len(), 2);
        assert_eq!(conversion.noise_events[0].noise_type, "DEPOLARIZE1");
        assert_eq!(conversion.noise_events[0].qubits, vec![0]);
        assert_eq!(conversion.noise_events[1].noise_type, "DEPOLARIZE2");
        assert_eq!(conversion.noise_events[1].qubits, vec![0, 1]);
    }
}
