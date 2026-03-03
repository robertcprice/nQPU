//! Quantum Intermediate Representation (QIR) Import/Export
//!
//! Implements a simplified text-based QIR format compatible with the Azure
//! Quantum Development Kit (QDK).  QIR is an LLVM-based intermediate
//! representation for quantum programs; this module captures the essential
//! semantics without depending on a full LLVM toolchain.
//!
//! # Capabilities
//!
//! - **Parse** text QIR (simplified LLVM-like syntax) into a [`QIRProgram`].
//! - **Emit** a [`QIRProgram`] back to text QIR.
//! - **Convert** between QIR and the internal [`NQPUCircuit`] representation.
//! - **Optimize** QIR programs (identity removal, rotation merging).
//! - **Validate** against Azure target profiles (Base, Adaptive, Full).
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::qir::*;
//!
//! let src = r#"
//! ; QIR Version 1
//! ; num_qubits 2
//! ; num_classical_bits 2
//! ; entry_point @main
//! call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 0 to %Qubit*))
//! call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Qubit* inttoptr (i64 1 to %Qubit*))
//! call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Result* inttoptr (i64 0 to %Result*))
//! "#;
//!
//! let program = QIRParser::new().parse(src).unwrap();
//! assert_eq!(program.num_qubits, 2);
//! assert_eq!(program.instructions.len(), 3);
//! ```

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors produced during QIR parsing, emission, conversion, or validation.
#[derive(Clone, Debug, PartialEq)]
pub enum QIRError {
    /// A line could not be parsed into a valid QIR instruction.
    ParseError(String),
    /// An instruction references a qubit index that exceeds the declared count.
    QubitOutOfRange { index: usize, num_qubits: usize },
    /// An instruction references a classical bit that exceeds the declared count.
    ClassicalBitOutOfRange { index: usize, num_cbits: usize },
    /// The program violates a target-profile constraint.
    ProfileViolation(String),
    /// A gate name is not recognized during conversion.
    UnknownGate(String),
    /// The program header is missing required metadata.
    InvalidHeader(String),
    /// A rotation angle could not be parsed as a floating-point number.
    InvalidAngle(String),
}

impl fmt::Display for QIRError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QIRError::ParseError(msg) => write!(f, "QIR parse error: {}", msg),
            QIRError::QubitOutOfRange { index, num_qubits } => {
                write!(
                    f,
                    "qubit index {} out of range (num_qubits = {})",
                    index, num_qubits
                )
            }
            QIRError::ClassicalBitOutOfRange { index, num_cbits } => {
                write!(
                    f,
                    "classical bit {} out of range (num_cbits = {})",
                    index, num_cbits
                )
            }
            QIRError::ProfileViolation(msg) => write!(f, "profile violation: {}", msg),
            QIRError::UnknownGate(name) => write!(f, "unknown gate: {}", name),
            QIRError::InvalidHeader(msg) => write!(f, "invalid header: {}", msg),
            QIRError::InvalidAngle(msg) => write!(f, "invalid angle: {}", msg),
        }
    }
}

impl std::error::Error for QIRError {}

/// Result type alias for QIR operations.
pub type Result<T> = std::result::Result<T, QIRError>;

// ============================================================
// ENUMS & CONFIG
// ============================================================

/// QIR specification version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QIRVersion {
    V1,
    V2,
}

impl Default for QIRVersion {
    fn default() -> Self {
        QIRVersion::V1
    }
}

impl fmt::Display for QIRVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QIRVersion::V1 => write!(f, "1"),
            QIRVersion::V2 => write!(f, "2"),
        }
    }
}

/// Azure Quantum target profile governing which features are permitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetProfile {
    /// No mid-circuit measurement, no classical feedback.
    BaseProfile,
    /// Mid-circuit measurement allowed; classical feedback allowed.
    AdaptiveProfile,
    /// All QIR features permitted.
    FullComputation,
}

impl Default for TargetProfile {
    fn default() -> Self {
        TargetProfile::FullComputation
    }
}

/// Output format for QIR emission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable LLVM-like text.
    Text,
    /// Compact binary encoding (placeholder -- emits text in this implementation).
    Binary,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Text
    }
}

/// Configuration for QIR operations (builder pattern).
#[derive(Clone, Debug)]
pub struct QIRConfig {
    pub version: QIRVersion,
    pub target_profile: TargetProfile,
    pub output_format: OutputFormat,
    pub optimize: bool,
}

impl Default for QIRConfig {
    fn default() -> Self {
        QIRConfig {
            version: QIRVersion::V1,
            target_profile: TargetProfile::FullComputation,
            output_format: OutputFormat::Text,
            optimize: true,
        }
    }
}

impl QIRConfig {
    /// Create a new config builder starting with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the QIR version.
    pub fn version(mut self, v: QIRVersion) -> Self {
        self.version = v;
        self
    }

    /// Set the target profile.
    pub fn target_profile(mut self, p: TargetProfile) -> Self {
        self.target_profile = p;
        self
    }

    /// Set the output format.
    pub fn output_format(mut self, f: OutputFormat) -> Self {
        self.output_format = f;
        self
    }

    /// Enable or disable optimization during emission.
    pub fn optimize(mut self, opt: bool) -> Self {
        self.optimize = opt;
        self
    }
}

// ============================================================
// QIR INSTRUCTION SET
// ============================================================

/// An operand passed to a QIR `Call` instruction.
#[derive(Clone, Debug, PartialEq)]
pub enum QIROperand {
    Qubit(usize),
    ClassicalBit(usize),
    Float(f64),
    Int(i64),
}

/// A single QIR instruction.
#[derive(Clone, Debug, PartialEq)]
pub enum QIRInstruction {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Rx(usize, f64),
    Ry(usize, f64),
    Rz(usize, f64),
    CNOT(usize, usize),
    CZ(usize, usize),
    SWAP(usize, usize),
    /// Measure qubit into classical bit.
    Measure(usize, usize),
    Reset(usize),
    Barrier(Vec<usize>),
    /// Classical-conditioned block: if classical_bit then execute body.
    If(usize, Vec<QIRInstruction>),
    /// Generic function call with operands.
    Call(String, Vec<QIROperand>),
}

// ============================================================
// QIR METADATA & PROGRAM
// ============================================================

/// Metadata attached to a QIR program.
#[derive(Clone, Debug, PartialEq)]
pub struct QIRMetadata {
    pub source_language: String,
    pub compiler_version: String,
    pub target: String,
    pub creation_date: String,
}

impl Default for QIRMetadata {
    fn default() -> Self {
        QIRMetadata {
            source_language: "nQPU".to_string(),
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            target: "simulator".to_string(),
            creation_date: String::new(),
        }
    }
}

/// A complete QIR program.
#[derive(Clone, Debug)]
pub struct QIRProgram {
    pub num_qubits: usize,
    pub num_classical_bits: usize,
    pub instructions: Vec<QIRInstruction>,
    pub metadata: QIRMetadata,
    pub entry_point: String,
}

impl QIRProgram {
    /// Create a new empty program.
    pub fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        QIRProgram {
            num_qubits,
            num_classical_bits,
            instructions: Vec::new(),
            metadata: QIRMetadata::default(),
            entry_point: "@main".to_string(),
        }
    }
}

// ============================================================
// PARSER
// ============================================================

/// Parses text QIR into a [`QIRProgram`].
pub struct QIRParser {
    _private: (),
}

impl QIRParser {
    pub fn new() -> Self {
        QIRParser { _private: () }
    }

    /// Parse a text QIR source into a [`QIRProgram`].
    pub fn parse(&self, input: &str) -> Result<QIRProgram> {
        let mut num_qubits: usize = 0;
        let mut num_classical_bits: usize = 0;
        let mut entry_point = "@main".to_string();
        let mut instructions: Vec<QIRInstruction> = Vec::new();
        let mut metadata = QIRMetadata::default();

        for raw_line in input.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            // Header comments
            if line.starts_with(';') {
                self.parse_header_comment(
                    line,
                    &mut num_qubits,
                    &mut num_classical_bits,
                    &mut entry_point,
                    &mut metadata,
                );
                continue;
            }

            // Instruction lines
            if line.starts_with("call ") || line.starts_with("call\t") {
                let inst = self.parse_call_line(line)?;
                instructions.push(inst);
            }
        }

        // Infer qubit/cbit counts from instructions if header did not specify.
        if num_qubits == 0 {
            num_qubits = self.infer_max_qubit(&instructions) + 1;
        }
        if num_classical_bits == 0 {
            num_classical_bits = self.infer_max_cbit(&instructions).map_or(0, |m| m + 1);
        }

        Ok(QIRProgram {
            num_qubits,
            num_classical_bits,
            instructions,
            metadata,
            entry_point,
        })
    }

    /// Parse a single QIR instruction from a standalone call line.
    pub fn parse_instruction(&self, line: &str) -> Result<QIRInstruction> {
        let trimmed = line.trim();
        if trimmed.starts_with("call ") || trimmed.starts_with("call\t") {
            self.parse_call_line(trimmed)
        } else {
            Err(QIRError::ParseError(format!(
                "expected 'call' instruction, got: {}",
                trimmed
            )))
        }
    }

    // ---- private helpers ----

    fn parse_header_comment(
        &self,
        line: &str,
        num_qubits: &mut usize,
        num_cbits: &mut usize,
        entry_point: &mut String,
        metadata: &mut QIRMetadata,
    ) {
        let content = line.trim_start_matches(';').trim();
        if let Some(rest) = content.strip_prefix("num_qubits") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                *num_qubits = n;
            }
        } else if let Some(rest) = content.strip_prefix("num_classical_bits") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                *num_cbits = n;
            }
        } else if let Some(rest) = content.strip_prefix("entry_point") {
            *entry_point = rest.trim().to_string();
        } else if let Some(rest) = content.strip_prefix("source_language") {
            metadata.source_language = rest.trim().to_string();
        } else if let Some(rest) = content.strip_prefix("compiler_version") {
            metadata.compiler_version = rest.trim().to_string();
        } else if let Some(rest) = content.strip_prefix("target") {
            metadata.target = rest.trim().to_string();
        } else if let Some(rest) = content.strip_prefix("creation_date") {
            metadata.creation_date = rest.trim().to_string();
        }
        // QIR Version comments are informational only -- we always parse V1 syntax.
    }

    fn parse_call_line(&self, line: &str) -> Result<QIRInstruction> {
        // General form:
        //   call void @__quantum__qis__<gate>__body(<operands>)
        // Extract the function name between '@' and '('
        let at_pos = line
            .find('@')
            .ok_or_else(|| QIRError::ParseError(format!("no '@' in call: {}", line)))?;
        let paren_pos = line[at_pos..]
            .find('(')
            .map(|p| p + at_pos)
            .ok_or_else(|| QIRError::ParseError(format!("no '(' in call: {}", line)))?;
        let func_name = &line[at_pos + 1..paren_pos];

        // Extract operand substring between outermost '(' and last ')'
        let args_start = paren_pos + 1;
        let args_end = line
            .rfind(')')
            .ok_or_else(|| QIRError::ParseError(format!("no ')' in call: {}", line)))?;
        let args_str = &line[args_start..args_end];

        // Parse qubit / result indices from operand list.
        let qubit_indices = self.extract_qubit_indices(args_str);
        let result_indices = self.extract_result_indices(args_str);
        let float_args = self.extract_float_args(args_str);

        // Map known QIS function names to instructions.
        self.map_function_to_instruction(func_name, &qubit_indices, &result_indices, &float_args)
    }

    fn map_function_to_instruction(
        &self,
        func_name: &str,
        qubits: &[usize],
        results: &[usize],
        floats: &[f64],
    ) -> Result<QIRInstruction> {
        // Normalize: strip the common prefix
        let name = func_name
            .strip_prefix("__quantum__qis__")
            .unwrap_or(func_name);
        // Strip trailing __body / __adj / __ctl
        let name = name
            .trim_end_matches("__body")
            .trim_end_matches("__adj")
            .trim_end_matches("__ctl");

        match name {
            "h" => Ok(QIRInstruction::H(*qubits.first().unwrap_or(&0))),
            "x" => Ok(QIRInstruction::X(*qubits.first().unwrap_or(&0))),
            "y" => Ok(QIRInstruction::Y(*qubits.first().unwrap_or(&0))),
            "z" => Ok(QIRInstruction::Z(*qubits.first().unwrap_or(&0))),
            "s" => Ok(QIRInstruction::S(*qubits.first().unwrap_or(&0))),
            "t" => Ok(QIRInstruction::T(*qubits.first().unwrap_or(&0))),
            "rx" => {
                let angle = floats.first().copied().unwrap_or(0.0);
                Ok(QIRInstruction::Rx(*qubits.first().unwrap_or(&0), angle))
            }
            "ry" => {
                let angle = floats.first().copied().unwrap_or(0.0);
                Ok(QIRInstruction::Ry(*qubits.first().unwrap_or(&0), angle))
            }
            "rz" => {
                let angle = floats.first().copied().unwrap_or(0.0);
                Ok(QIRInstruction::Rz(*qubits.first().unwrap_or(&0), angle))
            }
            "cnot" | "cx" => {
                let q0 = *qubits.first().unwrap_or(&0);
                let q1 = *qubits.get(1).unwrap_or(&0);
                Ok(QIRInstruction::CNOT(q0, q1))
            }
            "cz" => {
                let q0 = *qubits.first().unwrap_or(&0);
                let q1 = *qubits.get(1).unwrap_or(&0);
                Ok(QIRInstruction::CZ(q0, q1))
            }
            "swap" => {
                let q0 = *qubits.first().unwrap_or(&0);
                let q1 = *qubits.get(1).unwrap_or(&0);
                Ok(QIRInstruction::SWAP(q0, q1))
            }
            "mz" | "m" | "measure" => {
                let q = *qubits.first().unwrap_or(&0);
                let c = *results.first().unwrap_or(&0);
                Ok(QIRInstruction::Measure(q, c))
            }
            "reset" => Ok(QIRInstruction::Reset(*qubits.first().unwrap_or(&0))),
            _ => {
                // Generic Call fallback
                let mut operands = Vec::new();
                for &q in qubits {
                    operands.push(QIROperand::Qubit(q));
                }
                for &r in results {
                    operands.push(QIROperand::ClassicalBit(r));
                }
                for &f in floats {
                    operands.push(QIROperand::Float(f));
                }
                Ok(QIRInstruction::Call(func_name.to_string(), operands))
            }
        }
    }

    fn extract_qubit_indices(&self, args: &str) -> Vec<usize> {
        self.extract_typed_indices(args, "%Qubit*")
    }

    fn extract_result_indices(&self, args: &str) -> Vec<usize> {
        self.extract_typed_indices(args, "%Result*")
    }

    /// Extract `inttoptr (i64 <N> to %Type*)` indices for a given type.
    fn extract_typed_indices(&self, args: &str, type_marker: &str) -> Vec<usize> {
        let mut indices = Vec::new();
        let mut search_from = 0;
        while let Some(pos) = args[search_from..].find(type_marker) {
            let abs_pos = search_from + pos;
            // Find the nearest preceding "inttoptr" for this particular marker occurrence.
            // Search within the segment between the previous marker end (or start) and this marker.
            let segment_start = search_from;
            let before_marker = &args[segment_start..abs_pos];
            if let Some(rel_inttoptr) = before_marker.rfind("inttoptr") {
                let inttoptr_segment = &before_marker[rel_inttoptr..];
                if let Some(idx) = self.extract_i64_from_inttoptr(inttoptr_segment) {
                    indices.push(idx as usize);
                }
            }
            search_from = abs_pos + type_marker.len();
        }
        indices
    }

    fn extract_i64_from_inttoptr(&self, segment: &str) -> Option<i64> {
        // Pattern: "inttoptr (i64 <N> to"
        let after_i64 = segment.find("i64 ").map(|p| p + 4)?;
        let rest = &segment[after_i64..];
        let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')?;
        rest[..end].parse::<i64>().ok()
    }

    fn extract_float_args(&self, args: &str) -> Vec<f64> {
        // Look for `double <value>` patterns in the args string.
        let mut floats = Vec::new();
        let mut search_from = 0;
        while let Some(pos) = args[search_from..].find("double ") {
            let abs_pos = search_from + pos + 7;
            let rest = &args[abs_pos..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+')
                .unwrap_or(rest.len());
            if let Ok(v) = rest[..end].parse::<f64>() {
                floats.push(v);
            }
            search_from = abs_pos + end;
        }
        floats
    }

    fn infer_max_qubit(&self, instructions: &[QIRInstruction]) -> usize {
        let mut max_q: usize = 0;
        for inst in instructions {
            for q in Self::instruction_qubits(inst) {
                if q > max_q {
                    max_q = q;
                }
            }
        }
        max_q
    }

    fn infer_max_cbit(&self, instructions: &[QIRInstruction]) -> Option<usize> {
        let mut max_c: Option<usize> = None;
        for inst in instructions {
            if let QIRInstruction::Measure(_, c) = inst {
                max_c = Some(max_c.map_or(*c, |m: usize| m.max(*c)));
            }
        }
        max_c
    }

    fn instruction_qubits(inst: &QIRInstruction) -> Vec<usize> {
        match inst {
            QIRInstruction::H(q)
            | QIRInstruction::X(q)
            | QIRInstruction::Y(q)
            | QIRInstruction::Z(q)
            | QIRInstruction::S(q)
            | QIRInstruction::T(q)
            | QIRInstruction::Rx(q, _)
            | QIRInstruction::Ry(q, _)
            | QIRInstruction::Rz(q, _)
            | QIRInstruction::Reset(q) => vec![*q],
            QIRInstruction::CNOT(a, b)
            | QIRInstruction::CZ(a, b)
            | QIRInstruction::SWAP(a, b) => vec![*a, *b],
            QIRInstruction::Measure(q, _) => vec![*q],
            QIRInstruction::Barrier(qs) => qs.clone(),
            QIRInstruction::If(_, body) => {
                body.iter().flat_map(Self::instruction_qubits).collect()
            }
            QIRInstruction::Call(_, ops) => ops
                .iter()
                .filter_map(|o| match o {
                    QIROperand::Qubit(q) => Some(*q),
                    _ => None,
                })
                .collect(),
        }
    }
}

// ============================================================
// EMITTER
// ============================================================

/// Emits a [`QIRProgram`] as text or binary QIR.
pub struct QIREmitter {
    _private: (),
}

/// Magic bytes identifying the nQPU binary QIR format.
pub const QIR_BINARY_MAGIC: &[u8; 4] = b"nQIR";
/// Current binary QIR format version.
pub const QIR_BINARY_VERSION: u8 = 1;

// Opcodes for binary QIR encoding.
const OP_H: u8 = 0x01;
const OP_X: u8 = 0x02;
const OP_Y: u8 = 0x03;
const OP_Z: u8 = 0x04;
const OP_S: u8 = 0x05;
const OP_T: u8 = 0x06;
const OP_RX: u8 = 0x10;
const OP_RY: u8 = 0x11;
const OP_RZ: u8 = 0x12;
const OP_CNOT: u8 = 0x20;
const OP_CZ: u8 = 0x21;
const OP_SWAP: u8 = 0x22;
const OP_MEASURE: u8 = 0x30;
const OP_RESET: u8 = 0x31;
const OP_BARRIER: u8 = 0x40;
const OP_IF: u8 = 0x50;
const OP_END_IF: u8 = 0x51;
const OP_CALL: u8 = 0x60;

impl QIREmitter {
    pub fn new() -> Self {
        QIREmitter { _private: () }
    }

    /// Emit a QIR program in the format specified by the config.
    pub fn emit_with_config(&self, program: &QIRProgram, config: &QIRConfig) -> Vec<u8> {
        match config.output_format {
            OutputFormat::Text => self.emit(program).into_bytes(),
            OutputFormat::Binary => self.emit_binary(program),
        }
    }

    /// Emit a full text QIR program.
    pub fn emit(&self, program: &QIRProgram) -> String {
        let mut out = String::new();
        out.push_str(&self.emit_header(program));
        out.push('\n');
        out.push_str(&self.emit_entry_point(program));
        out.push('\n');
        for inst in &program.instructions {
            out.push_str(&self.emit_instruction(inst));
            out.push('\n');
        }
        out
    }

    /// Emit header metadata as QIR comments.
    pub fn emit_header(&self, program: &QIRProgram) -> String {
        let mut h = String::new();
        h.push_str("; QIR Version 1\n");
        h.push_str(&format!("; num_qubits {}\n", program.num_qubits));
        h.push_str(&format!(
            "; num_classical_bits {}\n",
            program.num_classical_bits
        ));
        if !program.metadata.source_language.is_empty() {
            h.push_str(&format!(
                "; source_language {}\n",
                program.metadata.source_language
            ));
        }
        if !program.metadata.compiler_version.is_empty() {
            h.push_str(&format!(
                "; compiler_version {}\n",
                program.metadata.compiler_version
            ));
        }
        if !program.metadata.target.is_empty() {
            h.push_str(&format!("; target {}\n", program.metadata.target));
        }
        if !program.metadata.creation_date.is_empty() {
            h.push_str(&format!(
                "; creation_date {}\n",
                program.metadata.creation_date
            ));
        }
        h
    }

    /// Emit the entry-point declaration.
    pub fn emit_entry_point(&self, program: &QIRProgram) -> String {
        format!("; entry_point {}\n", program.entry_point)
    }

    /// Emit a single instruction as a QIR call line.
    pub fn emit_instruction(&self, inst: &QIRInstruction) -> String {
        match inst {
            QIRInstruction::H(q) => self.emit_single_qubit_gate("h", *q),
            QIRInstruction::X(q) => self.emit_single_qubit_gate("x", *q),
            QIRInstruction::Y(q) => self.emit_single_qubit_gate("y", *q),
            QIRInstruction::Z(q) => self.emit_single_qubit_gate("z", *q),
            QIRInstruction::S(q) => self.emit_single_qubit_gate("s", *q),
            QIRInstruction::T(q) => self.emit_single_qubit_gate("t", *q),
            QIRInstruction::Rx(q, angle) => self.emit_rotation_gate("rx", *q, *angle),
            QIRInstruction::Ry(q, angle) => self.emit_rotation_gate("ry", *q, *angle),
            QIRInstruction::Rz(q, angle) => self.emit_rotation_gate("rz", *q, *angle),
            QIRInstruction::CNOT(c, t) => self.emit_two_qubit_gate("cnot", *c, *t),
            QIRInstruction::CZ(a, b) => self.emit_two_qubit_gate("cz", *a, *b),
            QIRInstruction::SWAP(a, b) => self.emit_two_qubit_gate("swap", *a, *b),
            QIRInstruction::Measure(q, c) => {
                format!(
                    "call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 {} to %Qubit*), %Result* inttoptr (i64 {} to %Result*))",
                    q, c
                )
            }
            QIRInstruction::Reset(q) => self.emit_single_qubit_gate("reset", *q),
            QIRInstruction::Barrier(qs) => {
                let qubit_str: Vec<String> = qs
                    .iter()
                    .map(|q| format!("%Qubit* inttoptr (i64 {} to %Qubit*)", q))
                    .collect();
                format!(
                    "call void @__quantum__qis__barrier__body({})",
                    qubit_str.join(", ")
                )
            }
            QIRInstruction::If(cbit, body) => {
                let mut s = format!(
                    "; if (result {} == 1) {{\n",
                    cbit
                );
                for b in body {
                    s.push_str("  ");
                    s.push_str(&self.emit_instruction(b));
                    s.push('\n');
                }
                s.push_str("; }");
                s
            }
            QIRInstruction::Call(name, ops) => {
                let args: Vec<String> = ops
                    .iter()
                    .map(|o| match o {
                        QIROperand::Qubit(q) => {
                            format!("%Qubit* inttoptr (i64 {} to %Qubit*)", q)
                        }
                        QIROperand::ClassicalBit(c) => {
                            format!("%Result* inttoptr (i64 {} to %Result*)", c)
                        }
                        QIROperand::Float(v) => format!("double {}", v),
                        QIROperand::Int(v) => format!("i64 {}", v),
                    })
                    .collect();
                format!("call void @{}({})", name, args.join(", "))
            }
        }
    }

    fn emit_single_qubit_gate(&self, name: &str, q: usize) -> String {
        format!(
            "call void @__quantum__qis__{}__body(%Qubit* inttoptr (i64 {} to %Qubit*))",
            name, q
        )
    }

    fn emit_two_qubit_gate(&self, name: &str, q0: usize, q1: usize) -> String {
        format!(
            "call void @__quantum__qis__{}__body(%Qubit* inttoptr (i64 {} to %Qubit*), %Qubit* inttoptr (i64 {} to %Qubit*))",
            name, q0, q1
        )
    }

    fn emit_rotation_gate(&self, name: &str, q: usize, angle: f64) -> String {
        format!(
            "call void @__quantum__qis__{}__body(double {}, %Qubit* inttoptr (i64 {} to %Qubit*))",
            name, angle, q
        )
    }

    // --------------------------------------------------------
    // Binary QIR Encoding
    // --------------------------------------------------------

    /// Emit a compact binary encoding of a QIR program.
    ///
    /// Format:
    /// - 4 bytes: magic "nQIR"
    /// - 1 byte: version
    /// - 4 bytes: num_qubits (u32 LE)
    /// - 4 bytes: num_classical_bits (u32 LE)
    /// - 4 bytes: num_instructions (u32 LE)
    /// - N instruction encodings (opcode + operands)
    pub fn emit_binary(&self, program: &QIRProgram) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.extend_from_slice(QIR_BINARY_MAGIC);
        buf.push(QIR_BINARY_VERSION);
        buf.extend_from_slice(&(program.num_qubits as u32).to_le_bytes());
        buf.extend_from_slice(&(program.num_classical_bits as u32).to_le_bytes());
        buf.extend_from_slice(&(program.instructions.len() as u32).to_le_bytes());

        // Instructions
        for inst in &program.instructions {
            self.encode_instruction(&mut buf, inst);
        }

        buf
    }

    /// Decode a binary QIR program back into a QIRProgram.
    pub fn decode_binary(data: &[u8]) -> Result<QIRProgram> {
        if data.len() < 17 {
            return Err(QIRError::ParseError("Binary QIR too short".to_string()));
        }
        if &data[0..4] != QIR_BINARY_MAGIC {
            return Err(QIRError::ParseError("Invalid QIR binary magic".to_string()));
        }
        if data[4] != QIR_BINARY_VERSION {
            return Err(QIRError::ParseError(format!(
                "Unsupported QIR binary version {}",
                data[4]
            )));
        }

        let num_qubits = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
        let num_cbits = u32::from_le_bytes([data[9], data[10], data[11], data[12]]) as usize;
        let num_insts = u32::from_le_bytes([data[13], data[14], data[15], data[16]]) as usize;

        let mut offset = 17;
        let mut instructions = Vec::with_capacity(num_insts);
        for _ in 0..num_insts {
            let (inst, consumed) = Self::decode_instruction(&data[offset..])?;
            instructions.push(inst);
            offset += consumed;
        }

        Ok(QIRProgram {
            num_qubits,
            num_classical_bits: num_cbits,
            instructions,
            metadata: QIRMetadata::default(),
            entry_point: "main".to_string(),
        })
    }

    fn encode_instruction(&self, buf: &mut Vec<u8>, inst: &QIRInstruction) {
        match inst {
            QIRInstruction::H(q) => {
                buf.push(OP_H);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::X(q) => {
                buf.push(OP_X);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::Y(q) => {
                buf.push(OP_Y);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::Z(q) => {
                buf.push(OP_Z);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::S(q) => {
                buf.push(OP_S);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::T(q) => {
                buf.push(OP_T);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::Rx(q, angle) => {
                buf.push(OP_RX);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
                buf.extend_from_slice(&angle.to_le_bytes());
            }
            QIRInstruction::Ry(q, angle) => {
                buf.push(OP_RY);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
                buf.extend_from_slice(&angle.to_le_bytes());
            }
            QIRInstruction::Rz(q, angle) => {
                buf.push(OP_RZ);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
                buf.extend_from_slice(&angle.to_le_bytes());
            }
            QIRInstruction::CNOT(c, t) => {
                buf.push(OP_CNOT);
                buf.extend_from_slice(&(*c as u32).to_le_bytes());
                buf.extend_from_slice(&(*t as u32).to_le_bytes());
            }
            QIRInstruction::CZ(a, b) => {
                buf.push(OP_CZ);
                buf.extend_from_slice(&(*a as u32).to_le_bytes());
                buf.extend_from_slice(&(*b as u32).to_le_bytes());
            }
            QIRInstruction::SWAP(a, b) => {
                buf.push(OP_SWAP);
                buf.extend_from_slice(&(*a as u32).to_le_bytes());
                buf.extend_from_slice(&(*b as u32).to_le_bytes());
            }
            QIRInstruction::Measure(q, c) => {
                buf.push(OP_MEASURE);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
                buf.extend_from_slice(&(*c as u32).to_le_bytes());
            }
            QIRInstruction::Reset(q) => {
                buf.push(OP_RESET);
                buf.extend_from_slice(&(*q as u32).to_le_bytes());
            }
            QIRInstruction::Barrier(qubits) => {
                buf.push(OP_BARRIER);
                buf.extend_from_slice(&(qubits.len() as u32).to_le_bytes());
                for q in qubits {
                    buf.extend_from_slice(&(*q as u32).to_le_bytes());
                }
            }
            QIRInstruction::If(cbit, body) => {
                buf.push(OP_IF);
                buf.extend_from_slice(&(*cbit as u32).to_le_bytes());
                buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
                for child in body {
                    self.encode_instruction(buf, child);
                }
                buf.push(OP_END_IF);
            }
            QIRInstruction::Call(name, operands) => {
                buf.push(OP_CALL);
                let name_bytes = name.as_bytes();
                buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(name_bytes);
                buf.extend_from_slice(&(operands.len() as u32).to_le_bytes());
                for op in operands {
                    match op {
                        QIROperand::Qubit(q) => {
                            buf.push(0x01);
                            buf.extend_from_slice(&(*q as u32).to_le_bytes());
                        }
                        QIROperand::ClassicalBit(c) => {
                            buf.push(0x02);
                            buf.extend_from_slice(&(*c as u32).to_le_bytes());
                        }
                        QIROperand::Float(f) => {
                            buf.push(0x03);
                            buf.extend_from_slice(&f.to_le_bytes());
                        }
                        QIROperand::Int(i) => {
                            buf.push(0x04);
                            buf.extend_from_slice(&i.to_le_bytes());
                        }
                    }
                }
            }
        }
    }

    fn decode_instruction(data: &[u8]) -> Result<(QIRInstruction, usize)> {
        if data.is_empty() {
            return Err(QIRError::ParseError("Unexpected end of binary QIR".to_string()));
        }
        let op = data[0];
        match op {
            OP_H | OP_X | OP_Y | OP_Z | OP_S | OP_T | OP_RESET => {
                let q = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let inst = match op {
                    OP_H => QIRInstruction::H(q),
                    OP_X => QIRInstruction::X(q),
                    OP_Y => QIRInstruction::Y(q),
                    OP_Z => QIRInstruction::Z(q),
                    OP_S => QIRInstruction::S(q),
                    OP_T => QIRInstruction::T(q),
                    OP_RESET => QIRInstruction::Reset(q),
                    _ => unreachable!(),
                };
                Ok((inst, 5))
            }
            OP_RX | OP_RY | OP_RZ => {
                let q = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let angle = f64::from_le_bytes([
                    data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12],
                ]);
                let inst = match op {
                    OP_RX => QIRInstruction::Rx(q, angle),
                    OP_RY => QIRInstruction::Ry(q, angle),
                    OP_RZ => QIRInstruction::Rz(q, angle),
                    _ => unreachable!(),
                };
                Ok((inst, 13))
            }
            OP_CNOT | OP_CZ | OP_SWAP | OP_MEASURE => {
                let a = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let b = u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
                let inst = match op {
                    OP_CNOT => QIRInstruction::CNOT(a, b),
                    OP_CZ => QIRInstruction::CZ(a, b),
                    OP_SWAP => QIRInstruction::SWAP(a, b),
                    OP_MEASURE => QIRInstruction::Measure(a, b),
                    _ => unreachable!(),
                };
                Ok((inst, 9))
            }
            OP_BARRIER => {
                let count =
                    u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let mut qubits = Vec::with_capacity(count);
                for i in 0..count {
                    let off = 5 + i * 4;
                    let q = u32::from_le_bytes([
                        data[off],
                        data[off + 1],
                        data[off + 2],
                        data[off + 3],
                    ]) as usize;
                    qubits.push(q);
                }
                Ok((QIRInstruction::Barrier(qubits), 5 + count * 4))
            }
            OP_IF => {
                let cbit =
                    u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let body_len =
                    u32::from_le_bytes([data[5], data[6], data[7], data[8]]) as usize;
                let mut offset = 9;
                let mut body = Vec::with_capacity(body_len);
                for _ in 0..body_len {
                    let (inst, consumed) = Self::decode_instruction(&data[offset..])?;
                    body.push(inst);
                    offset += consumed;
                }
                // Consume the OP_END_IF marker
                if data.get(offset) == Some(&OP_END_IF) {
                    offset += 1;
                }
                Ok((QIRInstruction::If(cbit, body), offset))
            }
            OP_CALL => {
                let name_len =
                    u16::from_le_bytes([data[1], data[2]]) as usize;
                let name =
                    String::from_utf8_lossy(&data[3..3 + name_len]).to_string();
                let num_ops_off = 3 + name_len;
                let num_ops = u32::from_le_bytes([
                    data[num_ops_off],
                    data[num_ops_off + 1],
                    data[num_ops_off + 2],
                    data[num_ops_off + 3],
                ]) as usize;
                let mut offset = num_ops_off + 4;
                let mut operands = Vec::with_capacity(num_ops);
                for _ in 0..num_ops {
                    let kind = data[offset];
                    offset += 1;
                    match kind {
                        0x01 => {
                            let q = u32::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                            ]) as usize;
                            offset += 4;
                            operands.push(QIROperand::Qubit(q));
                        }
                        0x02 => {
                            let c = u32::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                            ]) as usize;
                            offset += 4;
                            operands.push(QIROperand::ClassicalBit(c));
                        }
                        0x03 => {
                            let f = f64::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                                data[offset + 4],
                                data[offset + 5],
                                data[offset + 6],
                                data[offset + 7],
                            ]);
                            offset += 8;
                            operands.push(QIROperand::Float(f));
                        }
                        0x04 => {
                            let i = i64::from_le_bytes([
                                data[offset],
                                data[offset + 1],
                                data[offset + 2],
                                data[offset + 3],
                                data[offset + 4],
                                data[offset + 5],
                                data[offset + 6],
                                data[offset + 7],
                            ]);
                            offset += 8;
                            operands.push(QIROperand::Int(i));
                        }
                        _ => {
                            return Err(QIRError::ParseError(format!(
                                "Unknown QIR operand kind 0x{:02x}",
                                kind
                            )));
                        }
                    }
                }
                Ok((QIRInstruction::Call(name, operands), offset))
            }
            _ => Err(QIRError::ParseError(format!("Unknown QIR opcode 0x{:02x}", op))),
        }
    }
}

// ============================================================
// NQPU CIRCUIT (internal representation)
// ============================================================

/// Lightweight internal circuit representation for conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct NQPUCircuit {
    /// Triples of (gate_name, qubit_indices, float_params).
    pub gates: Vec<(String, Vec<usize>, Vec<f64>)>,
    pub num_qubits: usize,
    pub num_cbits: usize,
    pub measurements: Vec<(usize, usize)>,
}

impl NQPUCircuit {
    pub fn new(num_qubits: usize, num_cbits: usize) -> Self {
        NQPUCircuit {
            gates: Vec::new(),
            num_qubits,
            num_cbits,
            measurements: Vec::new(),
        }
    }
}

// ============================================================
// CONVERTER
// ============================================================

/// Converts between [`QIRProgram`] and [`NQPUCircuit`].
pub struct QIRConverter;

impl QIRConverter {
    /// Convert a QIR program to an nQPU circuit.
    pub fn qir_to_nqpu(program: &QIRProgram) -> Result<NQPUCircuit> {
        let mut circuit = NQPUCircuit::new(program.num_qubits, program.num_classical_bits);
        Self::convert_instructions(&program.instructions, &mut circuit)?;
        Ok(circuit)
    }

    fn convert_instructions(
        instructions: &[QIRInstruction],
        circuit: &mut NQPUCircuit,
    ) -> Result<()> {
        for inst in instructions {
            match inst {
                QIRInstruction::H(q) => circuit.gates.push(("H".into(), vec![*q], vec![])),
                QIRInstruction::X(q) => circuit.gates.push(("X".into(), vec![*q], vec![])),
                QIRInstruction::Y(q) => circuit.gates.push(("Y".into(), vec![*q], vec![])),
                QIRInstruction::Z(q) => circuit.gates.push(("Z".into(), vec![*q], vec![])),
                QIRInstruction::S(q) => circuit.gates.push(("S".into(), vec![*q], vec![])),
                QIRInstruction::T(q) => circuit.gates.push(("T".into(), vec![*q], vec![])),
                QIRInstruction::Rx(q, a) => {
                    circuit.gates.push(("Rx".into(), vec![*q], vec![*a]))
                }
                QIRInstruction::Ry(q, a) => {
                    circuit.gates.push(("Ry".into(), vec![*q], vec![*a]))
                }
                QIRInstruction::Rz(q, a) => {
                    circuit.gates.push(("Rz".into(), vec![*q], vec![*a]))
                }
                QIRInstruction::CNOT(c, t) => {
                    circuit.gates.push(("CNOT".into(), vec![*c, *t], vec![]))
                }
                QIRInstruction::CZ(a, b) => {
                    circuit.gates.push(("CZ".into(), vec![*a, *b], vec![]))
                }
                QIRInstruction::SWAP(a, b) => {
                    circuit.gates.push(("SWAP".into(), vec![*a, *b], vec![]))
                }
                QIRInstruction::Measure(q, c) => {
                    circuit.measurements.push((*q, *c));
                }
                QIRInstruction::Reset(q) => {
                    circuit.gates.push(("Reset".into(), vec![*q], vec![]))
                }
                QIRInstruction::Barrier(_) => {
                    // Barriers are scheduling hints -- no gate emitted.
                }
                QIRInstruction::If(_, body) => {
                    Self::convert_instructions(body, circuit)?;
                }
                QIRInstruction::Call(name, ops) => {
                    let qubits: Vec<usize> = ops
                        .iter()
                        .filter_map(|o| match o {
                            QIROperand::Qubit(q) => Some(*q),
                            _ => None,
                        })
                        .collect();
                    let params: Vec<f64> = ops
                        .iter()
                        .filter_map(|o| match o {
                            QIROperand::Float(v) => Some(*v),
                            _ => None,
                        })
                        .collect();
                    circuit.gates.push((name.clone(), qubits, params));
                }
            }
        }
        Ok(())
    }

    /// Convert an nQPU circuit to a QIR program.
    pub fn nqpu_to_qir(circuit: &NQPUCircuit, config: &QIRConfig) -> QIRProgram {
        let mut instructions: Vec<QIRInstruction> = Vec::new();

        for (name, qubits, params) in &circuit.gates {
            let inst = match name.as_str() {
                "H" => QIRInstruction::H(qubits[0]),
                "X" => QIRInstruction::X(qubits[0]),
                "Y" => QIRInstruction::Y(qubits[0]),
                "Z" => QIRInstruction::Z(qubits[0]),
                "S" => QIRInstruction::S(qubits[0]),
                "T" => QIRInstruction::T(qubits[0]),
                "Rx" => QIRInstruction::Rx(qubits[0], params[0]),
                "Ry" => QIRInstruction::Ry(qubits[0], params[0]),
                "Rz" => QIRInstruction::Rz(qubits[0], params[0]),
                "CNOT" | "CX" => QIRInstruction::CNOT(qubits[0], qubits[1]),
                "CZ" => QIRInstruction::CZ(qubits[0], qubits[1]),
                "SWAP" => QIRInstruction::SWAP(qubits[0], qubits[1]),
                "Reset" => QIRInstruction::Reset(qubits[0]),
                _ => {
                    let mut ops: Vec<QIROperand> =
                        qubits.iter().map(|q| QIROperand::Qubit(*q)).collect();
                    ops.extend(params.iter().map(|p| QIROperand::Float(*p)));
                    QIRInstruction::Call(name.clone(), ops)
                }
            };
            instructions.push(inst);
        }

        for &(q, c) in &circuit.measurements {
            instructions.push(QIRInstruction::Measure(q, c));
        }

        let mut metadata = QIRMetadata::default();
        metadata.target = match config.target_profile {
            TargetProfile::BaseProfile => "quantinuum.qpu.h1".to_string(),
            TargetProfile::AdaptiveProfile => "quantinuum.qpu.h1-adaptive".to_string(),
            TargetProfile::FullComputation => "simulator".to_string(),
        };

        QIRProgram {
            num_qubits: circuit.num_qubits,
            num_classical_bits: circuit.num_cbits,
            instructions,
            metadata,
            entry_point: "@main".to_string(),
        }
    }
}

// ============================================================
// OPTIMIZER
// ============================================================

/// Statistics returned by the optimizer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OptimizationStats {
    pub gates_removed: usize,
    pub rotations_merged: usize,
    pub total_gate_reduction: f64,
}

/// Peep-hole optimizer for [`QIRProgram`] instruction lists.
pub struct QIROptimizer;

impl QIROptimizer {
    /// Run all optimization passes on the program, returning statistics.
    pub fn optimize(program: &mut QIRProgram) -> OptimizationStats {
        let before = program.instructions.len();
        let mut stats = OptimizationStats::default();

        stats.gates_removed += Self::remove_identity_gates(program);
        stats.rotations_merged += Self::merge_rotations(program);
        stats.gates_removed += Self::remove_redundant_resets(program);

        let after = program.instructions.len();
        stats.total_gate_reduction = if before > 0 {
            (before - after) as f64 / before as f64
        } else {
            0.0
        };
        stats
    }

    /// Remove gates that act as identity (e.g. Rz(0), Rx(0), Ry(0)).
    pub fn remove_identity_gates(program: &mut QIRProgram) -> usize {
        let before = program.instructions.len();
        program.instructions.retain(|inst| {
            match inst {
                QIRInstruction::Rx(_, angle)
                | QIRInstruction::Ry(_, angle)
                | QIRInstruction::Rz(_, angle) => angle.abs() > 1e-12,
                _ => true,
            }
        });
        before - program.instructions.len()
    }

    /// Merge consecutive same-axis rotations on the same qubit.
    pub fn merge_rotations(program: &mut QIRProgram) -> usize {
        let mut merged = 0usize;
        let mut i = 0;
        while i + 1 < program.instructions.len() {
            let can_merge = match (&program.instructions[i], &program.instructions[i + 1]) {
                (QIRInstruction::Rz(q1, _), QIRInstruction::Rz(q2, _)) => q1 == q2,
                (QIRInstruction::Rx(q1, _), QIRInstruction::Rx(q2, _)) => q1 == q2,
                (QIRInstruction::Ry(q1, _), QIRInstruction::Ry(q2, _)) => q1 == q2,
                _ => false,
            };

            if can_merge {
                let new_inst = match (&program.instructions[i], &program.instructions[i + 1]) {
                    (QIRInstruction::Rz(q, a1), QIRInstruction::Rz(_, a2)) => {
                        QIRInstruction::Rz(*q, a1 + a2)
                    }
                    (QIRInstruction::Rx(q, a1), QIRInstruction::Rx(_, a2)) => {
                        QIRInstruction::Rx(*q, a1 + a2)
                    }
                    (QIRInstruction::Ry(q, a1), QIRInstruction::Ry(_, a2)) => {
                        QIRInstruction::Ry(*q, a1 + a2)
                    }
                    _ => unreachable!(),
                };
                program.instructions[i] = new_inst;
                program.instructions.remove(i + 1);
                merged += 1;
                // Do not advance -- the merged instruction may merge with the next one.
            } else {
                i += 1;
            }
        }
        merged
    }

    /// Remove consecutive resets on the same qubit (only the first matters).
    pub fn remove_redundant_resets(program: &mut QIRProgram) -> usize {
        let mut removed = 0usize;
        let mut i = 0;
        while i + 1 < program.instructions.len() {
            if let (QIRInstruction::Reset(q1), QIRInstruction::Reset(q2)) =
                (&program.instructions[i], &program.instructions[i + 1])
            {
                if q1 == q2 {
                    program.instructions.remove(i + 1);
                    removed += 1;
                    continue;
                }
            }
            i += 1;
        }
        removed
    }
}

// ============================================================
// VALIDATOR
// ============================================================

/// Severity of a validation finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Warning,
    Error,
}

/// A single validation finding.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidationError {
    pub instruction_index: usize,
    pub message: String,
    pub severity: Severity,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sev = match self.severity {
            Severity::Warning => "warning",
            Severity::Error => "error",
        };
        write!(
            f,
            "[{}] instruction {}: {}",
            sev, self.instruction_index, self.message
        )
    }
}

/// Validates a [`QIRProgram`] against structural and profile constraints.
pub struct QIRValidator;

impl QIRValidator {
    /// Run all validation checks, returning any findings.
    pub fn validate(program: &QIRProgram, profile: &TargetProfile) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        errors.extend(Self::check_qubit_bounds(program));
        errors.extend(Self::check_profile_compliance(program, profile));
        errors
    }

    /// Check that all qubit and classical-bit indices are within declared bounds.
    pub fn check_qubit_bounds(program: &QIRProgram) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        for (idx, inst) in program.instructions.iter().enumerate() {
            for q in QIRParser::instruction_qubits(inst) {
                if q >= program.num_qubits {
                    errors.push(ValidationError {
                        instruction_index: idx,
                        message: format!(
                            "qubit {} out of range (num_qubits = {})",
                            q, program.num_qubits
                        ),
                        severity: Severity::Error,
                    });
                }
            }
            if let QIRInstruction::Measure(_, c) = inst {
                if *c >= program.num_classical_bits {
                    errors.push(ValidationError {
                        instruction_index: idx,
                        message: format!(
                            "classical bit {} out of range (num_cbits = {})",
                            c, program.num_classical_bits
                        ),
                        severity: Severity::Error,
                    });
                }
            }
        }
        errors
    }

    /// Check that the program complies with the given target profile.
    pub fn check_profile_compliance(
        program: &QIRProgram,
        profile: &TargetProfile,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        match profile {
            TargetProfile::BaseProfile => {
                // Base profile: no mid-circuit measurement, no classical feedback.
                let mut seen_measure = false;
                for (idx, inst) in program.instructions.iter().enumerate() {
                    match inst {
                        QIRInstruction::Measure(_, _) => {
                            // If there are gates after this measurement, it is mid-circuit.
                            if program.instructions[idx + 1..]
                                .iter()
                                .any(|i| !matches!(i, QIRInstruction::Measure(_, _)))
                            {
                                errors.push(ValidationError {
                                    instruction_index: idx,
                                    message: "mid-circuit measurement not allowed in BaseProfile"
                                        .to_string(),
                                    severity: Severity::Error,
                                });
                            }
                            seen_measure = true;
                        }
                        QIRInstruction::If(_, _) => {
                            errors.push(ValidationError {
                                instruction_index: idx,
                                message: "classical feedback not allowed in BaseProfile"
                                    .to_string(),
                                severity: Severity::Error,
                            });
                        }
                        _ => {}
                    }
                }
                let _ = seen_measure; // suppress warning
            }
            TargetProfile::AdaptiveProfile => {
                // Adaptive profile allows mid-circuit measurement and feedback.
                // No additional constraints beyond qubit bounds.
            }
            TargetProfile::FullComputation => {
                // Everything is allowed.
            }
        }

        errors
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ---- Config builder ----

    #[test]
    fn test_config_builder_defaults() {
        let cfg = QIRConfig::new();
        assert_eq!(cfg.version, QIRVersion::V1);
        assert_eq!(cfg.target_profile, TargetProfile::FullComputation);
        assert_eq!(cfg.output_format, OutputFormat::Text);
        assert!(cfg.optimize);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = QIRConfig::new()
            .version(QIRVersion::V2)
            .target_profile(TargetProfile::BaseProfile)
            .output_format(OutputFormat::Binary)
            .optimize(false);
        assert_eq!(cfg.version, QIRVersion::V2);
        assert_eq!(cfg.target_profile, TargetProfile::BaseProfile);
        assert_eq!(cfg.output_format, OutputFormat::Binary);
        assert!(!cfg.optimize);
    }

    // ---- Parser: single instructions ----

    #[test]
    fn test_parse_h_gate() {
        let line = "call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 0 to %Qubit*))";
        let parser = QIRParser::new();
        let inst = parser.parse_instruction(line).unwrap();
        assert_eq!(inst, QIRInstruction::H(0));
    }

    #[test]
    fn test_parse_cnot_instruction() {
        let line = "call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Qubit* inttoptr (i64 1 to %Qubit*))";
        let parser = QIRParser::new();
        let inst = parser.parse_instruction(line).unwrap();
        assert_eq!(inst, QIRInstruction::CNOT(0, 1));
    }

    #[test]
    fn test_parse_rotation_with_angle() {
        let line = "call void @__quantum__qis__rz__body(double 1.5707963, %Qubit* inttoptr (i64 2 to %Qubit*))";
        let parser = QIRParser::new();
        let inst = parser.parse_instruction(line).unwrap();
        match inst {
            QIRInstruction::Rz(q, angle) => {
                assert_eq!(q, 2);
                assert!((angle - 1.5707963).abs() < 1e-6);
            }
            _ => panic!("expected Rz, got {:?}", inst),
        }
    }

    #[test]
    fn test_parse_measure() {
        let line = "call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Result* inttoptr (i64 0 to %Result*))";
        let parser = QIRParser::new();
        let inst = parser.parse_instruction(line).unwrap();
        assert_eq!(inst, QIRInstruction::Measure(0, 0));
    }

    // ---- Parser: full program ----

    #[test]
    fn test_parse_full_program() {
        let src = r#"
; QIR Version 1
; num_qubits 3
; num_classical_bits 2
; entry_point @main
; source_language Q#
call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 0 to %Qubit*))
call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Qubit* inttoptr (i64 1 to %Qubit*))
call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Result* inttoptr (i64 0 to %Result*))
call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* inttoptr (i64 1 to %Result*))
"#;
        let parser = QIRParser::new();
        let prog = parser.parse(src).unwrap();
        assert_eq!(prog.num_qubits, 3);
        assert_eq!(prog.num_classical_bits, 2);
        assert_eq!(prog.entry_point, "@main");
        assert_eq!(prog.metadata.source_language, "Q#");
        assert_eq!(prog.instructions.len(), 4);
    }

    // ---- Emitter + roundtrip ----

    #[test]
    fn test_emit_and_reparse_roundtrip() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::CNOT(0, 1));
        prog.instructions
            .push(QIRInstruction::Rz(1, PI / 4.0));
        prog.instructions.push(QIRInstruction::Measure(0, 0));

        let emitter = QIREmitter::new();
        let text = emitter.emit(&prog);

        let parser = QIRParser::new();
        let reparsed = parser.parse(&text).unwrap();

        assert_eq!(reparsed.num_qubits, prog.num_qubits);
        assert_eq!(reparsed.instructions.len(), prog.instructions.len());

        // Check first two instructions exactly.
        assert_eq!(reparsed.instructions[0], QIRInstruction::H(0));
        assert_eq!(reparsed.instructions[1], QIRInstruction::CNOT(0, 1));

        // Rotation angle should survive roundtrip within tolerance.
        match &reparsed.instructions[2] {
            QIRInstruction::Rz(q, a) => {
                assert_eq!(*q, 1);
                assert!((a - PI / 4.0).abs() < 1e-6);
            }
            other => panic!("expected Rz, got {:?}", other),
        }

        assert_eq!(reparsed.instructions[3], QIRInstruction::Measure(0, 0));
    }

    // ---- Converter: QIR -> nQPU ----

    #[test]
    fn test_convert_qir_to_nqpu() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::CNOT(0, 1));
        prog.instructions.push(QIRInstruction::Measure(0, 0));

        let circuit = QIRConverter::qir_to_nqpu(&prog).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 2);
        assert_eq!(circuit.gates[0].0, "H");
        assert_eq!(circuit.gates[1].0, "CNOT");
        assert_eq!(circuit.measurements, vec![(0, 0)]);
    }

    // ---- Converter: nQPU -> QIR ----

    #[test]
    fn test_convert_nqpu_to_qir() {
        let mut circuit = NQPUCircuit::new(3, 2);
        circuit.gates.push(("H".into(), vec![0], vec![]));
        circuit.gates.push(("CNOT".into(), vec![0, 1], vec![]));
        circuit.gates.push(("Rz".into(), vec![2], vec![PI / 2.0]));
        circuit.measurements.push((0, 0));
        circuit.measurements.push((1, 1));

        let config = QIRConfig::new().target_profile(TargetProfile::BaseProfile);
        let prog = QIRConverter::nqpu_to_qir(&circuit, &config);

        assert_eq!(prog.num_qubits, 3);
        assert_eq!(prog.num_classical_bits, 2);
        assert_eq!(prog.instructions.len(), 5); // 3 gates + 2 measures
        assert_eq!(prog.metadata.target, "quantinuum.qpu.h1");
    }

    // ---- Roundtrip conversion ----

    #[test]
    fn test_roundtrip_conversion_preserves_gates() {
        let mut orig = NQPUCircuit::new(3, 1);
        orig.gates.push(("H".into(), vec![0], vec![]));
        orig.gates.push(("X".into(), vec![1], vec![]));
        orig.gates.push(("CNOT".into(), vec![0, 2], vec![]));
        orig.gates.push(("Rx".into(), vec![1], vec![1.23]));
        orig.measurements.push((0, 0));

        let config = QIRConfig::new();
        let prog = QIRConverter::nqpu_to_qir(&orig, &config);
        let restored = QIRConverter::qir_to_nqpu(&prog).unwrap();

        assert_eq!(restored.num_qubits, orig.num_qubits);
        assert_eq!(restored.gates.len(), orig.gates.len());
        for (a, b) in restored.gates.iter().zip(orig.gates.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            for (pa, pb) in a.2.iter().zip(b.2.iter()) {
                assert!((pa - pb).abs() < 1e-12);
            }
        }
        assert_eq!(restored.measurements, orig.measurements);
    }

    // ---- Validation: base profile ----

    #[test]
    fn test_base_profile_rejects_mid_circuit_measure() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::Measure(0, 0));
        prog.instructions.push(QIRInstruction::X(1)); // gate after measure

        let errors = QIRValidator::validate(&prog, &TargetProfile::BaseProfile);
        assert!(
            errors.iter().any(|e| e.message.contains("mid-circuit")),
            "expected mid-circuit measurement error, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_base_profile_rejects_classical_feedback() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::Measure(0, 0));
        prog.instructions
            .push(QIRInstruction::If(0, vec![QIRInstruction::X(1)]));

        let errors = QIRValidator::validate(&prog, &TargetProfile::BaseProfile);
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("classical feedback")),
            "expected classical feedback error, got: {:?}",
            errors
        );
    }

    // ---- Validation: adaptive profile ----

    #[test]
    fn test_adaptive_profile_allows_feedback() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::Measure(0, 0));
        prog.instructions
            .push(QIRInstruction::If(0, vec![QIRInstruction::X(1)]));

        let errors = QIRValidator::validate(&prog, &TargetProfile::AdaptiveProfile);
        assert!(
            errors.is_empty(),
            "adaptive profile should allow feedback, got: {:?}",
            errors
        );
    }

    // ---- Validation: qubit bounds ----

    #[test]
    fn test_qubit_bounds_validation() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::H(5)); // out of range

        let errors = QIRValidator::check_qubit_bounds(&prog);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].severity, Severity::Error);
        assert!(errors[0].message.contains("qubit 5"));
    }

    // ---- Optimizer: identity removal ----

    #[test]
    fn test_optimize_removes_identity_rotations() {
        let mut prog = QIRProgram::new(2, 0);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::Rz(0, 0.0));
        prog.instructions.push(QIRInstruction::Rx(1, 0.0));
        prog.instructions.push(QIRInstruction::X(1));

        let removed = QIROptimizer::remove_identity_gates(&mut prog);
        assert_eq!(removed, 2);
        assert_eq!(prog.instructions.len(), 2);
        assert_eq!(prog.instructions[0], QIRInstruction::H(0));
        assert_eq!(prog.instructions[1], QIRInstruction::X(1));
    }

    // ---- Optimizer: rotation merging ----

    #[test]
    fn test_optimize_merges_consecutive_rotations() {
        let mut prog = QIRProgram::new(1, 0);
        prog.instructions.push(QIRInstruction::Rz(0, PI / 4.0));
        prog.instructions.push(QIRInstruction::Rz(0, PI / 4.0));
        prog.instructions.push(QIRInstruction::H(0));

        let merged = QIROptimizer::merge_rotations(&mut prog);
        assert_eq!(merged, 1);
        assert_eq!(prog.instructions.len(), 2);
        match &prog.instructions[0] {
            QIRInstruction::Rz(q, a) => {
                assert_eq!(*q, 0);
                assert!((a - PI / 2.0).abs() < 1e-12);
            }
            other => panic!("expected merged Rz, got {:?}", other),
        }
    }

    #[test]
    fn test_optimize_merges_three_consecutive_rotations() {
        let mut prog = QIRProgram::new(1, 0);
        prog.instructions.push(QIRInstruction::Rx(0, 0.1));
        prog.instructions.push(QIRInstruction::Rx(0, 0.2));
        prog.instructions.push(QIRInstruction::Rx(0, 0.3));

        let merged = QIROptimizer::merge_rotations(&mut prog);
        assert_eq!(merged, 2);
        assert_eq!(prog.instructions.len(), 1);
        match &prog.instructions[0] {
            QIRInstruction::Rx(_, a) => assert!((a - 0.6).abs() < 1e-12),
            other => panic!("expected merged Rx, got {:?}", other),
        }
    }

    // ---- Optimizer: full pass ----

    #[test]
    fn test_full_optimization_pass() {
        let mut prog = QIRProgram::new(2, 0);
        prog.instructions.push(QIRInstruction::Rz(0, PI / 4.0));
        prog.instructions.push(QIRInstruction::Rz(0, PI / 4.0));
        prog.instructions.push(QIRInstruction::Ry(1, 0.0)); // identity
        prog.instructions.push(QIRInstruction::Reset(0));
        prog.instructions.push(QIRInstruction::Reset(0)); // redundant
        prog.instructions.push(QIRInstruction::H(0));

        let stats = QIROptimizer::optimize(&mut prog);
        // identity Ry removed (1), redundant reset removed (1), Rz merged (1)
        assert!(stats.gates_removed >= 2);
        assert_eq!(stats.rotations_merged, 1);
        assert!(stats.total_gate_reduction > 0.0);
        // Remaining: merged Rz, Reset, H = 3
        assert_eq!(prog.instructions.len(), 3);
    }

    // ---- Metadata preservation ----

    #[test]
    fn test_metadata_preservation_through_emit() {
        let mut prog = QIRProgram::new(1, 0);
        prog.metadata.source_language = "Q#".to_string();
        prog.metadata.compiler_version = "1.2.3".to_string();
        prog.metadata.target = "ionq.qpu".to_string();
        prog.metadata.creation_date = "2026-02-15".to_string();
        prog.instructions.push(QIRInstruction::H(0));

        let emitter = QIREmitter::new();
        let text = emitter.emit(&prog);

        let parser = QIRParser::new();
        let reparsed = parser.parse(&text).unwrap();

        assert_eq!(reparsed.metadata.source_language, "Q#");
        assert_eq!(reparsed.metadata.compiler_version, "1.2.3");
        assert_eq!(reparsed.metadata.target, "ionq.qpu");
        assert_eq!(reparsed.metadata.creation_date, "2026-02-15");
    }

    // ---- Multi-qubit roundtrip ----

    #[test]
    fn test_multi_qubit_circuit_roundtrip() {
        let mut circuit = NQPUCircuit::new(5, 3);
        circuit.gates.push(("H".into(), vec![0], vec![]));
        circuit.gates.push(("CNOT".into(), vec![0, 1], vec![]));
        circuit.gates.push(("CZ".into(), vec![1, 2], vec![]));
        circuit.gates.push(("SWAP".into(), vec![2, 3], vec![]));
        circuit.gates.push(("T".into(), vec![4], vec![]));
        circuit.gates.push(("S".into(), vec![3], vec![]));
        circuit.gates.push(("Ry".into(), vec![0], vec![PI / 3.0]));
        circuit.measurements.push((0, 0));
        circuit.measurements.push((1, 1));
        circuit.measurements.push((4, 2));

        let config = QIRConfig::new();
        let prog = QIRConverter::nqpu_to_qir(&circuit, &config);

        // Emit to text and reparse.
        let emitter = QIREmitter::new();
        let text = emitter.emit(&prog);
        let parser = QIRParser::new();
        let reparsed = parser.parse(&text).unwrap();

        // Convert back to nQPU.
        let restored = QIRConverter::qir_to_nqpu(&reparsed).unwrap();

        assert_eq!(restored.num_qubits, circuit.num_qubits);
        assert_eq!(restored.gates.len(), circuit.gates.len());
        assert_eq!(restored.measurements.len(), circuit.measurements.len());

        // Verify gate names match.
        for (orig, rest) in circuit.gates.iter().zip(restored.gates.iter()) {
            assert_eq!(orig.0, rest.0, "gate name mismatch");
            assert_eq!(orig.1, rest.1, "qubit indices mismatch for {}", orig.0);
        }
    }

    // ---- Error cases ----

    #[test]
    fn test_parse_invalid_line_returns_error() {
        let parser = QIRParser::new();
        let result = parser.parse_instruction("not a valid instruction");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let err = QIRError::QubitOutOfRange {
            index: 5,
            num_qubits: 3,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("qubit index 5"));
        assert!(msg.contains("num_qubits = 3"));
    }

    #[test]
    fn test_validation_error_display() {
        let ve = ValidationError {
            instruction_index: 2,
            message: "bad gate".to_string(),
            severity: Severity::Warning,
        };
        let msg = format!("{}", ve);
        assert!(msg.contains("[warning]"));
        assert!(msg.contains("instruction 2"));
    }

    // ---- Qubit inference ----

    #[test]
    fn test_qubit_count_inferred_from_instructions() {
        let src = r#"
call void @__quantum__qis__h__body(%Qubit* inttoptr (i64 0 to %Qubit*))
call void @__quantum__qis__cnot__body(%Qubit* inttoptr (i64 0 to %Qubit*), %Qubit* inttoptr (i64 4 to %Qubit*))
"#;
        let parser = QIRParser::new();
        let prog = parser.parse(src).unwrap();
        assert_eq!(prog.num_qubits, 5); // 0..4 inclusive
    }

    // ---- Classical bit bounds validation ----

    #[test]
    fn test_classical_bit_bounds_validation() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::Measure(0, 5)); // cbit 5 out of range

        let errors = QIRValidator::check_qubit_bounds(&prog);
        assert!(
            errors.iter().any(|e| e.message.contains("classical bit 5")),
            "expected classical bit bounds error, got: {:?}",
            errors
        );
    }

    // ---- Optimizer does not merge different qubits ----

    #[test]
    fn test_optimizer_does_not_merge_different_qubit_rotations() {
        let mut prog = QIRProgram::new(2, 0);
        prog.instructions.push(QIRInstruction::Rz(0, 0.5));
        prog.instructions.push(QIRInstruction::Rz(1, 0.5));

        let merged = QIROptimizer::merge_rotations(&mut prog);
        assert_eq!(merged, 0);
        assert_eq!(prog.instructions.len(), 2);
    }

    // ---- NQPUCircuit basic ----

    #[test]
    fn test_nqpu_circuit_new() {
        let c = NQPUCircuit::new(4, 2);
        assert_eq!(c.num_qubits, 4);
        assert_eq!(c.num_cbits, 2);
        assert!(c.gates.is_empty());
        assert!(c.measurements.is_empty());
    }

    // ---- Base profile accepts end-of-circuit measurement ----

    #[test]
    fn test_base_profile_accepts_terminal_measurements() {
        let mut prog = QIRProgram::new(2, 2);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::CNOT(0, 1));
        prog.instructions.push(QIRInstruction::Measure(0, 0));
        prog.instructions.push(QIRInstruction::Measure(1, 1));

        let errors = QIRValidator::validate(&prog, &TargetProfile::BaseProfile);
        assert!(
            errors.is_empty(),
            "terminal measurements should be allowed in BaseProfile, got: {:?}",
            errors
        );
    }

    // --------------------------------------------------------
    // Binary QIR encoding/decoding
    // --------------------------------------------------------

    #[test]
    fn test_binary_roundtrip_simple() {
        let mut prog = QIRProgram::new(3, 1);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::CNOT(0, 1));
        prog.instructions.push(QIRInstruction::Measure(0, 0));

        let emitter = QIREmitter::new();
        let binary = emitter.emit_binary(&prog);

        // Check magic and version
        assert_eq!(&binary[0..4], b"nQIR");
        assert_eq!(binary[4], 1);

        // Roundtrip decode
        let decoded = QIREmitter::decode_binary(&binary).unwrap();
        assert_eq!(decoded.num_qubits, 3);
        assert_eq!(decoded.num_classical_bits, 1);
        assert_eq!(decoded.instructions.len(), 3);
        assert_eq!(decoded.instructions[0], QIRInstruction::H(0));
        assert_eq!(decoded.instructions[1], QIRInstruction::CNOT(0, 1));
        assert_eq!(decoded.instructions[2], QIRInstruction::Measure(0, 0));
    }

    #[test]
    fn test_binary_roundtrip_rotations() {
        let mut prog = QIRProgram::new(2, 0);
        prog.instructions.push(QIRInstruction::Rx(0, 1.5707963267948966));
        prog.instructions.push(QIRInstruction::Ry(1, 0.7853981633974483));
        prog.instructions.push(QIRInstruction::Rz(0, 3.141592653589793));

        let emitter = QIREmitter::new();
        let binary = emitter.emit_binary(&prog);
        let decoded = QIREmitter::decode_binary(&binary).unwrap();

        assert_eq!(decoded.instructions.len(), 3);
        match &decoded.instructions[0] {
            QIRInstruction::Rx(q, angle) => {
                assert_eq!(*q, 0);
                assert!((angle - 1.5707963267948966).abs() < 1e-15);
            }
            _ => panic!("Expected Rx"),
        }
    }

    #[test]
    fn test_binary_roundtrip_all_gates() {
        let mut prog = QIRProgram::new(4, 2);
        prog.instructions.push(QIRInstruction::H(0));
        prog.instructions.push(QIRInstruction::X(1));
        prog.instructions.push(QIRInstruction::Y(2));
        prog.instructions.push(QIRInstruction::Z(3));
        prog.instructions.push(QIRInstruction::S(0));
        prog.instructions.push(QIRInstruction::T(1));
        prog.instructions.push(QIRInstruction::CZ(0, 1));
        prog.instructions.push(QIRInstruction::SWAP(2, 3));
        prog.instructions.push(QIRInstruction::Reset(0));
        prog.instructions.push(QIRInstruction::Barrier(vec![0, 1, 2]));
        prog.instructions.push(QIRInstruction::Measure(0, 0));
        prog.instructions.push(QIRInstruction::Measure(1, 1));

        let emitter = QIREmitter::new();
        let binary = emitter.emit_binary(&prog);
        let decoded = QIREmitter::decode_binary(&binary).unwrap();

        assert_eq!(decoded.instructions.len(), 12);
        assert_eq!(decoded.instructions[6], QIRInstruction::CZ(0, 1));
        assert_eq!(decoded.instructions[7], QIRInstruction::SWAP(2, 3));
        assert_eq!(decoded.instructions[8], QIRInstruction::Reset(0));
        assert_eq!(
            decoded.instructions[9],
            QIRInstruction::Barrier(vec![0, 1, 2])
        );
    }

    #[test]
    fn test_binary_compact_size() {
        // Binary should be more compact than text
        let mut prog = QIRProgram::new(10, 10);
        for i in 0..10 {
            prog.instructions.push(QIRInstruction::H(i));
            prog.instructions.push(QIRInstruction::CNOT(i, (i + 1) % 10));
            prog.instructions.push(QIRInstruction::Measure(i, i));
        }

        let emitter = QIREmitter::new();
        let text = emitter.emit(&prog);
        let binary = emitter.emit_binary(&prog);

        assert!(
            binary.len() < text.len(),
            "Binary ({} bytes) should be smaller than text ({} bytes)",
            binary.len(),
            text.len()
        );
    }

    #[test]
    fn test_binary_invalid_magic() {
        let data = vec![0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = QIREmitter::decode_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_too_short() {
        let data = vec![b'n', b'Q', b'I', b'R'];
        let result = QIREmitter::decode_binary(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_emit_with_config_text() {
        let mut prog = QIRProgram::new(1, 0);
        prog.instructions.push(QIRInstruction::H(0));
        let emitter = QIREmitter::new();
        let config = QIRConfig::new();
        let output = emitter.emit_with_config(&prog, &config);
        let text = String::from_utf8(output).unwrap();
        assert!(text.contains("QIR Version"));
    }

    #[test]
    fn test_emit_with_config_binary() {
        let mut prog = QIRProgram::new(1, 0);
        prog.instructions.push(QIRInstruction::H(0));
        let emitter = QIREmitter::new();
        let config = QIRConfig::new().output_format(OutputFormat::Binary);
        let output = emitter.emit_with_config(&prog, &config);
        assert_eq!(&output[0..4], b"nQIR");
    }

    #[test]
    fn test_binary_conditional_if() {
        let mut prog = QIRProgram::new(2, 1);
        prog.instructions.push(QIRInstruction::If(
            0,
            vec![QIRInstruction::X(1)],
        ));

        let emitter = QIREmitter::new();
        let binary = emitter.emit_binary(&prog);
        let decoded = QIREmitter::decode_binary(&binary).unwrap();

        assert_eq!(decoded.instructions.len(), 1);
        match &decoded.instructions[0] {
            QIRInstruction::If(cbit, body) => {
                assert_eq!(*cbit, 0);
                assert_eq!(body.len(), 1);
                assert_eq!(body[0], QIRInstruction::X(1));
            }
            _ => panic!("Expected If instruction"),
        }
    }
}
