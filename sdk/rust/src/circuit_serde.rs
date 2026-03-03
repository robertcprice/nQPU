//! Circuit Serialization/Deserialization for nQPU-Metal
//!
//! Provides JSON, compact JSON, binary, and OpenQASM 3.0 serialization
//! for quantum circuits, enabling save/load, sharing, and interoperability.
//!
//! # Supported formats
//!
//! - **JSON**: Human-readable with optional pretty-printing
//! - **Compact JSON**: Minified for smaller file sizes
//! - **Binary**: Custom compact encoding for maximum efficiency
//! - **OpenQASM 3.0**: Standard quantum assembly export
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::circuit_serde::*;
//!
//! let mut circuit = SerializableCircuit::new(3);
//! circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
//! circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
//!
//! let config = SerdeConfig::default();
//! let json = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
//! let restored = deserialize_circuit(&json, CircuitFormat::Json).unwrap();
//! assert_eq!(circuit.gates.len(), restored.gates.len());
//! ```

use std::collections::HashMap;
use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during circuit serialization/deserialization.
#[derive(Debug)]
pub enum CircuitSerdeError {
    /// Serialization failed with a descriptive message.
    SerializationFailed(String),
    /// Deserialization failed with a descriptive message.
    DeserializationFailed(String),
    /// The input data is not in a recognized format.
    InvalidFormat(String),
    /// The format version in the data does not match what this library supports.
    VersionMismatch { expected: u32, found: u32 },
}

impl fmt::Display for CircuitSerdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitSerdeError::SerializationFailed(msg) => {
                write!(f, "serialization failed: {}", msg)
            }
            CircuitSerdeError::DeserializationFailed(msg) => {
                write!(f, "deserialization failed: {}", msg)
            }
            CircuitSerdeError::InvalidFormat(msg) => {
                write!(f, "invalid format: {}", msg)
            }
            CircuitSerdeError::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "version mismatch: expected {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for CircuitSerdeError {}

/// Result type alias for circuit serde operations.
pub type Result<T> = std::result::Result<T, CircuitSerdeError>;

// ============================================================
// FORMAT VERSION
// ============================================================

/// Current binary/JSON format version for backward compatibility.
const FORMAT_VERSION: u32 = 1;

/// Magic bytes identifying the nQPU-Metal binary circuit format.
const BINARY_MAGIC: &[u8; 4] = b"NQCR";

// ============================================================
// CORE DATA TYPES
// ============================================================

/// Output format for circuit serialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitFormat {
    /// Pretty-printed JSON (human-readable).
    Json,
    /// Minified JSON (smaller file size).
    CompactJson,
    /// Custom compact binary encoding.
    Binary,
    /// OpenQASM 3.0 text format.
    OpenQASM3,
}

/// Configuration for serialization behavior.
#[derive(Clone, Debug)]
pub struct SerdeConfig {
    /// Whether to pretty-print JSON output.
    pub pretty_print: bool,
    /// Whether to include metadata in the output.
    pub include_metadata: bool,
    /// Whether to compress binary output (reserved for future use).
    pub compress: bool,
}

impl Default for SerdeConfig {
    fn default() -> Self {
        Self {
            pretty_print: true,
            include_metadata: true,
            compress: false,
        }
    }
}

/// Metadata describing a quantum circuit.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CircuitMetadata {
    pub name: String,
    pub description: String,
    pub author: String,
    pub creation_date: String,
    pub qubit_count: usize,
    pub gate_count: usize,
    pub depth: usize,
    pub tags: Vec<String>,
}

impl Default for CircuitMetadata {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            author: String::new(),
            creation_date: String::new(),
            qubit_count: 0,
            gate_count: 0,
            depth: 0,
            tags: Vec::new(),
        }
    }
}

/// A single gate in its serializable representation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SerializableGate {
    /// Gate type name (e.g., "H", "CNOT", "Rz").
    pub gate_type: String,
    /// Target qubit indices.
    pub targets: Vec<usize>,
    /// Control qubit indices (empty for non-controlled gates).
    pub controls: Vec<usize>,
    /// Gate parameters as f64 values (empty for non-parametric gates).
    pub params: Vec<f64>,
}

impl SerializableGate {
    /// Create a new serializable gate.
    pub fn new(
        gate_type: &str,
        targets: Vec<usize>,
        controls: Vec<usize>,
        params: Vec<f64>,
    ) -> Self {
        Self {
            gate_type: gate_type.to_string(),
            targets,
            controls,
            params,
        }
    }

    /// Map a gate type string to a compact numeric ID for binary encoding.
    fn gate_type_to_id(gate_type: &str) -> u8 {
        match gate_type {
            "H" => 0,
            "X" => 1,
            "Y" => 2,
            "Z" => 3,
            "S" => 4,
            "T" => 5,
            "Rx" => 6,
            "Ry" => 7,
            "Rz" => 8,
            "CNOT" => 9,
            "CZ" => 10,
            "SWAP" => 11,
            "Toffoli" => 12,
            "CRx" => 13,
            "CRy" => 14,
            "CRz" => 15,
            "CR" => 16,
            "SX" => 17,
            "Phase" => 18,
            "ISWAP" => 19,
            "CCZ" => 20,
            "U" => 21,
            _ => 255, // Custom / unknown
        }
    }

    /// Map a compact numeric ID back to a gate type string.
    fn id_to_gate_type(id: u8) -> &'static str {
        match id {
            0 => "H",
            1 => "X",
            2 => "Y",
            3 => "Z",
            4 => "S",
            5 => "T",
            6 => "Rx",
            7 => "Ry",
            8 => "Rz",
            9 => "CNOT",
            10 => "CZ",
            11 => "SWAP",
            12 => "Toffoli",
            13 => "CRx",
            14 => "CRy",
            15 => "CRz",
            16 => "CR",
            17 => "SX",
            18 => "Phase",
            19 => "ISWAP",
            20 => "CCZ",
            21 => "U",
            _ => "Custom",
        }
    }

    /// Convert this gate to the crate's `GateType` enum, if possible.
    pub fn to_gate_type(&self) -> Option<crate::gates::GateType> {
        match self.gate_type.as_str() {
            "H" => Some(crate::gates::GateType::H),
            "X" => Some(crate::gates::GateType::X),
            "Y" => Some(crate::gates::GateType::Y),
            "Z" => Some(crate::gates::GateType::Z),
            "S" => Some(crate::gates::GateType::S),
            "T" => Some(crate::gates::GateType::T),
            "Rx" => self.params.first().map(|&p| crate::gates::GateType::Rx(p)),
            "Ry" => self.params.first().map(|&p| crate::gates::GateType::Ry(p)),
            "Rz" => self.params.first().map(|&p| crate::gates::GateType::Rz(p)),
            "CNOT" => Some(crate::gates::GateType::CNOT),
            "CZ" => Some(crate::gates::GateType::CZ),
            "SWAP" => Some(crate::gates::GateType::SWAP),
            "Toffoli" => Some(crate::gates::GateType::Toffoli),
            "CRx" => self.params.first().map(|&p| crate::gates::GateType::CRx(p)),
            "CRy" => self.params.first().map(|&p| crate::gates::GateType::CRy(p)),
            "CRz" => self.params.first().map(|&p| crate::gates::GateType::CRz(p)),
            "CR" => self.params.first().map(|&p| crate::gates::GateType::CR(p)),
            "SX" => Some(crate::gates::GateType::SX),
            "Phase" => self.params.first().map(|&p| crate::gates::GateType::Phase(p)),
            "ISWAP" => Some(crate::gates::GateType::ISWAP),
            "CCZ" => Some(crate::gates::GateType::CCZ),
            "U" => {
                if self.params.len() >= 3 {
                    Some(crate::gates::GateType::U {
                        theta: self.params[0],
                        phi: self.params[1],
                        lambda: self.params[2],
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Create a `SerializableGate` from the crate's `GateType`, target, and control qubits.
    pub fn from_gate_type(
        gate: &crate::gates::GateType,
        targets: Vec<usize>,
        controls: Vec<usize>,
    ) -> Self {
        let (name, params) = match gate {
            crate::gates::GateType::H => ("H", vec![]),
            crate::gates::GateType::X => ("X", vec![]),
            crate::gates::GateType::Y => ("Y", vec![]),
            crate::gates::GateType::Z => ("Z", vec![]),
            crate::gates::GateType::S => ("S", vec![]),
            crate::gates::GateType::T => ("T", vec![]),
            crate::gates::GateType::Rx(p) => ("Rx", vec![*p]),
            crate::gates::GateType::Ry(p) => ("Ry", vec![*p]),
            crate::gates::GateType::Rz(p) => ("Rz", vec![*p]),
            crate::gates::GateType::CNOT => ("CNOT", vec![]),
            crate::gates::GateType::CZ => ("CZ", vec![]),
            crate::gates::GateType::SWAP => ("SWAP", vec![]),
            crate::gates::GateType::Toffoli => ("Toffoli", vec![]),
            crate::gates::GateType::CRx(p) => ("CRx", vec![*p]),
            crate::gates::GateType::CRy(p) => ("CRy", vec![*p]),
            crate::gates::GateType::CRz(p) => ("CRz", vec![*p]),
            crate::gates::GateType::CR(p) => ("CR", vec![*p]),
            crate::gates::GateType::SX => ("SX", vec![]),
            crate::gates::GateType::Phase(p) => ("Phase", vec![*p]),
            crate::gates::GateType::ISWAP => ("ISWAP", vec![]),
            crate::gates::GateType::CCZ => ("CCZ", vec![]),
            crate::gates::GateType::U {
                theta,
                phi,
                lambda,
            } => ("U", vec![*theta, *phi, *lambda]),
            crate::gates::GateType::Rxx(p) => ("Rxx", vec![*p]),
            crate::gates::GateType::Ryy(p) => ("Ryy", vec![*p]),
            crate::gates::GateType::Rzz(p) => ("Rzz", vec![*p]),
            crate::gates::GateType::CSWAP => ("CSWAP", vec![]),
            crate::gates::GateType::CU { theta, phi, lambda, gamma } => ("CU", vec![*theta, *phi, *lambda, *gamma]),
            crate::gates::GateType::Custom(_) => ("Custom", vec![]),
        };
        Self::new(name, targets, controls, params)
    }
}

/// A complete quantum circuit in its serializable representation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SerializableCircuit {
    /// Format version for backward compatibility.
    pub version: u32,
    /// Number of qubits in the circuit.
    pub num_qubits: usize,
    /// Ordered list of gates.
    pub gates: Vec<SerializableGate>,
    /// Optional circuit metadata.
    pub metadata: Option<CircuitMetadata>,
}

impl SerializableCircuit {
    /// Create a new empty circuit with the given qubit count.
    pub fn new(num_qubits: usize) -> Self {
        Self {
            version: FORMAT_VERSION,
            num_qubits,
            gates: Vec::new(),
            metadata: None,
        }
    }

    /// Create a new circuit with metadata.
    pub fn with_metadata(num_qubits: usize, metadata: CircuitMetadata) -> Self {
        Self {
            version: FORMAT_VERSION,
            num_qubits,
            gates: Vec::new(),
            metadata: Some(metadata),
        }
    }

    /// Add a gate to the circuit.
    pub fn add_gate(&mut self, gate: SerializableGate) {
        self.gates.push(gate);
    }

    /// Compute a simple circuit depth estimate.
    ///
    /// This assigns each gate to the earliest time-step where all its
    /// target and control qubits are free, then returns the total number
    /// of time-steps.
    pub fn compute_depth(&self) -> usize {
        if self.gates.is_empty() {
            return 0;
        }
        let mut qubit_depth = vec![0usize; self.num_qubits];
        for gate in &self.gates {
            let max_d = gate
                .targets
                .iter()
                .chain(gate.controls.iter())
                .filter_map(|&q| qubit_depth.get(q))
                .copied()
                .max()
                .unwrap_or(0);
            let next = max_d + 1;
            for &q in gate.targets.iter().chain(gate.controls.iter()) {
                if q < self.num_qubits {
                    qubit_depth[q] = next;
                }
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }

    /// Build metadata from the current circuit state.
    pub fn build_metadata(&mut self) {
        let depth = self.compute_depth();
        let gate_count = self.gates.len();
        let existing = self.metadata.take().unwrap_or_default();
        self.metadata = Some(CircuitMetadata {
            qubit_count: self.num_qubits,
            gate_count,
            depth,
            ..existing
        });
    }
}

// ============================================================
// JSON SERIALIZATION (manual, no serde required)
// ============================================================

/// Escape a string for JSON output.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Serialize a circuit to a JSON string (manual implementation).
fn circuit_to_json(circuit: &SerializableCircuit, pretty: bool) -> Result<String> {
    let mut out = String::new();
    let (nl, indent, colon) = if pretty {
        ("\n", "  ", ": ")
    } else {
        ("", "", ":")
    };

    out.push('{');
    out.push_str(nl);

    // version
    out.push_str(&format!("{}\"version\"{}{}", indent, colon, circuit.version));
    out.push(',');
    out.push_str(nl);

    // num_qubits
    out.push_str(&format!(
        "{}\"num_qubits\"{}{}",
        indent, colon, circuit.num_qubits
    ));
    out.push(',');
    out.push_str(nl);

    // gates
    out.push_str(&format!("{}\"gates\"{}[", indent, colon));
    if !circuit.gates.is_empty() {
        out.push_str(nl);
    }
    for (i, gate) in circuit.gates.iter().enumerate() {
        let gi = if pretty { "    " } else { "" };
        out.push_str(&format!("{}{}", gi, gate_to_json(gate, pretty)?));
        if i + 1 < circuit.gates.len() {
            out.push(',');
        }
        out.push_str(nl);
    }
    if !circuit.gates.is_empty() {
        out.push_str(indent);
    }
    out.push(']');

    // metadata
    if let Some(ref meta) = circuit.metadata {
        out.push(',');
        out.push_str(nl);
        out.push_str(&format!(
            "{}\"metadata\"{}{}",
            indent,
            colon,
            metadata_to_json(meta, pretty)?
        ));
    }

    out.push_str(nl);
    out.push('}');
    Ok(out)
}

/// Serialize a single gate to a JSON object string.
fn gate_to_json(gate: &SerializableGate, pretty: bool) -> Result<String> {
    let colon = if pretty { ": " } else { ":" };
    let targets_str: Vec<String> = gate.targets.iter().map(|t| t.to_string()).collect();
    let controls_str: Vec<String> = gate.controls.iter().map(|c| c.to_string()).collect();
    let params_str: Vec<String> = gate.params.iter().map(|p| format_f64(*p)).collect();

    Ok(format!(
        "{{\"gate_type\"{}\"{}\"{}\"targets\"{}[{}]{}\"controls\"{}[{}]{}\"params\"{}[{}]}}",
        colon,
        json_escape(&gate.gate_type),
        if pretty { ", " } else { "," },
        colon,
        targets_str.join(","),
        if pretty { ", " } else { "," },
        colon,
        controls_str.join(","),
        if pretty { ", " } else { "," },
        colon,
        params_str.join(","),
    ))
}

/// Format an f64 value for JSON, ensuring no unnecessary precision loss.
fn format_f64(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        // Integer-valued float: emit as e.g. "1.0"
        format!("{:.1}", v)
    } else {
        // Full precision
        format!("{}", v)
    }
}

/// Serialize metadata to a JSON object string.
fn metadata_to_json(meta: &CircuitMetadata, pretty: bool) -> Result<String> {
    let colon = if pretty { ": " } else { ":" };
    let sep = if pretty { ", " } else { "," };
    let tags: Vec<String> = meta.tags.iter().map(|t| format!("\"{}\"", json_escape(t))).collect();
    Ok(format!(
        "{{\"name\"{}\"{}\"{}\"description\"{}\"{}\"{}\"author\"{}\"{}\"{}\"creation_date\"{}\"{}\"{}\"qubit_count\"{}{}{}\"gate_count\"{}{}{}\"depth\"{}{}{}\"tags\"{}[{}]}}",
        colon, json_escape(&meta.name),
        sep,
        colon, json_escape(&meta.description),
        sep,
        colon, json_escape(&meta.author),
        sep,
        colon, json_escape(&meta.creation_date),
        sep,
        colon, meta.qubit_count,
        sep,
        colon, meta.gate_count,
        sep,
        colon, meta.depth,
        sep,
        colon, tags.join(","),
    ))
}

// ============================================================
// JSON DESERIALIZATION (manual, no serde required)
// ============================================================

/// A minimal JSON value type for manual parsing.
#[derive(Debug, Clone)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_u64(&self) -> Option<u64> {
        match self {
            JsonValue::Number(n) => Some(*n as u64),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a),
            _ => None,
        }
    }

    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()
            .and_then(|pairs| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

/// A simple recursive-descent JSON parser.
struct JsonParser<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len()
            && matches!(self.input[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let c = self.input.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn expect(&mut self, ch: u8) -> Result<()> {
        self.skip_whitespace();
        match self.advance() {
            Some(c) if c == ch => Ok(()),
            Some(c) => Err(CircuitSerdeError::DeserializationFailed(format!(
                "expected '{}', found '{}' at position {}",
                ch as char, c as char, self.pos
            ))),
            None => Err(CircuitSerdeError::DeserializationFailed(format!(
                "expected '{}', found end of input",
                ch as char
            ))),
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => self.parse_string().map(JsonValue::Str),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            Some(c) => Err(CircuitSerdeError::DeserializationFailed(format!(
                "unexpected character '{}' at position {}",
                c as char, self.pos
            ))),
            None => Err(CircuitSerdeError::DeserializationFailed(
                "unexpected end of input".to_string(),
            )),
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                Some(b'"') => return Ok(s),
                Some(b'\\') => match self.advance() {
                    Some(b'"') => s.push('"'),
                    Some(b'\\') => s.push('\\'),
                    Some(b'/') => s.push('/'),
                    Some(b'n') => s.push('\n'),
                    Some(b'r') => s.push('\r'),
                    Some(b't') => s.push('\t'),
                    Some(b'u') => {
                        let mut hex = String::with_capacity(4);
                        for _ in 0..4 {
                            match self.advance() {
                                Some(c) => hex.push(c as char),
                                None => {
                                    return Err(CircuitSerdeError::DeserializationFailed(
                                        "incomplete unicode escape".to_string(),
                                    ))
                                }
                            }
                        }
                        let code = u32::from_str_radix(&hex, 16).map_err(|_| {
                            CircuitSerdeError::DeserializationFailed(format!(
                                "invalid unicode escape: \\u{}",
                                hex
                            ))
                        })?;
                        if let Some(ch) = char::from_u32(code) {
                            s.push(ch);
                        }
                    }
                    _ => {
                        return Err(CircuitSerdeError::DeserializationFailed(
                            "invalid escape sequence".to_string(),
                        ))
                    }
                },
                Some(c) => s.push(c as char),
                None => {
                    return Err(CircuitSerdeError::DeserializationFailed(
                        "unterminated string".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.input.len()
            && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.input.len()
                && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
            {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let slice = &self.input[start..self.pos];
        let text = std::str::from_utf8(slice).map_err(|_| {
            CircuitSerdeError::DeserializationFailed("invalid UTF-8 in number".to_string())
        })?;
        let val: f64 = text.parse().map_err(|_| {
            CircuitSerdeError::DeserializationFailed(format!("invalid number: {}", text))
        })?;
        Ok(JsonValue::Number(val))
    }

    fn parse_bool(&mut self) -> Result<JsonValue> {
        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(CircuitSerdeError::DeserializationFailed(format!(
                "invalid boolean at position {}",
                self.pos
            )))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue> {
        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(CircuitSerdeError::DeserializationFailed(format!(
                "invalid null at position {}",
                self.pos
            )))
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.expect(b'[')?;
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            arr.push(self.parse_value()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(arr));
                }
                _ => {
                    return Err(CircuitSerdeError::DeserializationFailed(
                        "expected ',' or ']' in array".to_string(),
                    ))
                }
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            self.skip_whitespace();
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            let val = self.parse_value()?;
            pairs.push((key, val));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(pairs));
                }
                _ => {
                    return Err(CircuitSerdeError::DeserializationFailed(
                        "expected ',' or '}' in object".to_string(),
                    ))
                }
            }
        }
    }
}

/// Parse a JSON string into a `JsonValue`.
fn parse_json(input: &str) -> Result<JsonValue> {
    let mut parser = JsonParser::new(input);
    let val = parser.parse_value()?;
    Ok(val)
}

/// Reconstruct a `SerializableCircuit` from a parsed `JsonValue`.
fn circuit_from_json_value(val: &JsonValue) -> Result<SerializableCircuit> {
    let version = val
        .get("version")
        .and_then(|v| v.as_u64())
        .unwrap_or(FORMAT_VERSION as u64) as u32;

    if version > FORMAT_VERSION {
        return Err(CircuitSerdeError::VersionMismatch {
            expected: FORMAT_VERSION,
            found: version,
        });
    }

    let num_qubits = val
        .get("num_qubits")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            CircuitSerdeError::DeserializationFailed("missing 'num_qubits' field".to_string())
        })? as usize;

    let gates_val = val.get("gates").ok_or_else(|| {
        CircuitSerdeError::DeserializationFailed("missing 'gates' field".to_string())
    })?;
    let gates_arr = gates_val.as_array().ok_or_else(|| {
        CircuitSerdeError::DeserializationFailed("'gates' is not an array".to_string())
    })?;

    let mut gates = Vec::with_capacity(gates_arr.len());
    for g in gates_arr {
        gates.push(gate_from_json_value(g)?);
    }

    let metadata = val.get("metadata").and_then(|m| metadata_from_json_value(m).ok());

    Ok(SerializableCircuit {
        version,
        num_qubits,
        gates,
        metadata,
    })
}

/// Reconstruct a `SerializableGate` from a parsed `JsonValue`.
fn gate_from_json_value(val: &JsonValue) -> Result<SerializableGate> {
    let gate_type = val
        .get("gate_type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CircuitSerdeError::DeserializationFailed("missing 'gate_type' in gate".to_string())
        })?
        .to_string();

    let targets = extract_usize_array(val, "targets")?;
    let controls = extract_usize_array(val, "controls")?;
    let params = extract_f64_array(val, "params")?;

    Ok(SerializableGate {
        gate_type,
        targets,
        controls,
        params,
    })
}

/// Extract a `Vec<usize>` from a JSON object field.
fn extract_usize_array(val: &JsonValue, key: &str) -> Result<Vec<usize>> {
    let arr = val
        .get(key)
        .and_then(|v| v.as_array())
        .unwrap_or(&[]);
    arr.iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .ok_or_else(|| {
                    CircuitSerdeError::DeserializationFailed(format!(
                        "non-integer in '{}'",
                        key
                    ))
                })
        })
        .collect()
}

/// Extract a `Vec<f64>` from a JSON object field.
fn extract_f64_array(val: &JsonValue, key: &str) -> Result<Vec<f64>> {
    let arr = val
        .get(key)
        .and_then(|v| v.as_array())
        .unwrap_or(&[]);
    arr.iter()
        .map(|v| {
            v.as_f64().ok_or_else(|| {
                CircuitSerdeError::DeserializationFailed(format!(
                    "non-number in '{}'",
                    key
                ))
            })
        })
        .collect()
}

/// Reconstruct `CircuitMetadata` from a parsed `JsonValue`.
fn metadata_from_json_value(val: &JsonValue) -> Result<CircuitMetadata> {
    Ok(CircuitMetadata {
        name: val
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        description: val
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        author: val
            .get("author")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        creation_date: val
            .get("creation_date")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        qubit_count: val
            .get("qubit_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        gate_count: val
            .get("gate_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        depth: val
            .get("depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        tags: val
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

// ============================================================
// BINARY SERIALIZATION
// ============================================================

/// Encode a circuit into a compact binary format.
///
/// Binary layout:
/// ```text
/// [4 bytes]  magic "NQCR"
/// [4 bytes]  version (u32 LE)
/// [4 bytes]  num_qubits (u32 LE)
/// [4 bytes]  gate_count (u32 LE)
/// For each gate:
///   [1 byte]  gate_type_id
///   [1 byte]  target_count
///   [1 byte]  control_count
///   [1 byte]  param_count
///   [target_count * 2 bytes]  target qubit indices (u16 LE)
///   [control_count * 2 bytes] control qubit indices (u16 LE)
///   [param_count * 8 bytes]   parameters (f64 LE)
///   If gate_type_id == 255 (Custom):
///     [2 bytes]  gate_name_len (u16 LE)
///     [gate_name_len bytes]  gate_name (UTF-8)
/// ```
fn circuit_to_binary(circuit: &SerializableCircuit) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(256);

    // Header
    buf.extend_from_slice(BINARY_MAGIC);
    buf.extend_from_slice(&circuit.version.to_le_bytes());
    buf.extend_from_slice(&(circuit.num_qubits as u32).to_le_bytes());
    buf.extend_from_slice(&(circuit.gates.len() as u32).to_le_bytes());

    // Gates
    for gate in &circuit.gates {
        let id = SerializableGate::gate_type_to_id(&gate.gate_type);
        buf.push(id);
        buf.push(gate.targets.len() as u8);
        buf.push(gate.controls.len() as u8);
        buf.push(gate.params.len() as u8);

        for &t in &gate.targets {
            buf.extend_from_slice(&(t as u16).to_le_bytes());
        }
        for &c in &gate.controls {
            buf.extend_from_slice(&(c as u16).to_le_bytes());
        }
        for &p in &gate.params {
            buf.extend_from_slice(&p.to_le_bytes());
        }

        // For custom gates, encode the name
        if id == 255 {
            let name_bytes = gate.gate_type.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
        }
    }

    Ok(buf)
}

/// Decode a circuit from the compact binary format.
fn circuit_from_binary(data: &[u8]) -> Result<SerializableCircuit> {
    if data.len() < 16 {
        return Err(CircuitSerdeError::DeserializationFailed(
            "binary data too short for header".to_string(),
        ));
    }

    // Validate magic
    if &data[0..4] != BINARY_MAGIC {
        return Err(CircuitSerdeError::InvalidFormat(
            "missing NQCR magic bytes".to_string(),
        ));
    }

    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if version > FORMAT_VERSION {
        return Err(CircuitSerdeError::VersionMismatch {
            expected: FORMAT_VERSION,
            found: version,
        });
    }

    let num_qubits =
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let gate_count =
        u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

    let mut pos = 16;
    let mut gates = Vec::with_capacity(gate_count);

    for _ in 0..gate_count {
        if pos + 4 > data.len() {
            return Err(CircuitSerdeError::DeserializationFailed(
                "unexpected end of binary data in gate header".to_string(),
            ));
        }

        let gate_id = data[pos];
        let target_count = data[pos + 1] as usize;
        let control_count = data[pos + 2] as usize;
        let param_count = data[pos + 3] as usize;
        pos += 4;

        let needed =
            target_count * 2 + control_count * 2 + param_count * 8;
        if pos + needed > data.len() {
            return Err(CircuitSerdeError::DeserializationFailed(
                "unexpected end of binary data in gate body".to_string(),
            ));
        }

        let mut targets = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            targets.push(u16::from_le_bytes([data[pos], data[pos + 1]]) as usize);
            pos += 2;
        }

        let mut controls = Vec::with_capacity(control_count);
        for _ in 0..control_count {
            controls.push(u16::from_le_bytes([data[pos], data[pos + 1]]) as usize);
            pos += 2;
        }

        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            let bytes: [u8; 8] = data[pos..pos + 8]
                .try_into()
                .map_err(|_| {
                    CircuitSerdeError::DeserializationFailed(
                        "failed to read f64 parameter".to_string(),
                    )
                })?;
            params.push(f64::from_le_bytes(bytes));
            pos += 8;
        }

        let gate_type = if gate_id == 255 {
            // Custom gate: read name
            if pos + 2 > data.len() {
                return Err(CircuitSerdeError::DeserializationFailed(
                    "unexpected end of binary data reading custom gate name length".to_string(),
                ));
            }
            let name_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
            pos += 2;
            if pos + name_len > data.len() {
                return Err(CircuitSerdeError::DeserializationFailed(
                    "unexpected end of binary data reading custom gate name".to_string(),
                ));
            }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|_| {
                    CircuitSerdeError::DeserializationFailed(
                        "invalid UTF-8 in custom gate name".to_string(),
                    )
                })?
                .to_string();
            pos += name_len;
            name
        } else {
            SerializableGate::id_to_gate_type(gate_id).to_string()
        };

        gates.push(SerializableGate {
            gate_type,
            targets,
            controls,
            params,
        });
    }

    Ok(SerializableCircuit {
        version,
        num_qubits,
        gates,
        metadata: None,
    })
}

// ============================================================
// OPENQASM 3.0 EXPORT
// ============================================================

/// Export a circuit as an OpenQASM 3.0 string.
fn circuit_to_qasm3(circuit: &SerializableCircuit) -> Result<String> {
    let mut out = String::new();

    out.push_str("OPENQASM 3.0;\n");
    out.push_str("include \"stdgates.inc\";\n\n");

    // Qubit declaration
    out.push_str(&format!("qubit[{}] q;\n\n", circuit.num_qubits));

    // Gates
    for gate in &circuit.gates {
        match gate.gate_type.as_str() {
            "H" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("h q[{}];\n", t));
                }
            }
            "X" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("x q[{}];\n", t));
                }
            }
            "Y" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("y q[{}];\n", t));
                }
            }
            "Z" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("z q[{}];\n", t));
                }
            }
            "S" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("s q[{}];\n", t));
                }
            }
            "T" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("t q[{}];\n", t));
                }
            }
            "SX" => {
                if let Some(&t) = gate.targets.first() {
                    out.push_str(&format!("sx q[{}];\n", t));
                }
            }
            "Rx" => {
                if let (Some(&t), Some(&p)) =
                    (gate.targets.first(), gate.params.first())
                {
                    out.push_str(&format!("rx({}) q[{}];\n", p, t));
                }
            }
            "Ry" => {
                if let (Some(&t), Some(&p)) =
                    (gate.targets.first(), gate.params.first())
                {
                    out.push_str(&format!("ry({}) q[{}];\n", p, t));
                }
            }
            "Rz" => {
                if let (Some(&t), Some(&p)) =
                    (gate.targets.first(), gate.params.first())
                {
                    out.push_str(&format!("rz({}) q[{}];\n", p, t));
                }
            }
            "Phase" => {
                if let (Some(&t), Some(&p)) =
                    (gate.targets.first(), gate.params.first())
                {
                    out.push_str(&format!("p({}) q[{}];\n", p, t));
                }
            }
            "U" => {
                if let Some(&t) = gate.targets.first() {
                    if gate.params.len() >= 3 {
                        out.push_str(&format!(
                            "U({}, {}, {}) q[{}];\n",
                            gate.params[0], gate.params[1], gate.params[2], t
                        ));
                    }
                }
            }
            "CNOT" => {
                if let (Some(&c), Some(&t)) =
                    (gate.controls.first(), gate.targets.first())
                {
                    out.push_str(&format!("cx q[{}], q[{}];\n", c, t));
                }
            }
            "CZ" => {
                if let (Some(&c), Some(&t)) =
                    (gate.controls.first(), gate.targets.first())
                {
                    out.push_str(&format!("cz q[{}], q[{}];\n", c, t));
                }
            }
            "SWAP" => {
                if gate.targets.len() >= 2 {
                    out.push_str(&format!(
                        "swap q[{}], q[{}];\n",
                        gate.targets[0], gate.targets[1]
                    ));
                }
            }
            "ISWAP" => {
                // iSWAP is not in stdgates, emit as a comment + decomposition hint
                if gate.targets.len() >= 2 {
                    out.push_str(&format!(
                        "// iswap q[{}], q[{}]; (decompose into native gates)\n",
                        gate.targets[0], gate.targets[1]
                    ));
                }
            }
            "CRx" => {
                if let (Some(&c), Some(&t), Some(&p)) = (
                    gate.controls.first(),
                    gate.targets.first(),
                    gate.params.first(),
                ) {
                    out.push_str(&format!("crx({}) q[{}], q[{}];\n", p, c, t));
                }
            }
            "CRy" => {
                if let (Some(&c), Some(&t), Some(&p)) = (
                    gate.controls.first(),
                    gate.targets.first(),
                    gate.params.first(),
                ) {
                    out.push_str(&format!("cry({}) q[{}], q[{}];\n", p, c, t));
                }
            }
            "CRz" => {
                if let (Some(&c), Some(&t), Some(&p)) = (
                    gate.controls.first(),
                    gate.targets.first(),
                    gate.params.first(),
                ) {
                    out.push_str(&format!("crz({}) q[{}], q[{}];\n", p, c, t));
                }
            }
            "CR" => {
                if let (Some(&c), Some(&t), Some(&p)) = (
                    gate.controls.first(),
                    gate.targets.first(),
                    gate.params.first(),
                ) {
                    out.push_str(&format!("cp({}) q[{}], q[{}];\n", p, c, t));
                }
            }
            "Toffoli" => {
                // Toffoli: 2 controls + 1 target
                if gate.controls.len() >= 2 {
                    if let Some(&t) = gate.targets.first() {
                        out.push_str(&format!(
                            "ccx q[{}], q[{}], q[{}];\n",
                            gate.controls[0], gate.controls[1], t
                        ));
                    }
                }
            }
            "CCZ" => {
                if gate.controls.len() >= 2 {
                    if let Some(&t) = gate.targets.first() {
                        out.push_str(&format!(
                            "// ccz q[{}], q[{}], q[{}]; (decompose into native gates)\n",
                            gate.controls[0], gate.controls[1], t
                        ));
                    }
                }
            }
            other => {
                out.push_str(&format!("// unsupported gate: {}\n", other));
            }
        }
    }

    Ok(out)
}

// ============================================================
// PUBLIC API
// ============================================================

/// Serialize a circuit to bytes in the specified format.
///
/// For text formats (JSON, CompactJson, OpenQASM3), the bytes are UTF-8 text.
/// For Binary format, the bytes are the raw binary encoding.
pub fn serialize_circuit(
    circuit: &SerializableCircuit,
    format: CircuitFormat,
    config: &SerdeConfig,
) -> Result<Vec<u8>> {
    match format {
        CircuitFormat::Json => {
            let json = circuit_to_json(circuit, config.pretty_print)?;
            Ok(json.into_bytes())
        }
        CircuitFormat::CompactJson => {
            let json = circuit_to_json(circuit, false)?;
            Ok(json.into_bytes())
        }
        CircuitFormat::Binary => circuit_to_binary(circuit),
        CircuitFormat::OpenQASM3 => {
            let qasm = circuit_to_qasm3(circuit)?;
            Ok(qasm.into_bytes())
        }
    }
}

/// Deserialize a circuit from bytes in the specified format.
///
/// For text formats, the bytes must be valid UTF-8.
/// For Binary format, the raw binary encoding is expected.
pub fn deserialize_circuit(
    data: &[u8],
    format: CircuitFormat,
) -> Result<SerializableCircuit> {
    match format {
        CircuitFormat::Json | CircuitFormat::CompactJson => {
            let text = std::str::from_utf8(data).map_err(|e| {
                CircuitSerdeError::DeserializationFailed(format!("invalid UTF-8: {}", e))
            })?;
            let val = parse_json(text)?;
            circuit_from_json_value(&val)
        }
        CircuitFormat::Binary => circuit_from_binary(data),
        CircuitFormat::OpenQASM3 => Err(CircuitSerdeError::InvalidFormat(
            "OpenQASM3 import is not supported; use export only".to_string(),
        )),
    }
}

/// Convert a circuit between formats, returning the serialized bytes.
pub fn convert_format(
    data: &[u8],
    from: CircuitFormat,
    to: CircuitFormat,
    config: &SerdeConfig,
) -> Result<Vec<u8>> {
    let circuit = deserialize_circuit(data, from)?;
    serialize_circuit(&circuit, to, config)
}

// ============================================================
// CIRCUIT LIBRARY
// ============================================================

/// A named collection of circuits with search and filter capabilities.
#[derive(Clone, Debug)]
pub struct CircuitLibrary {
    circuits: HashMap<String, SerializableCircuit>,
}

impl CircuitLibrary {
    /// Create a new empty circuit library.
    pub fn new() -> Self {
        Self {
            circuits: HashMap::new(),
        }
    }

    /// Add a circuit to the library under the given name.
    pub fn add(&mut self, name: &str, circuit: SerializableCircuit) {
        self.circuits.insert(name.to_string(), circuit);
    }

    /// Retrieve a circuit by name.
    pub fn get(&self, name: &str) -> Option<&SerializableCircuit> {
        self.circuits.get(name)
    }

    /// Remove a circuit by name, returning it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<SerializableCircuit> {
        self.circuits.remove(name)
    }

    /// Return the number of circuits in the library.
    pub fn len(&self) -> usize {
        self.circuits.len()
    }

    /// Return true if the library is empty.
    pub fn is_empty(&self) -> bool {
        self.circuits.is_empty()
    }

    /// List all circuit names in the library.
    pub fn names(&self) -> Vec<&str> {
        self.circuits.keys().map(|k| k.as_str()).collect()
    }

    /// Search for circuits whose name contains the query (case-insensitive).
    pub fn search_by_name(&self, query: &str) -> Vec<(&str, &SerializableCircuit)> {
        let query_lower = query.to_lowercase();
        self.circuits
            .iter()
            .filter(|(name, _)| name.to_lowercase().contains(&query_lower))
            .map(|(name, circuit)| (name.as_str(), circuit))
            .collect()
    }

    /// Search for circuits that have a metadata tag matching the query (case-insensitive).
    pub fn search_by_tag(&self, tag: &str) -> Vec<(&str, &SerializableCircuit)> {
        let tag_lower = tag.to_lowercase();
        self.circuits
            .iter()
            .filter(|(_, circuit)| {
                circuit
                    .metadata
                    .as_ref()
                    .map(|m| m.tags.iter().any(|t| t.to_lowercase() == tag_lower))
                    .unwrap_or(false)
            })
            .map(|(name, circuit)| (name.as_str(), circuit))
            .collect()
    }

    /// Filter circuits by minimum qubit count.
    pub fn filter_by_qubit_count(
        &self,
        min_qubits: usize,
    ) -> Vec<(&str, &SerializableCircuit)> {
        self.circuits
            .iter()
            .filter(|(_, circuit)| circuit.num_qubits >= min_qubits)
            .map(|(name, circuit)| (name.as_str(), circuit))
            .collect()
    }

    /// Serialize the entire library to JSON bytes.
    pub fn to_json(&self, config: &SerdeConfig) -> Result<Vec<u8>> {
        let mut out = String::new();
        let (nl, indent, colon) = if config.pretty_print {
            ("\n", "  ", ": ")
        } else {
            ("", "", ":")
        };

        out.push('{');
        out.push_str(nl);

        let names: Vec<&String> = self.circuits.keys().collect();
        for (i, name) in names.iter().enumerate() {
            let circuit = &self.circuits[*name];
            let circuit_json = circuit_to_json(circuit, config.pretty_print)?;
            out.push_str(&format!(
                "{}\"{}\"{}{}",
                indent,
                json_escape(name),
                colon,
                circuit_json
            ));
            if i + 1 < names.len() {
                out.push(',');
            }
            out.push_str(nl);
        }

        out.push('}');
        Ok(out.into_bytes())
    }

    /// Deserialize a library from JSON bytes.
    pub fn from_json(data: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(data).map_err(|e| {
            CircuitSerdeError::DeserializationFailed(format!("invalid UTF-8: {}", e))
        })?;
        let val = parse_json(text)?;
        let pairs = val.as_object().ok_or_else(|| {
            CircuitSerdeError::DeserializationFailed(
                "library JSON root must be an object".to_string(),
            )
        })?;

        let mut lib = CircuitLibrary::new();
        for (name, circuit_val) in pairs {
            let circuit = circuit_from_json_value(circuit_val)?;
            lib.add(name, circuit);
        }
        Ok(lib)
    }
}

impl Default for CircuitLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a simple Bell-state circuit (H + CNOT on 2 qubits).
    fn bell_circuit() -> SerializableCircuit {
        let mut circuit = SerializableCircuit::new(2);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        circuit
    }

    #[test]
    fn serialize_empty_circuit() {
        let circuit = SerializableCircuit::new(3);
        let config = SerdeConfig::default();
        let bytes = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let restored =
            deserialize_circuit(&bytes, CircuitFormat::Json).unwrap();
        assert_eq!(restored.num_qubits, 3);
        assert!(restored.gates.is_empty());
        assert_eq!(restored.version, FORMAT_VERSION);
    }

    #[test]
    fn serialize_single_gate() {
        let mut circuit = SerializableCircuit::new(1);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        let config = SerdeConfig::default();
        let bytes = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let restored =
            deserialize_circuit(&bytes, CircuitFormat::Json).unwrap();
        assert_eq!(restored.gates.len(), 1);
        assert_eq!(restored.gates[0].gate_type, "H");
        assert_eq!(restored.gates[0].targets, vec![0]);
        assert!(restored.gates[0].controls.is_empty());
        assert!(restored.gates[0].params.is_empty());
    }

    #[test]
    fn serialize_multi_gate() {
        let mut circuit = SerializableCircuit::new(3);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        circuit.add_gate(SerializableGate::new(
            "Rz",
            vec![2],
            vec![],
            vec![std::f64::consts::PI / 4.0],
        ));

        let config = SerdeConfig::default();
        let bytes = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let restored =
            deserialize_circuit(&bytes, CircuitFormat::Json).unwrap();

        assert_eq!(restored.num_qubits, 3);
        assert_eq!(restored.gates.len(), 3);
        assert_eq!(restored.gates[0].gate_type, "H");
        assert_eq!(restored.gates[1].gate_type, "CNOT");
        assert_eq!(restored.gates[1].controls, vec![0]);
        assert_eq!(restored.gates[1].targets, vec![1]);
        assert_eq!(restored.gates[2].gate_type, "Rz");
    }

    #[test]
    fn serialize_parametric() {
        let theta = 1.234_567_890_123_456_f64;
        let mut circuit = SerializableCircuit::new(2);
        circuit.add_gate(SerializableGate::new("Rx", vec![0], vec![], vec![theta]));
        circuit.add_gate(SerializableGate::new(
            "U",
            vec![1],
            vec![],
            vec![0.1, 0.2, 0.3],
        ));

        let config = SerdeConfig::default();
        let bytes = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let restored =
            deserialize_circuit(&bytes, CircuitFormat::Json).unwrap();

        assert_eq!(restored.gates[0].params.len(), 1);
        assert!(
            (restored.gates[0].params[0] - theta).abs() < 1e-12,
            "Rx parameter mismatch: {} vs {}",
            restored.gates[0].params[0],
            theta
        );
        assert_eq!(restored.gates[1].params, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn metadata_included() {
        let meta = CircuitMetadata {
            name: "Bell State".to_string(),
            description: "Creates a Bell pair".to_string(),
            author: "nQPU-Metal".to_string(),
            creation_date: "2026-02-14".to_string(),
            qubit_count: 2,
            gate_count: 2,
            depth: 2,
            tags: vec!["entanglement".to_string(), "bell".to_string()],
        };
        let mut circuit = SerializableCircuit::with_metadata(2, meta);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));

        let config = SerdeConfig {
            include_metadata: true,
            ..Default::default()
        };
        let bytes = serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let json_str = std::str::from_utf8(&bytes).unwrap();

        assert!(json_str.contains("\"metadata\""));
        assert!(json_str.contains("Bell State"));
        assert!(json_str.contains("entanglement"));
        assert!(json_str.contains("bell"));
        assert!(json_str.contains("2026-02-14"));

        let restored =
            deserialize_circuit(&bytes, CircuitFormat::Json).unwrap();
        let rm = restored.metadata.unwrap();
        assert_eq!(rm.name, "Bell State");
        assert_eq!(rm.tags, vec!["entanglement", "bell"]);
    }

    #[test]
    fn compact_json() {
        let circuit = bell_circuit();
        let pretty_config = SerdeConfig {
            pretty_print: true,
            ..Default::default()
        };
        let compact_config = SerdeConfig {
            pretty_print: false,
            ..Default::default()
        };

        let pretty_bytes =
            serialize_circuit(&circuit, CircuitFormat::Json, &pretty_config).unwrap();
        let compact_bytes =
            serialize_circuit(&circuit, CircuitFormat::CompactJson, &compact_config)
                .unwrap();

        assert!(
            compact_bytes.len() < pretty_bytes.len(),
            "compact ({} bytes) should be smaller than pretty ({} bytes)",
            compact_bytes.len(),
            pretty_bytes.len()
        );

        // Both should deserialize to the same circuit
        let restored_pretty =
            deserialize_circuit(&pretty_bytes, CircuitFormat::Json).unwrap();
        let restored_compact =
            deserialize_circuit(&compact_bytes, CircuitFormat::CompactJson).unwrap();
        assert_eq!(restored_pretty.gates.len(), restored_compact.gates.len());
    }

    #[test]
    fn binary_round_trip() {
        let mut circuit = SerializableCircuit::new(4);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        circuit.add_gate(SerializableGate::new(
            "Rz",
            vec![2],
            vec![],
            vec![std::f64::consts::FRAC_PI_4],
        ));
        circuit.add_gate(SerializableGate::new(
            "Toffoli",
            vec![3],
            vec![0, 1],
            vec![],
        ));

        let config = SerdeConfig::default();
        let binary =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();
        let restored =
            deserialize_circuit(&binary, CircuitFormat::Binary).unwrap();

        assert_eq!(restored.num_qubits, 4);
        assert_eq!(restored.gates.len(), 4);
        assert_eq!(restored.gates[0].gate_type, "H");
        assert_eq!(restored.gates[1].gate_type, "CNOT");
        assert_eq!(restored.gates[1].controls, vec![0]);
        assert_eq!(restored.gates[2].gate_type, "Rz");
        assert!(
            (restored.gates[2].params[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-15
        );
        assert_eq!(restored.gates[3].gate_type, "Toffoli");
        assert_eq!(restored.gates[3].controls, vec![0, 1]);
    }

    #[test]
    fn binary_smaller() {
        let mut circuit = SerializableCircuit::new(5);
        for i in 0..20 {
            circuit.add_gate(SerializableGate::new(
                "Rx",
                vec![i % 5],
                vec![],
                vec![i as f64 * 0.1],
            ));
        }

        let config = SerdeConfig::default();
        let json_bytes =
            serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let binary_bytes =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();

        assert!(
            binary_bytes.len() < json_bytes.len(),
            "binary ({} bytes) should be smaller than JSON ({} bytes)",
            binary_bytes.len(),
            json_bytes.len()
        );
    }

    #[test]
    fn circuit_library_add_search() {
        let mut lib = CircuitLibrary::new();

        // Add a Bell circuit
        let mut bell = bell_circuit();
        bell.metadata = Some(CircuitMetadata {
            name: "Bell State".to_string(),
            tags: vec!["entanglement".to_string(), "basic".to_string()],
            ..Default::default()
        });
        lib.add("bell_state", bell);

        // Add a GHZ circuit
        let mut ghz = SerializableCircuit::new(3);
        ghz.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        ghz.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        ghz.add_gate(SerializableGate::new("CNOT", vec![2], vec![1], vec![]));
        ghz.metadata = Some(CircuitMetadata {
            name: "GHZ State".to_string(),
            tags: vec!["entanglement".to_string(), "multi-qubit".to_string()],
            ..Default::default()
        });
        lib.add("ghz_state", ghz);

        // Add a QFT circuit
        let mut qft = SerializableCircuit::new(4);
        qft.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        qft.metadata = Some(CircuitMetadata {
            name: "QFT".to_string(),
            tags: vec!["algorithm".to_string()],
            ..Default::default()
        });
        lib.add("qft_4", qft);

        assert_eq!(lib.len(), 3);

        // Search by name
        let bell_results = lib.search_by_name("bell");
        assert_eq!(bell_results.len(), 1);
        assert_eq!(bell_results[0].0, "bell_state");

        // Search by tag
        let entangled = lib.search_by_tag("entanglement");
        assert_eq!(entangled.len(), 2);

        let algo = lib.search_by_tag("algorithm");
        assert_eq!(algo.len(), 1);

        // Filter by qubit count
        let big = lib.filter_by_qubit_count(3);
        assert_eq!(big.len(), 2); // GHZ (3) + QFT (4)

        // Get by name
        assert!(lib.get("bell_state").is_some());
        assert!(lib.get("nonexistent").is_none());
    }

    #[test]
    fn format_conversion() {
        let mut circuit = SerializableCircuit::new(2);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        circuit.add_gate(SerializableGate::new(
            "Rz",
            vec![0],
            vec![],
            vec![1.5707963267948966],
        ));

        let config = SerdeConfig::default();

        // JSON -> Binary
        let json_bytes =
            serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let binary_bytes = convert_format(
            &json_bytes,
            CircuitFormat::Json,
            CircuitFormat::Binary,
            &config,
        )
        .unwrap();

        // Binary -> JSON
        let json_back = convert_format(
            &binary_bytes,
            CircuitFormat::Binary,
            CircuitFormat::Json,
            &config,
        )
        .unwrap();
        let restored =
            deserialize_circuit(&json_back, CircuitFormat::Json).unwrap();
        assert_eq!(restored.gates.len(), 3);
        assert_eq!(restored.gates[0].gate_type, "H");

        // JSON -> QASM3
        let qasm_bytes = convert_format(
            &json_bytes,
            CircuitFormat::Json,
            CircuitFormat::OpenQASM3,
            &config,
        )
        .unwrap();
        let qasm_str = std::str::from_utf8(&qasm_bytes).unwrap();
        assert!(qasm_str.contains("OPENQASM 3.0"));
        assert!(qasm_str.contains("h q[0]"));
        assert!(qasm_str.contains("cx q[0], q[1]"));
    }

    #[test]
    fn version_header() {
        let circuit = SerializableCircuit::new(1);
        let config = SerdeConfig::default();

        // JSON version
        let json_bytes =
            serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let json_str = std::str::from_utf8(&json_bytes).unwrap();
        assert!(json_str.contains(&format!("\"version\""))); // version field present
        assert!(json_str.contains(&FORMAT_VERSION.to_string()));

        // Binary version: magic + version in header
        let binary =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();
        assert_eq!(&binary[0..4], BINARY_MAGIC);
        let ver = u32::from_le_bytes([binary[4], binary[5], binary[6], binary[7]]);
        assert_eq!(ver, FORMAT_VERSION);
    }

    #[test]
    fn invalid_json_error() {
        let bad_json = b"this is not json {{{";
        let result = deserialize_circuit(bad_json, CircuitFormat::Json);
        assert!(result.is_err());
        match result.unwrap_err() {
            CircuitSerdeError::DeserializationFailed(_) => {} // expected
            other => panic!("expected DeserializationFailed, got: {:?}", other),
        }

        // Valid JSON but missing required fields
        let bad_structure = b"{\"foo\": 42}";
        let result = deserialize_circuit(bad_structure, CircuitFormat::Json);
        assert!(result.is_err());

        // Truncated binary
        let bad_binary = b"NQCR";
        let result = deserialize_circuit(bad_binary, CircuitFormat::Binary);
        assert!(result.is_err());
    }

    #[test]
    fn qasm3_export() {
        let mut circuit = SerializableCircuit::new(3);
        circuit.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        circuit.add_gate(SerializableGate::new(
            "Rz",
            vec![2],
            vec![],
            vec![std::f64::consts::FRAC_PI_2],
        ));
        circuit.add_gate(SerializableGate::new("X", vec![1], vec![], vec![]));
        circuit.add_gate(SerializableGate::new("CZ", vec![2], vec![0], vec![]));

        let config = SerdeConfig::default();
        let bytes =
            serialize_circuit(&circuit, CircuitFormat::OpenQASM3, &config).unwrap();
        let qasm = std::str::from_utf8(&bytes).unwrap();

        assert!(qasm.starts_with("OPENQASM 3.0;"));
        assert!(qasm.contains("include \"stdgates.inc\";"));
        assert!(qasm.contains("qubit[3] q;"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("rz("));
        assert!(qasm.contains("x q[1];"));
        assert!(qasm.contains("cz q[0], q[2];"));
    }

    #[test]
    fn large_circuit() {
        let num_qubits = 10;
        let mut circuit = SerializableCircuit::new(num_qubits);

        // Build a 100-gate circuit with a mix of gate types
        for i in 0..100 {
            let gate = match i % 5 {
                0 => SerializableGate::new("H", vec![i % num_qubits], vec![], vec![]),
                1 => SerializableGate::new(
                    "CNOT",
                    vec![(i + 1) % num_qubits],
                    vec![i % num_qubits],
                    vec![],
                ),
                2 => SerializableGate::new(
                    "Rz",
                    vec![i % num_qubits],
                    vec![],
                    vec![i as f64 * 0.0314159],
                ),
                3 => SerializableGate::new("X", vec![i % num_qubits], vec![], vec![]),
                4 => SerializableGate::new(
                    "CRz",
                    vec![(i + 1) % num_qubits],
                    vec![i % num_qubits],
                    vec![i as f64 * 0.1],
                ),
                _ => unreachable!(),
            };
            circuit.add_gate(gate);
        }

        assert_eq!(circuit.gates.len(), 100);

        let config = SerdeConfig::default();

        // JSON round-trip
        let json_bytes =
            serialize_circuit(&circuit, CircuitFormat::Json, &config).unwrap();
        let restored_json =
            deserialize_circuit(&json_bytes, CircuitFormat::Json).unwrap();
        assert_eq!(restored_json.gates.len(), 100);
        assert_eq!(restored_json.num_qubits, num_qubits);

        // Verify gate types match
        for (original, restored) in circuit.gates.iter().zip(restored_json.gates.iter()) {
            assert_eq!(original.gate_type, restored.gate_type);
            assert_eq!(original.targets, restored.targets);
            assert_eq!(original.controls, restored.controls);
            assert_eq!(original.params.len(), restored.params.len());
            for (p_orig, p_rest) in original.params.iter().zip(restored.params.iter()) {
                assert!(
                    (p_orig - p_rest).abs() < 1e-12,
                    "parameter mismatch: {} vs {}",
                    p_orig,
                    p_rest
                );
            }
        }

        // Binary round-trip
        let binary_bytes =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();
        let restored_binary =
            deserialize_circuit(&binary_bytes, CircuitFormat::Binary).unwrap();
        assert_eq!(restored_binary.gates.len(), 100);

        for (original, restored) in circuit.gates.iter().zip(restored_binary.gates.iter())
        {
            assert_eq!(original.gate_type, restored.gate_type);
            assert_eq!(original.targets, restored.targets);
            assert_eq!(original.controls, restored.controls);
            for (p_orig, p_rest) in original.params.iter().zip(restored.params.iter()) {
                assert!(
                    (p_orig - p_rest).abs() < 1e-15,
                    "binary parameter mismatch: {} vs {}",
                    p_orig,
                    p_rest
                );
            }
        }
    }

    #[test]
    fn gate_type_conversion() {
        // Verify round-trip through GateType enum
        let gate = SerializableGate::new("Rz", vec![0], vec![], vec![1.5]);
        let gate_type = gate.to_gate_type().unwrap();
        match gate_type {
            crate::gates::GateType::Rz(p) => assert!((p - 1.5).abs() < 1e-15),
            other => panic!("expected Rz, got {:?}", other),
        }

        // Round-trip via from_gate_type
        let restored =
            SerializableGate::from_gate_type(&gate_type, vec![0], vec![]);
        assert_eq!(restored.gate_type, "Rz");
        assert_eq!(restored.params, vec![1.5]);
    }

    #[test]
    fn circuit_depth_computation() {
        // H on q0, then CNOT(q0, q1) => depth 2
        let circuit = bell_circuit();
        assert_eq!(circuit.compute_depth(), 2);

        // Parallel H on q0 and q1 => depth 1
        let mut parallel = SerializableCircuit::new(2);
        parallel.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        parallel.add_gate(SerializableGate::new("H", vec![1], vec![], vec![]));
        assert_eq!(parallel.compute_depth(), 1);

        // Empty circuit => depth 0
        let empty = SerializableCircuit::new(2);
        assert_eq!(empty.compute_depth(), 0);
    }

    #[test]
    fn library_json_round_trip() {
        let mut lib = CircuitLibrary::new();
        lib.add("bell", bell_circuit());

        let mut ghz = SerializableCircuit::new(3);
        ghz.add_gate(SerializableGate::new("H", vec![0], vec![], vec![]));
        ghz.add_gate(SerializableGate::new("CNOT", vec![1], vec![0], vec![]));
        ghz.add_gate(SerializableGate::new("CNOT", vec![2], vec![1], vec![]));
        lib.add("ghz", ghz);

        let config = SerdeConfig::default();
        let bytes = lib.to_json(&config).unwrap();
        let restored = CircuitLibrary::from_json(&bytes).unwrap();

        assert_eq!(restored.len(), 2);
        assert!(restored.get("bell").is_some());
        assert!(restored.get("ghz").is_some());
        assert_eq!(restored.get("ghz").unwrap().gates.len(), 3);
    }

    #[test]
    fn custom_gate_binary_round_trip() {
        let mut circuit = SerializableCircuit::new(2);
        circuit.add_gate(SerializableGate::new(
            "MyCustomGate",
            vec![0, 1],
            vec![],
            vec![0.5, 1.0],
        ));

        let config = SerdeConfig::default();
        let binary =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();
        let restored =
            deserialize_circuit(&binary, CircuitFormat::Binary).unwrap();

        assert_eq!(restored.gates.len(), 1);
        assert_eq!(restored.gates[0].gate_type, "MyCustomGate");
        assert_eq!(restored.gates[0].targets, vec![0, 1]);
        assert_eq!(restored.gates[0].params, vec![0.5, 1.0]);
    }

    #[test]
    fn binary_version_mismatch() {
        let circuit = SerializableCircuit::new(1);
        let config = SerdeConfig::default();
        let mut binary =
            serialize_circuit(&circuit, CircuitFormat::Binary, &config).unwrap();

        // Overwrite version to a future version
        let future_version: u32 = FORMAT_VERSION + 100;
        binary[4..8].copy_from_slice(&future_version.to_le_bytes());

        let result = deserialize_circuit(&binary, CircuitFormat::Binary);
        assert!(result.is_err());
        match result.unwrap_err() {
            CircuitSerdeError::VersionMismatch { expected, found } => {
                assert_eq!(expected, FORMAT_VERSION);
                assert_eq!(found, future_version);
            }
            other => panic!("expected VersionMismatch, got: {:?}", other),
        }
    }
}
