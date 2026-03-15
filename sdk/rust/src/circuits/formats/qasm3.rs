//! OpenQASM 3.0 Parser for nQPU-Metal
//!
//! Hand-written recursive descent parser supporting a subset of OpenQASM 2.0 and 3.0.
//! Does not depend on the `nom` crate — uses simple string-based tokenization
//! so the module compiles regardless of feature flags.
//!
//! **Note**: This is a partial OpenQASM 3.0 implementation covering the most common
//! constructs. Advanced features (classical types, arrays, extern functions, pulse-level
//! control via `defcal`) are not yet supported.
//!
//! # Supported syntax
//!
//! **QASM 3.0 subset**: `qubit[N] q;`, `bit[N] c;`, `q[i]` indexing, gate definitions,
//! subroutine definitions, `if`, `while`, `for`, `measure`, `reset`, `barrier`,
//! `delay`, and all standard gate names.
//!
//! **QASM 2.0 backward-compat**: `qreg q[N];`, `creg c[N];`, `include` directives.
//!
//! # Example
//!
//! ```ignore
//! use nqpu_metal::qasm3::QASM3Parser;
//!
//! let src = r#"
//! OPENQASM 3.0;
//! qubit[2] q;
//! bit[2] c;
//! h q[0];
//! cx q[0], q[1];
//! c[0] = measure q[0];
//! "#;
//!
//! let mut parser = QASM3Parser::new();
//! let program = parser.parse(src).unwrap();
//! let gates = QASM3Parser::to_gates(&program).unwrap();
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::gates::{Gate, GateType};
use crate::QuantumSimulator;
use crate::C64;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Detected QASM version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QasmVersion {
    V2_0,
    V3_0,
}

/// A qubit reference: `reg[index]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QasmQubit {
    pub reg: String,
    pub index: usize,
}

/// A classical-bit reference: `reg[index]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QasmCbit {
    pub reg: String,
    pub index: usize,
}

/// A classical condition used in `if` / `while` statements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassicalCondition {
    pub register: String,
    pub value: u64,
}

/// Parsed QASM3 statement.
#[derive(Clone, Debug, PartialEq)]
pub enum QASM3Statement {
    /// Gate application: `gate_name(params) target_qubits;`
    GateCall {
        name: String,
        params: Vec<f64>,
        qubits: Vec<QasmQubit>,
    },
    /// Classical if: `if (creg == val) { body }`
    If {
        condition: ClassicalCondition,
        body: Vec<QASM3Statement>,
    },
    /// While loop: `while (creg == val) { body }`
    While {
        condition: ClassicalCondition,
        body: Vec<QASM3Statement>,
    },
    /// For loop: `for uint i in [start:end] { body }`
    For {
        var: String,
        range: (i64, i64),
        body: Vec<QASM3Statement>,
    },
    /// Subroutine call.
    SubroutineCall {
        name: String,
        args: Vec<f64>,
        qubits: Vec<QasmQubit>,
    },
    /// `barrier q;` or `barrier q[0], q[1];`
    Barrier { qubits: Vec<QasmQubit> },
    /// `delay(duration) q[0];`
    Delay {
        duration: f64,
        qubits: Vec<QasmQubit>,
    },
    /// `measure q[0] -> c[0];` or `c[0] = measure q[0];`
    Measure { qubit: QasmQubit, cbit: QasmCbit },
    /// `reset q[0];`
    Reset { qubit: QasmQubit },
    /// Qubit register declaration: `qubit[N] name;` or `qreg name[N];`
    QubitDecl { name: String, size: usize },
    /// Classical bit register declaration: `bit[N] name;` or `creg name[N];`
    CbitDecl { name: String, size: usize },
    /// Gate definition.
    GateDef {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Vec<QASM3Statement>,
    },
    /// Subroutine definition.
    SubroutineDef {
        name: String,
        params: Vec<String>,
        qubits: Vec<String>,
        body: Vec<QASM3Statement>,
    },
}

/// A fully parsed QASM program.
#[derive(Clone, Debug)]
pub struct QASM3Program {
    pub version: QasmVersion,
    pub statements: Vec<QASM3Statement>,
    pub num_qubits: usize,
    pub num_cbits: usize,
}

/// Errors produced by the QASM3 parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QASM3Error {
    ParseError(String),
    UndefinedGate(String),
    UndefinedRegister(String),
    QubitOutOfRange {
        reg: String,
        index: usize,
        size: usize,
    },
    InvalidSyntax(String),
}

impl fmt::Display for QASM3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QASM3Error::ParseError(msg) => write!(f, "parse error: {}", msg),
            QASM3Error::UndefinedGate(name) => write!(f, "undefined gate: {}", name),
            QASM3Error::UndefinedRegister(name) => write!(f, "undefined register: {}", name),
            QASM3Error::QubitOutOfRange { reg, index, size } => {
                write!(f, "qubit {}[{}] out of range (size {})", reg, index, size)
            }
            QASM3Error::InvalidSyntax(msg) => write!(f, "invalid syntax: {}", msg),
        }
    }
}

impl std::error::Error for QASM3Error {}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct GateDefinition {
    params: Vec<String>,
    qubits: Vec<String>,
    body: Vec<QASM3Statement>,
}

#[derive(Clone, Debug)]
struct SubroutineDefinition {
    params: Vec<String>,
    qubits: Vec<String>,
    body: Vec<QASM3Statement>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Main QASM3 parser.
///
/// Maintains register and definition state across a parse invocation.
pub struct QASM3Parser {
    qubit_registers: HashMap<String, usize>,
    cbit_registers: HashMap<String, usize>,
    gate_definitions: HashMap<String, GateDefinition>,
    subroutines: HashMap<String, SubroutineDefinition>,
}

impl QASM3Parser {
    /// Create a new parser instance.
    pub fn new() -> Self {
        Self {
            qubit_registers: HashMap::new(),
            cbit_registers: HashMap::new(),
            gate_definitions: HashMap::new(),
            subroutines: HashMap::new(),
        }
    }

    /// Detect whether `input` declares QASM 2.0 or 3.0.
    pub fn detect_version(input: &str) -> QasmVersion {
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("OPENQASM") {
                if trimmed.contains("3") {
                    return QasmVersion::V3_0;
                }
                return QasmVersion::V2_0;
            }
        }
        // Default to 3.0 when no header present.
        QasmVersion::V3_0
    }

    /// Parse a complete QASM source string.
    pub fn parse(&mut self, input: &str) -> Result<QASM3Program, QASM3Error> {
        self.qubit_registers.clear();
        self.cbit_registers.clear();
        self.gate_definitions.clear();
        self.subroutines.clear();

        let version = Self::detect_version(input);
        let lines = Self::preprocess(input);
        let statements = self.parse_lines(&lines, version)?;

        let num_qubits: usize = self.qubit_registers.values().sum();
        let num_cbits: usize = self.cbit_registers.values().sum();

        Ok(QASM3Program {
            version,
            statements,
            num_qubits,
            num_cbits,
        })
    }

    /// Convert a parsed QASM program into a list of nQPU-Metal [`Gate`]s.
    ///
    /// Declarations, definitions, and control-flow statements are skipped in
    /// gate output; only concrete gate applications and measurements (which
    /// are ignored for pure unitary simulation) are lowered.
    pub fn to_gates(program: &QASM3Program) -> Result<Vec<Gate>, QASM3Error> {
        let mut qubit_offsets: HashMap<String, usize> = HashMap::new();
        let mut offset = 0usize;

        // First pass: build register offset map from declarations.
        for stmt in &program.statements {
            if let QASM3Statement::QubitDecl { name, size } = stmt {
                qubit_offsets.insert(name.clone(), offset);
                offset += size;
            }
        }

        let mut gates = Vec::new();
        Self::lower_statements(&program.statements, &qubit_offsets, &mut gates)?;
        Ok(gates)
    }

    // ------------------------------------------------------------------
    // Preprocessing
    // ------------------------------------------------------------------

    /// Strip comments, collapse multi-line braced blocks into single logical
    /// lines, and return a list of trimmed non-empty logical lines.
    fn preprocess(input: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut current = String::new();
        let mut brace_depth: i32 = 0;

        for raw_line in input.lines() {
            // Strip line comments.
            let line = match raw_line.find("//") {
                Some(pos) => &raw_line[..pos],
                None => raw_line,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            for ch in line.chars() {
                if ch == '{' {
                    brace_depth += 1;
                } else if ch == '}' {
                    brace_depth -= 1;
                }
                current.push(ch);
            }

            if brace_depth <= 0 {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    out.push(s);
                }
                current.clear();
                brace_depth = 0;
            } else {
                // Inside a braced block; add a space so tokens don't run
                // together across source lines.
                current.push(' ');
            }
        }

        // Flush any trailing content (malformed input without closing brace).
        let s = current.trim().to_string();
        if !s.is_empty() {
            out.push(s);
        }
        out
    }

    // ------------------------------------------------------------------
    // Line-level dispatch
    // ------------------------------------------------------------------

    fn parse_lines(
        &mut self,
        lines: &[String],
        version: QasmVersion,
    ) -> Result<Vec<QASM3Statement>, QASM3Error> {
        let mut stmts = Vec::new();

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Skip version and include directives.
            if line.starts_with("OPENQASM") || line.starts_with("include") {
                continue;
            }

            if let Some(s) = self.parse_single_line(line, version)? {
                stmts.push(s);
            }
        }
        Ok(stmts)
    }

    fn parse_single_line(
        &mut self,
        line: &str,
        version: QasmVersion,
    ) -> Result<Option<QASM3Statement>, QASM3Error> {
        let line = line.trim().trim_end_matches(';');
        if line.is_empty() {
            return Ok(None);
        }

        // QASM 2.0 register declarations.
        if line.starts_with("qreg") {
            return self.parse_qreg(line).map(Some);
        }
        if line.starts_with("creg") {
            return self.parse_creg(line).map(Some);
        }

        // QASM 3.0 typed declarations.
        if line.starts_with("qubit") {
            return self.parse_qubit_decl(line).map(Some);
        }
        if line.starts_with("bit") && !line.starts_with("bit_") {
            return self.parse_cbit_decl(line).map(Some);
        }

        // Gate/subroutine definitions.
        if line.starts_with("gate ") {
            return self.parse_gate_def(line).map(Some);
        }
        if line.starts_with("def ") {
            return self.parse_subroutine_def(line).map(Some);
        }

        // Control flow.
        if line.starts_with("if") {
            return self.parse_if(line, version).map(Some);
        }
        if line.starts_with("while") {
            return self.parse_while(line, version).map(Some);
        }
        if line.starts_with("for") {
            return self.parse_for(line, version).map(Some);
        }

        // Special statements.
        if line.starts_with("measure") {
            return self.parse_measure_arrow(line).map(Some);
        }
        // QASM 3.0 assignment-style measure: c[0] = measure q[0]
        if line.contains("= measure") {
            return self.parse_measure_assign(line).map(Some);
        }
        if line.starts_with("reset") {
            return self.parse_reset(line).map(Some);
        }
        if line.starts_with("barrier") {
            return self.parse_barrier(line).map(Some);
        }
        if line.starts_with("delay") {
            return self.parse_delay(line).map(Some);
        }

        // Default: gate call.
        self.parse_gate_call(line).map(Some)
    }

    // ------------------------------------------------------------------
    // Declarations
    // ------------------------------------------------------------------

    /// `qreg q[2];`
    fn parse_qreg(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("qreg").unwrap_or("").trim();
        let (name, size) = Self::parse_name_bracket_size(rest)?;
        self.qubit_registers.insert(name.clone(), size);
        Ok(QASM3Statement::QubitDecl { name, size })
    }

    /// `creg c[2];`
    fn parse_creg(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("creg").unwrap_or("").trim();
        let (name, size) = Self::parse_name_bracket_size(rest)?;
        self.cbit_registers.insert(name.clone(), size);
        Ok(QASM3Statement::CbitDecl { name, size })
    }

    /// `qubit[2] q;`
    fn parse_qubit_decl(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("qubit").unwrap_or("").trim();
        let (size, name) = Self::parse_bracket_size_name(rest)?;
        self.qubit_registers.insert(name.clone(), size);
        Ok(QASM3Statement::QubitDecl { name, size })
    }

    /// `bit[2] c;`
    fn parse_cbit_decl(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("bit").unwrap_or("").trim();
        let (size, name) = Self::parse_bracket_size_name(rest)?;
        self.cbit_registers.insert(name.clone(), size);
        Ok(QASM3Statement::CbitDecl { name, size })
    }

    /// Parse `name[N]` returning `(name, N)`. Used by qreg/creg.
    fn parse_name_bracket_size(s: &str) -> Result<(String, usize), QASM3Error> {
        let s = s.trim().trim_end_matches(';');
        let open = s.find('[').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '[' in register declaration: {}", s))
        })?;
        let close = s.find(']').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected ']' in register declaration: {}", s))
        })?;
        let name = s[..open].trim().to_string();
        let size: usize = s[open + 1..close]
            .trim()
            .parse()
            .map_err(|_| QASM3Error::ParseError(format!("invalid register size in: {}", s)))?;
        Ok((name, size))
    }

    /// Parse `[N] name` returning `(N, name)`. Used by qubit/bit decls.
    fn parse_bracket_size_name(s: &str) -> Result<(usize, String), QASM3Error> {
        let s = s.trim().trim_end_matches(';');
        let open = s.find('[').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '[' in type declaration: {}", s))
        })?;
        let close = s.find(']').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected ']' in type declaration: {}", s))
        })?;
        let size: usize = s[open + 1..close]
            .trim()
            .parse()
            .map_err(|_| QASM3Error::ParseError(format!("invalid size in: {}", s)))?;
        let name = s[close + 1..].trim().to_string();
        if name.is_empty() {
            return Err(QASM3Error::InvalidSyntax(format!(
                "missing register name in: {}",
                s
            )));
        }
        Ok((size, name))
    }

    // ------------------------------------------------------------------
    // Gate and subroutine definitions
    // ------------------------------------------------------------------

    /// `gate mygate(param) q0, q1 { h q0; cx q0, q1; }`
    fn parse_gate_def(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("gate").unwrap_or("").trim();

        // Extract body between { }.
        let (header, body_str) = Self::split_braced_body(rest)?;

        let (name, params, qubit_names) = Self::parse_def_header(header)?;

        let body = self.parse_body_statements(body_str)?;

        self.gate_definitions.insert(
            name.clone(),
            GateDefinition {
                params: params.clone(),
                qubits: qubit_names.clone(),
                body: body.clone(),
            },
        );

        Ok(QASM3Statement::GateDef {
            name,
            params,
            qubits: qubit_names,
            body,
        })
    }

    /// `def mydef(param) qubit q0, qubit q1 { ... }`
    fn parse_subroutine_def(&mut self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("def").unwrap_or("").trim();
        let (header, body_str) = Self::split_braced_body(rest)?;
        let (name, params, qubit_names) = Self::parse_def_header(header)?;

        let body = self.parse_body_statements(body_str)?;

        self.subroutines.insert(
            name.clone(),
            SubroutineDefinition {
                params: params.clone(),
                qubits: qubit_names.clone(),
                body: body.clone(),
            },
        );

        Ok(QASM3Statement::SubroutineDef {
            name,
            params,
            qubits: qubit_names,
            body,
        })
    }

    /// Split `header { body }` returning (header, body_content).
    fn split_braced_body(s: &str) -> Result<(&str, &str), QASM3Error> {
        let open = s.find('{').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '{{' in definition: {}", s))
        })?;
        let close = s.rfind('}').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '}}' in definition: {}", s))
        })?;
        Ok((s[..open].trim(), s[open + 1..close].trim()))
    }

    /// Parse `name(p0, p1) q0, q1` into (name, param_names, qubit_names).
    fn parse_def_header(header: &str) -> Result<(String, Vec<String>, Vec<String>), QASM3Error> {
        let header = header.trim();
        let mut params = Vec::new();
        let mut qubits = Vec::new();

        let (name_part, rest) = if let Some(paren_open) = header.find('(') {
            let name = header[..paren_open].trim().to_string();
            let paren_close = header.find(')').ok_or_else(|| {
                QASM3Error::InvalidSyntax(format!("unmatched '(' in: {}", header))
            })?;
            let param_str = &header[paren_open + 1..paren_close];
            params = param_str
                .split(',')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
            (name, header[paren_close + 1..].trim())
        } else {
            // No parentheses -- split on first whitespace.
            match header.split_once(char::is_whitespace) {
                Some((name, rest)) => (name.trim().to_string(), rest.trim()),
                None => (header.to_string(), ""),
            }
        };

        for q in rest.split(',') {
            let q = q.trim().replace("qubit", "").trim().to_string();
            if !q.is_empty() {
                qubits.push(q);
            }
        }

        Ok((name_part, params, qubits))
    }

    /// Parse the semicolon-separated statements inside a braced body.
    fn parse_body_statements(&mut self, body: &str) -> Result<Vec<QASM3Statement>, QASM3Error> {
        let mut stmts = Vec::new();
        for part in body.split(';') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(s) = self.parse_single_line(part, QasmVersion::V3_0)? {
                stmts.push(s);
            }
        }
        Ok(stmts)
    }

    // ------------------------------------------------------------------
    // Control flow
    // ------------------------------------------------------------------

    /// `if (c == 1) { h q[0]; }`
    fn parse_if(&mut self, line: &str, version: QasmVersion) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("if").unwrap_or("").trim();
        let (cond, body_str) = self.parse_condition_and_body(rest)?;
        let body = self.parse_body_statements(body_str)?;
        let _ = version;
        Ok(QASM3Statement::If {
            condition: cond,
            body,
        })
    }

    /// `while (c == 0) { ... }`
    fn parse_while(
        &mut self,
        line: &str,
        version: QasmVersion,
    ) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("while").unwrap_or("").trim();
        let (cond, body_str) = self.parse_condition_and_body(rest)?;
        let body = self.parse_body_statements(body_str)?;
        let _ = version;
        Ok(QASM3Statement::While {
            condition: cond,
            body,
        })
    }

    /// `for uint i in [0:4] { ... }`
    fn parse_for(
        &mut self,
        line: &str,
        version: QasmVersion,
    ) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("for").unwrap_or("").trim();
        let (header, body_str) = Self::split_braced_body(rest)?;

        // header: "uint i in [0:4]" or "i in [0:4]"
        let header = header.trim();
        // Strip optional type keyword.
        let header = header
            .strip_prefix("uint")
            .or_else(|| header.strip_prefix("int"))
            .unwrap_or(header)
            .trim();

        // Split: "i in [0:4]"
        let parts: Vec<&str> = header.splitn(3, char::is_whitespace).collect();
        if parts.len() < 3 || parts[1] != "in" {
            return Err(QASM3Error::InvalidSyntax(format!(
                "invalid for loop header: {}",
                header
            )));
        }
        let var = parts[0].to_string();
        let range_str = parts[2].trim().trim_matches(|c| c == '[' || c == ']');
        let range_parts: Vec<&str> = range_str.split(':').collect();
        if range_parts.len() != 2 {
            return Err(QASM3Error::InvalidSyntax(format!(
                "expected [start:end] range, got: {}",
                parts[2]
            )));
        }
        let start: i64 = range_parts[0].trim().parse().map_err(|_| {
            QASM3Error::ParseError(format!("invalid range start: {}", range_parts[0]))
        })?;
        let end: i64 = range_parts[1].trim().parse().map_err(|_| {
            QASM3Error::ParseError(format!("invalid range end: {}", range_parts[1]))
        })?;

        let body = self.parse_body_statements(body_str)?;
        let _ = version;

        Ok(QASM3Statement::For {
            var,
            range: (start, end),
            body,
        })
    }

    /// Extract `(condition, body_str)` from `(reg == val) { body }`.
    fn parse_condition_and_body<'a>(
        &self,
        s: &'a str,
    ) -> Result<(ClassicalCondition, &'a str), QASM3Error> {
        // Find the condition in parentheses.
        let paren_open = s.find('(').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '(' in condition: {}", s))
        })?;
        let paren_close = s.find(')').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected ')' in condition: {}", s))
        })?;
        let cond_str = &s[paren_open + 1..paren_close];
        let cond = Self::parse_condition(cond_str)?;

        // The rest should contain { body }.
        let rest = s[paren_close + 1..].trim();
        let brace_open = rest.find('{').ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!("expected '{{' after condition in: {}", s))
        })?;
        let brace_close = rest
            .rfind('}')
            .ok_or_else(|| QASM3Error::InvalidSyntax(format!("expected '}}' in: {}", s)))?;
        let body_str = rest[brace_open + 1..brace_close].trim();
        Ok((cond, body_str))
    }

    /// Parse `reg == val` into a ClassicalCondition.
    fn parse_condition(s: &str) -> Result<ClassicalCondition, QASM3Error> {
        let parts: Vec<&str> = s.split("==").collect();
        if parts.len() != 2 {
            return Err(QASM3Error::InvalidSyntax(format!(
                "expected 'reg == val', got: {}",
                s
            )));
        }
        let register = parts[0].trim().to_string();
        let value: u64 = parts[1].trim().parse().map_err(|_| {
            QASM3Error::ParseError(format!("invalid condition value: {}", parts[1].trim()))
        })?;
        Ok(ClassicalCondition { register, value })
    }

    // ------------------------------------------------------------------
    // Special statements
    // ------------------------------------------------------------------

    /// `measure q[0] -> c[0]`
    fn parse_measure_arrow(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("measure").unwrap_or("").trim();
        let parts: Vec<&str> = rest.split("->").collect();
        if parts.len() != 2 {
            return Err(QASM3Error::InvalidSyntax(format!(
                "expected 'measure qubit -> cbit', got: {}",
                line
            )));
        }
        let qubit = Self::parse_qubit_ref(parts[0].trim())?;
        let cbit = Self::parse_cbit_ref(parts[1].trim())?;
        Ok(QASM3Statement::Measure { qubit, cbit })
    }

    /// `c[0] = measure q[0]`
    fn parse_measure_assign(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let parts: Vec<&str> = line.split("= measure").collect();
        if parts.len() != 2 {
            return Err(QASM3Error::InvalidSyntax(format!(
                "expected 'cbit = measure qubit', got: {}",
                line
            )));
        }
        let cbit = Self::parse_cbit_ref(parts[0].trim())?;
        let qubit = Self::parse_qubit_ref(parts[1].trim())?;
        Ok(QASM3Statement::Measure { qubit, cbit })
    }

    /// `reset q[0]`
    fn parse_reset(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("reset").unwrap_or("").trim();
        let qubit = Self::parse_qubit_ref(rest)?;
        Ok(QASM3Statement::Reset { qubit })
    }

    /// `barrier q[0], q[1]` or `barrier q`
    fn parse_barrier(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("barrier").unwrap_or("").trim();
        let qubits = Self::parse_qubit_list(rest)?;
        Ok(QASM3Statement::Barrier { qubits })
    }

    /// `delay(100) q[0]`
    fn parse_delay(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let rest = line.strip_prefix("delay").unwrap_or("").trim();
        let (duration, qubit_str) = Self::parse_parenthesized_param_and_rest(rest)?;
        let qubits = Self::parse_qubit_list(qubit_str)?;
        Ok(QASM3Statement::Delay {
            duration: duration.first().copied().unwrap_or(0.0),
            qubits,
        })
    }

    // ------------------------------------------------------------------
    // Gate call
    // ------------------------------------------------------------------

    /// `h q[0]` or `rx(pi/4) q[0]` or `cx q[0], q[1]`
    fn parse_gate_call(&self, line: &str) -> Result<QASM3Statement, QASM3Error> {
        let line = line.trim();

        // Split gate name (possibly with params) from qubit arguments.
        let (name, params, qubit_str) = if let Some(paren_open) = line.find('(') {
            let name = line[..paren_open].trim().to_string();
            let paren_close = line
                .find(')')
                .ok_or_else(|| QASM3Error::InvalidSyntax(format!("unmatched '(' in: {}", line)))?;
            let param_str = &line[paren_open + 1..paren_close];
            let params = Self::parse_float_list(param_str)?;
            let rest = line[paren_close + 1..].trim();
            (name, params, rest)
        } else {
            // No parentheses: split on first whitespace.
            match line.split_once(char::is_whitespace) {
                Some((name, rest)) => (name.trim().to_string(), Vec::new(), rest.trim()),
                None => {
                    return Err(QASM3Error::InvalidSyntax(format!(
                        "expected gate arguments in: {}",
                        line
                    )));
                }
            }
        };

        let qubits = Self::parse_qubit_list(qubit_str)?;
        Ok(QASM3Statement::GateCall {
            name,
            params,
            qubits,
        })
    }

    // ------------------------------------------------------------------
    // Reference parsers
    // ------------------------------------------------------------------

    /// Parse `q[0]` into a QasmQubit.
    fn parse_qubit_ref(s: &str) -> Result<QasmQubit, QASM3Error> {
        let s = s.trim().trim_end_matches(';');
        if let Some(open) = s.find('[') {
            let close = s.find(']').ok_or_else(|| {
                QASM3Error::InvalidSyntax(format!("expected ']' in qubit ref: {}", s))
            })?;
            let reg = s[..open].trim().to_string();
            let index: usize = s[open + 1..close]
                .trim()
                .parse()
                .map_err(|_| QASM3Error::ParseError(format!("invalid qubit index in: {}", s)))?;
            Ok(QasmQubit { reg, index })
        } else {
            // Bare register name -- index 0.
            Ok(QasmQubit {
                reg: s.trim().to_string(),
                index: 0,
            })
        }
    }

    /// Parse `c[0]` into a QasmCbit.
    fn parse_cbit_ref(s: &str) -> Result<QasmCbit, QASM3Error> {
        let s = s.trim().trim_end_matches(';');
        if let Some(open) = s.find('[') {
            let close = s.find(']').ok_or_else(|| {
                QASM3Error::InvalidSyntax(format!("expected ']' in cbit ref: {}", s))
            })?;
            let reg = s[..open].trim().to_string();
            let index: usize = s[open + 1..close]
                .trim()
                .parse()
                .map_err(|_| QASM3Error::ParseError(format!("invalid cbit index in: {}", s)))?;
            Ok(QasmCbit { reg, index })
        } else {
            Ok(QasmCbit {
                reg: s.trim().to_string(),
                index: 0,
            })
        }
    }

    /// Parse a comma-separated list of qubit references.
    fn parse_qubit_list(s: &str) -> Result<Vec<QasmQubit>, QASM3Error> {
        let s = s.trim().trim_end_matches(';');
        let mut qubits = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            qubits.push(Self::parse_qubit_ref(part)?);
        }
        Ok(qubits)
    }

    /// Parse a comma-separated list of f64 values (supporting pi expressions).
    fn parse_float_list(s: &str) -> Result<Vec<f64>, QASM3Error> {
        let mut vals = Vec::new();
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            vals.push(Self::parse_float_expr(part)?);
        }
        Ok(vals)
    }

    /// Parse a simple float expression, supporting `pi`, `-pi`, `pi/N`, `N*pi`.
    fn parse_float_expr(s: &str) -> Result<f64, QASM3Error> {
        let s = s.trim();

        // Direct numeric literal.
        if let Ok(v) = s.parse::<f64>() {
            return Ok(v);
        }

        // Handle "pi" and common expressions.
        let s_lower = s.to_lowercase();

        if s_lower == "pi" {
            return Ok(std::f64::consts::PI);
        }
        if s_lower == "-pi" {
            return Ok(-std::f64::consts::PI);
        }

        // Division: "pi/4", "-pi/2", "3*pi/4"
        if s_lower.contains('/') {
            let parts: Vec<&str> = s.split('/').collect();
            if parts.len() == 2 {
                let numer = Self::parse_float_expr(parts[0])?;
                let denom = Self::parse_float_expr(parts[1])?;
                if denom.abs() < 1e-30 {
                    return Err(QASM3Error::ParseError(format!(
                        "division by zero in: {}",
                        s
                    )));
                }
                return Ok(numer / denom);
            }
        }

        // Multiplication: "2*pi", "pi*2"
        if s_lower.contains('*') {
            let parts: Vec<&str> = s.split('*').collect();
            if parts.len() == 2 {
                let a = Self::parse_float_expr(parts[0])?;
                let b = Self::parse_float_expr(parts[1])?;
                return Ok(a * b);
            }
        }

        // Negation: "-expr"
        if let Some(rest) = s.strip_prefix('-') {
            return Self::parse_float_expr(rest).map(|v| -v);
        }

        Err(QASM3Error::ParseError(format!(
            "cannot parse float expression: {}",
            s
        )))
    }

    /// Parse `(params) rest` returning (params, rest).
    fn parse_parenthesized_param_and_rest(s: &str) -> Result<(Vec<f64>, &str), QASM3Error> {
        let s = s.trim();
        if let Some(open) = s.find('(') {
            let close = s
                .find(')')
                .ok_or_else(|| QASM3Error::InvalidSyntax(format!("unmatched '(' in: {}", s)))?;
            let params = Self::parse_float_list(&s[open + 1..close])?;
            Ok((params, s[close + 1..].trim()))
        } else {
            Ok((Vec::new(), s))
        }
    }

    // ------------------------------------------------------------------
    // Gate lowering
    // ------------------------------------------------------------------

    /// Recursively lower statements to Gate list.
    fn lower_statements(
        stmts: &[QASM3Statement],
        offsets: &HashMap<String, usize>,
        out: &mut Vec<Gate>,
    ) -> Result<(), QASM3Error> {
        for stmt in stmts {
            match stmt {
                QASM3Statement::GateCall {
                    name,
                    params,
                    qubits,
                } => {
                    let gate = Self::lower_gate_call(name, params, qubits, offsets)?;
                    out.push(gate);
                }
                QASM3Statement::If { body, .. }
                | QASM3Statement::While { body, .. }
                | QASM3Statement::For { body, .. } => {
                    // Flatten control flow for pure unitary lowering.
                    Self::lower_statements(body, offsets, out)?;
                }
                // Declarations, definitions, measure, reset, barrier, delay
                // are not lowered to unitary gates.
                _ => {}
            }
        }
        Ok(())
    }

    /// Resolve a QasmQubit to a flat qubit index.
    fn resolve_qubit(
        qubit: &QasmQubit,
        offsets: &HashMap<String, usize>,
    ) -> Result<usize, QASM3Error> {
        let base = offsets
            .get(&qubit.reg)
            .ok_or_else(|| QASM3Error::UndefinedRegister(qubit.reg.clone()))?;
        Ok(base + qubit.index)
    }

    /// Lower a single gate call to an nQPU-Metal Gate.
    fn lower_gate_call(
        name: &str,
        params: &[f64],
        qubits: &[QasmQubit],
        offsets: &HashMap<String, usize>,
    ) -> Result<Gate, QASM3Error> {
        let resolved: Vec<usize> = qubits
            .iter()
            .map(|q| Self::resolve_qubit(q, offsets))
            .collect::<Result<Vec<_>, _>>()?;

        match name {
            // -- Single-qubit gates --
            "h" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::H, resolved[0]))
            }
            "x" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::X, resolved[0]))
            }
            "y" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::Y, resolved[0]))
            }
            "z" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::Z, resolved[0]))
            }
            "s" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::S, resolved[0]))
            }
            "t" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::T, resolved[0]))
            }
            "sx" => {
                Self::expect_qubits(name, &resolved, 1)?;
                Ok(Gate::single(GateType::SX, resolved[0]))
            }

            // -- Parameterized single-qubit --
            "rx" => {
                Self::expect_qubits(name, &resolved, 1)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::single(GateType::Rx(angle), resolved[0]))
            }
            "ry" => {
                Self::expect_qubits(name, &resolved, 1)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::single(GateType::Ry(angle), resolved[0]))
            }
            "rz" => {
                Self::expect_qubits(name, &resolved, 1)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::single(GateType::Rz(angle), resolved[0]))
            }
            "p" | "phase" => {
                Self::expect_qubits(name, &resolved, 1)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::single(GateType::Phase(angle), resolved[0]))
            }
            "u" | "u3" => {
                Self::expect_qubits(name, &resolved, 1)?;
                let theta = Self::expect_param(name, params, 0)?;
                let phi = Self::expect_param(name, params, 1)?;
                let lambda = Self::expect_param(name, params, 2)?;
                Ok(Gate::single(
                    GateType::U { theta, phi, lambda },
                    resolved[0],
                ))
            }

            // -- Two-qubit gates --
            "cx" | "cnot" | "CX" => {
                Self::expect_qubits(name, &resolved, 2)?;
                Ok(Gate::two(GateType::CNOT, resolved[0], resolved[1]))
            }
            "cz" => {
                Self::expect_qubits(name, &resolved, 2)?;
                Ok(Gate::two(GateType::CZ, resolved[0], resolved[1]))
            }
            "swap" => {
                Self::expect_qubits(name, &resolved, 2)?;
                Ok(Gate::swap(resolved[0], resolved[1]))
            }
            "iswap" => {
                Self::expect_qubits(name, &resolved, 2)?;
                Ok(Gate::iswap(resolved[0], resolved[1]))
            }

            // -- Controlled rotations --
            "crx" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::two(GateType::CRx(angle), resolved[0], resolved[1]))
            }
            "cry" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::two(GateType::CRy(angle), resolved[0], resolved[1]))
            }
            "crz" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::two(GateType::CRz(angle), resolved[0], resolved[1]))
            }
            "cr" | "cp" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::two(GateType::CR(angle), resolved[0], resolved[1]))
            }

            // -- Three-qubit gates --
            "ccx" | "toffoli" | "ccnot" => {
                Self::expect_qubits(name, &resolved, 3)?;
                Ok(Gate::toffoli(resolved[0], resolved[1], resolved[2]))
            }
            "ccz" => {
                Self::expect_qubits(name, &resolved, 3)?;
                Ok(Gate::ccz(resolved[0], resolved[1], resolved[2]))
            }

            // -- Two-qubit rotation gates --
            "rxx" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::rxx(resolved[0], resolved[1], angle))
            }
            "ryy" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::ryy(resolved[0], resolved[1], angle))
            }
            "rzz" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let angle = Self::expect_param(name, params, 0)?;
                Ok(Gate::rzz(resolved[0], resolved[1], angle))
            }

            // -- Controlled-SWAP (Fredkin gate) --
            "cswap" | "fredkin" => {
                Self::expect_qubits(name, &resolved, 3)?;
                Ok(Gate::cswap(resolved[0], resolved[1], resolved[2]))
            }

            // -- Generic controlled-U gate --
            "cu" => {
                Self::expect_qubits(name, &resolved, 2)?;
                let theta = Self::expect_param(name, params, 0)?;
                let phi = Self::expect_param(name, params, 1)?;
                let lambda = Self::expect_param(name, params, 2)?;
                let gamma = Self::expect_param(name, params, 3)?;
                Ok(Gate::cu(
                    resolved[0],
                    resolved[1],
                    theta,
                    phi,
                    lambda,
                    gamma,
                ))
            }

            _ => Err(QASM3Error::UndefinedGate(name.to_string())),
        }
    }

    /// Assert the expected number of qubit arguments.
    fn expect_qubits(
        gate_name: &str,
        resolved: &[usize],
        expected: usize,
    ) -> Result<(), QASM3Error> {
        if resolved.len() != expected {
            Err(QASM3Error::InvalidSyntax(format!(
                "gate '{}' expects {} qubit(s), got {}",
                gate_name,
                expected,
                resolved.len()
            )))
        } else {
            Ok(())
        }
    }

    /// Retrieve a parameter by index or return an error.
    fn expect_param(gate_name: &str, params: &[f64], index: usize) -> Result<f64, QASM3Error> {
        params.get(index).copied().ok_or_else(|| {
            QASM3Error::InvalidSyntax(format!(
                "gate '{}' missing parameter at index {}",
                gate_name, index
            ))
        })
    }
}

impl Default for QASM3Parser {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Version detection
    // ---------------------------------------------------------------

    #[test]
    fn test_detect_version_2() {
        let src = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
        assert_eq!(QASM3Parser::detect_version(src), QasmVersion::V2_0);
    }

    #[test]
    fn test_detect_version_3() {
        let src = "OPENQASM 3.0;\nqubit[2] q;\n";
        assert_eq!(QASM3Parser::detect_version(src), QasmVersion::V3_0);
    }

    #[test]
    fn test_detect_version_missing_defaults_to_3() {
        let src = "qubit[2] q;\nh q[0];\n";
        assert_eq!(QASM3Parser::detect_version(src), QasmVersion::V3_0);
    }

    // ---------------------------------------------------------------
    // QASM 2.0 -- Bell state
    // ---------------------------------------------------------------

    #[test]
    fn test_qasm2_bell_circuit() {
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        assert_eq!(prog.version, QasmVersion::V2_0);
        assert_eq!(prog.num_qubits, 2);
        assert_eq!(prog.num_cbits, 2);

        // QubitDecl, CbitDecl, H, CX, Measure, Measure = 6 statements.
        assert_eq!(prog.statements.len(), 6);

        let gates = QASM3Parser::to_gates(&prog).unwrap();
        assert_eq!(gates.len(), 2); // h, cx
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[1].gate_type, GateType::CNOT);
    }

    // ---------------------------------------------------------------
    // QASM 3.0 -- qubit[N] declaration
    // ---------------------------------------------------------------

    #[test]
    fn test_qasm3_qubit_decl() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        assert_eq!(prog.version, QasmVersion::V3_0);
        assert_eq!(prog.num_qubits, 2);
        assert_eq!(prog.num_cbits, 2);

        let gates = QASM3Parser::to_gates(&prog).unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[0].targets, vec![0]);
        assert_eq!(gates[1].gate_type, GateType::CNOT);
        assert_eq!(gates[1].controls, vec![0]);
        assert_eq!(gates[1].targets, vec![1]);
    }

    // ---------------------------------------------------------------
    // Parameterized gates
    // ---------------------------------------------------------------

    #[test]
    fn test_parameterized_gates() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
rx(pi/4) q[0];
ry(1.5707963) q[0];
rz(pi/2) q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 3);

        match &gates[0].gate_type {
            GateType::Rx(angle) => {
                assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
            }
            other => panic!("expected Rx, got {:?}", other),
        }

        match &gates[1].gate_type {
            GateType::Ry(angle) => {
                assert!((angle - 1.5707963).abs() < 1e-6);
            }
            other => panic!("expected Ry, got {:?}", other),
        }

        match &gates[2].gate_type {
            GateType::Rz(angle) => {
                assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
            }
            other => panic!("expected Rz, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // U gate with three parameters
    // ---------------------------------------------------------------

    #[test]
    fn test_u_gate() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
u(pi/2, 0, pi) q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 1);
        match &gates[0].gate_type {
            GateType::U { theta, phi, lambda } => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
                assert!(phi.abs() < 1e-10);
                assert!((lambda - std::f64::consts::PI).abs() < 1e-10);
            }
            other => panic!("expected U, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Measure and reset
    // ---------------------------------------------------------------

    #[test]
    fn test_measure_arrow_syntax() {
        let src = r#"
OPENQASM 2.0;
qreg q[2];
creg c[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let measures: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::Measure { .. }))
            .collect();
        assert_eq!(measures.len(), 2);
    }

    #[test]
    fn test_measure_assign_syntax() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
h q[0];
c[0] = measure q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let measures: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::Measure { .. }))
            .collect();
        assert_eq!(measures.len(), 1);

        if let QASM3Statement::Measure { qubit, cbit } = &measures[0] {
            assert_eq!(qubit.reg, "q");
            assert_eq!(qubit.index, 0);
            assert_eq!(cbit.reg, "c");
            assert_eq!(cbit.index, 0);
        }
    }

    #[test]
    fn test_reset() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
x q[0];
reset q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let resets: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::Reset { .. }))
            .collect();
        assert_eq!(resets.len(), 1);
    }

    // ---------------------------------------------------------------
    // Gate definition
    // ---------------------------------------------------------------

    #[test]
    fn test_gate_definition() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
gate bell a, b { h a; cx a, b; }
bell q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let defs: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::GateDef { .. }))
            .collect();
        assert_eq!(defs.len(), 1);

        if let QASM3Statement::GateDef {
            name, qubits, body, ..
        } = &defs[0]
        {
            assert_eq!(name, "bell");
            assert_eq!(qubits.len(), 2);
            assert_eq!(body.len(), 2);
        }
    }

    // ---------------------------------------------------------------
    // If statement
    // ---------------------------------------------------------------

    #[test]
    fn test_if_statement() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
c[0] = measure q[0];
if (c == 1) { x q[1]; }
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let ifs: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::If { .. }))
            .collect();
        assert_eq!(ifs.len(), 1);

        if let QASM3Statement::If { condition, body } = &ifs[0] {
            assert_eq!(condition.register, "c");
            assert_eq!(condition.value, 1);
            assert_eq!(body.len(), 1);
        }
    }

    // ---------------------------------------------------------------
    // For loop
    // ---------------------------------------------------------------

    #[test]
    fn test_for_loop() {
        let src = r#"
OPENQASM 3.0;
qubit[4] q;
for uint i in [0:4] { h q[0]; }
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let fors: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::For { .. }))
            .collect();
        assert_eq!(fors.len(), 1);

        if let QASM3Statement::For { var, range, body } = &fors[0] {
            assert_eq!(var, "i");
            assert_eq!(*range, (0, 4));
            assert_eq!(body.len(), 1);
        }
    }

    // ---------------------------------------------------------------
    // Multi-register program
    // ---------------------------------------------------------------

    #[test]
    fn test_multi_register() {
        let src = r#"
OPENQASM 3.0;
qubit[2] a;
qubit[3] b;
bit[5] c;
h a[0];
cx a[1], b[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        assert_eq!(prog.num_qubits, 5);
        assert_eq!(prog.num_cbits, 5);

        let gates = QASM3Parser::to_gates(&prog).unwrap();
        assert_eq!(gates.len(), 2);

        // a[0] -> index 0
        assert_eq!(gates[0].targets, vec![0]);

        // a[1] -> index 1 (control), b[0] -> index 2 (target)
        assert_eq!(gates[1].controls, vec![1]);
        assert_eq!(gates[1].targets, vec![2]);
    }

    // ---------------------------------------------------------------
    // Error: undefined gate
    // ---------------------------------------------------------------

    #[test]
    fn test_error_undefined_gate() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
foobar q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let result = QASM3Parser::to_gates(&prog);

        assert!(result.is_err());
        match result.unwrap_err() {
            QASM3Error::UndefinedGate(name) => assert_eq!(name, "foobar"),
            other => panic!("expected UndefinedGate, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Error: undefined register in to_gates
    // ---------------------------------------------------------------

    #[test]
    fn test_error_undefined_register() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
h z[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let result = QASM3Parser::to_gates(&prog);

        assert!(result.is_err());
        match result.unwrap_err() {
            QASM3Error::UndefinedRegister(name) => assert_eq!(name, "z"),
            other => panic!("expected UndefinedRegister, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Error: wrong number of qubits for gate
    // ---------------------------------------------------------------

    #[test]
    fn test_error_wrong_qubit_count() {
        let src = r#"
OPENQASM 3.0;
qubit[3] q;
h q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let result = QASM3Parser::to_gates(&prog);

        assert!(result.is_err());
        match result.unwrap_err() {
            QASM3Error::InvalidSyntax(msg) => {
                assert!(msg.contains("expects 1 qubit"), "got: {}", msg);
            }
            other => panic!("expected InvalidSyntax, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Barrier
    // ---------------------------------------------------------------

    #[test]
    fn test_barrier() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
h q[0];
barrier q[0], q[1];
cx q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let barriers: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::Barrier { .. }))
            .collect();
        assert_eq!(barriers.len(), 1);

        if let QASM3Statement::Barrier { qubits } = &barriers[0] {
            assert_eq!(qubits.len(), 2);
        }
    }

    // ---------------------------------------------------------------
    // Controlled rotations
    // ---------------------------------------------------------------

    #[test]
    fn test_controlled_rotations() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
crx(pi/4) q[0], q[1];
cry(pi/2) q[0], q[1];
crz(pi) q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 3);

        match &gates[0].gate_type {
            GateType::CRx(a) => assert!((a - std::f64::consts::FRAC_PI_4).abs() < 1e-10),
            other => panic!("expected CRx, got {:?}", other),
        }
        match &gates[1].gate_type {
            GateType::CRy(a) => assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-10),
            other => panic!("expected CRy, got {:?}", other),
        }
        match &gates[2].gate_type {
            GateType::CRz(a) => assert!((a - std::f64::consts::PI).abs() < 1e-10),
            other => panic!("expected CRz, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Three-qubit gates: Toffoli, CCZ
    // ---------------------------------------------------------------

    #[test]
    fn test_three_qubit_gates() {
        let src = r#"
OPENQASM 3.0;
qubit[3] q;
ccx q[0], q[1], q[2];
ccz q[0], q[1], q[2];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].gate_type, GateType::Toffoli);
        assert_eq!(gates[0].controls, vec![0, 1]);
        assert_eq!(gates[0].targets, vec![2]);

        assert_eq!(gates[1].gate_type, GateType::CCZ);
        assert_eq!(gates[1].controls, vec![0, 1]);
        assert_eq!(gates[1].targets, vec![2]);
    }

    // ---------------------------------------------------------------
    // Comments are stripped
    // ---------------------------------------------------------------

    #[test]
    fn test_comments() {
        let src = r#"
OPENQASM 3.0;
// This is a comment
qubit[2] q; // inline comment
h q[0]; // apply Hadamard
cx q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 2);
    }

    // ---------------------------------------------------------------
    // Float expression parsing (pi arithmetic)
    // ---------------------------------------------------------------

    #[test]
    fn test_float_expr_parsing() {
        let pi = std::f64::consts::PI;

        assert!((QASM3Parser::parse_float_expr("pi").unwrap() - pi).abs() < 1e-10);
        assert!((QASM3Parser::parse_float_expr("-pi").unwrap() + pi).abs() < 1e-10);
        assert!((QASM3Parser::parse_float_expr("pi/2").unwrap() - pi / 2.0).abs() < 1e-10);
        assert!((QASM3Parser::parse_float_expr("pi/4").unwrap() - pi / 4.0).abs() < 1e-10);
        assert!((QASM3Parser::parse_float_expr("2*pi").unwrap() - 2.0 * pi).abs() < 1e-10);
        assert!((QASM3Parser::parse_float_expr("3.14").unwrap() - 3.14).abs() < 1e-10);
    }

    // ---------------------------------------------------------------
    // SWAP and ISWAP gates
    // ---------------------------------------------------------------

    #[test]
    fn test_swap_and_iswap() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
swap q[0], q[1];
iswap q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].gate_type, GateType::SWAP);
        assert_eq!(gates[0].targets, vec![0, 1]);
        assert_eq!(gates[1].gate_type, GateType::ISWAP);
        assert_eq!(gates[1].targets, vec![0, 1]);
    }

    // ---------------------------------------------------------------
    // Delay statement
    // ---------------------------------------------------------------

    #[test]
    fn test_delay() {
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
delay(100) q[0];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        let delays: Vec<_> = prog
            .statements
            .iter()
            .filter(|s| matches!(s, QASM3Statement::Delay { .. }))
            .collect();
        assert_eq!(delays.len(), 1);

        if let QASM3Statement::Delay { duration, qubits } = &delays[0] {
            assert!((duration - 100.0).abs() < 1e-10);
            assert_eq!(qubits.len(), 1);
        }
    }

    // ---------------------------------------------------------------
    // Error display
    // ---------------------------------------------------------------

    #[test]
    fn test_error_display() {
        let e = QASM3Error::UndefinedGate("foo".to_string());
        assert_eq!(format!("{}", e), "undefined gate: foo");

        let e = QASM3Error::QubitOutOfRange {
            reg: "q".to_string(),
            index: 5,
            size: 2,
        };
        assert_eq!(format!("{}", e), "qubit q[5] out of range (size 2)");
    }

    // ---------------------------------------------------------------
    // Full roundtrip: GHZ state
    // ---------------------------------------------------------------

    #[test]
    fn test_ghz_roundtrip() {
        let src = r#"
OPENQASM 3.0;
qubit[4] q;
bit[4] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();

        assert_eq!(prog.num_qubits, 4);
        assert_eq!(prog.num_cbits, 4);

        let gates = QASM3Parser::to_gates(&prog).unwrap();
        assert_eq!(gates.len(), 4); // h + 3x cx

        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[0].targets, vec![0]);

        for (i, gate) in gates[1..].iter().enumerate() {
            assert_eq!(gate.gate_type, GateType::CNOT);
            assert_eq!(gate.controls, vec![i]);
            assert_eq!(gate.targets, vec![i + 1]);
        }
    }

    // ---------------------------------------------------------------
    // Two-qubit rotation gates: Rxx, Ryy, Rzz
    // ---------------------------------------------------------------

    #[test]
    fn test_rxx_ryy_rzz_gates() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
rxx(pi/4) q[0], q[1];
ryy(pi/2) q[0], q[1];
rzz(pi) q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 3);

        // Rxx(pi/4)
        match &gates[0].gate_type {
            GateType::Rxx(angle) => {
                assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
            }
            other => panic!("expected Rxx, got {:?}", other),
        }
        assert_eq!(gates[0].targets, vec![0, 1]);
        assert!(gates[0].controls.is_empty());

        // Ryy(pi/2)
        match &gates[1].gate_type {
            GateType::Ryy(angle) => {
                assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
            }
            other => panic!("expected Ryy, got {:?}", other),
        }
        assert_eq!(gates[1].targets, vec![0, 1]);

        // Rzz(pi)
        match &gates[2].gate_type {
            GateType::Rzz(angle) => {
                assert!((angle - std::f64::consts::PI).abs() < 1e-10);
            }
            other => panic!("expected Rzz, got {:?}", other),
        }
        assert_eq!(gates[2].targets, vec![0, 1]);
    }

    // ---------------------------------------------------------------
    // CSWAP (Fredkin gate)
    // ---------------------------------------------------------------

    #[test]
    fn test_cswap_gate() {
        let src = r#"
OPENQASM 3.0;
qubit[3] q;
cswap q[0], q[1], q[2];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::CSWAP);
        assert_eq!(gates[0].controls, vec![0]);
        assert_eq!(gates[0].targets, vec![1, 2]);
    }

    #[test]
    fn test_fredkin_alias() {
        let src = r#"
OPENQASM 3.0;
qubit[3] q;
fredkin q[0], q[1], q[2];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::CSWAP);
    }

    // ---------------------------------------------------------------
    // CU (generic controlled-U) gate
    // ---------------------------------------------------------------

    #[test]
    fn test_cu_gate() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
cu(pi/2, 0, pi, pi/4) q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let gates = QASM3Parser::to_gates(&prog).unwrap();

        assert_eq!(gates.len(), 1);
        match &gates[0].gate_type {
            GateType::CU {
                theta,
                phi,
                lambda,
                gamma,
            } => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
                assert!(phi.abs() < 1e-10);
                assert!((lambda - std::f64::consts::PI).abs() < 1e-10);
                assert!((gamma - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
            }
            other => panic!("expected CU, got {:?}", other),
        }
        assert_eq!(gates[0].controls, vec![0]);
        assert_eq!(gates[0].targets, vec![1]);
    }

    #[test]
    fn test_cu_missing_param_error() {
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
cu(pi/2, 0, pi) q[0], q[1];
"#;
        let mut parser = QASM3Parser::new();
        let prog = parser.parse(src).unwrap();
        let result = QASM3Parser::to_gates(&prog);
        assert!(result.is_err(), "cu with 3 params should fail (needs 4)");
    }

    // ---------------------------------------------------------------
    // Rxx/Ryy/Rzz matrix correctness
    // ---------------------------------------------------------------

    #[test]
    fn test_rxx_matrix_identity_at_zero() {
        // Rxx(0) should be the 4x4 identity
        let m = GateType::Rxx(0.0).matrix();
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j {
                    C64::new(1.0, 0.0)
                } else {
                    C64::new(0.0, 0.0)
                };
                assert!(
                    (m[i][j] - expected).norm() < 1e-12,
                    "Rxx(0) [{},{}]: got {:?}, expected {:?}",
                    i,
                    j,
                    m[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rzz_matrix_diagonal() {
        // Rzz(theta) should be diagonal: diag(e^{-i*t/2}, e^{i*t/2}, e^{i*t/2}, e^{-i*t/2})
        let theta = std::f64::consts::FRAC_PI_2;
        let m = GateType::Rzz(theta).matrix();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // Check diagonal entries
        assert!((m[0][0] - C64::new(cos, -sin)).norm() < 1e-12);
        assert!((m[1][1] - C64::new(cos, sin)).norm() < 1e-12);
        assert!((m[2][2] - C64::new(cos, sin)).norm() < 1e-12);
        assert!((m[3][3] - C64::new(cos, -sin)).norm() < 1e-12);

        // Check off-diagonal entries are zero
        assert!((m[0][1]).norm() < 1e-12);
        assert!((m[0][2]).norm() < 1e-12);
        assert!((m[1][0]).norm() < 1e-12);
    }

    #[test]
    fn test_cswap_matrix_structure() {
        // CSWAP should be 8x8, identity on first 5 diag, then swap |101> <-> |110>
        let m = GateType::CSWAP.matrix();
        assert_eq!(m.len(), 8);
        assert_eq!(m[0].len(), 8);

        // Diagonal entries (should be 1 except rows 5,6)
        for i in [0, 1, 2, 3, 4, 7] {
            assert!(
                (m[i][i] - C64::new(1.0, 0.0)).norm() < 1e-12,
                "CSWAP [{},{}] should be 1",
                i,
                i
            );
        }
        // Swap entries
        assert!((m[5][5]).norm() < 1e-12, "CSWAP [5,5] should be 0");
        assert!((m[6][6]).norm() < 1e-12, "CSWAP [6,6] should be 0");
        assert!(
            (m[5][6] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "CSWAP [5,6] should be 1"
        );
        assert!(
            (m[6][5] - C64::new(1.0, 0.0)).norm() < 1e-12,
            "CSWAP [6,5] should be 1"
        );
    }

    #[test]
    fn test_cu_reduces_to_cnot_case() {
        // CU(pi, 0, pi, 0) should act like CNOT (up to global phase)
        // CNOT has matrix: diag(I, X) = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
        // CU(pi, 0, pi, 0) = |0><0| x I + |1><1| x U(pi, 0, pi)
        // U(pi, 0, pi) = [[0, -1], [1, 0]] = -iY = X (up to global phase)
        let m = GateType::CU {
            theta: std::f64::consts::PI,
            phi: 0.0,
            lambda: std::f64::consts::PI,
            gamma: 0.0,
        }
        .matrix();

        // |00> -> |00>, |01> -> |01> (identity in top-left 2x2)
        assert!((m[0][0] - C64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((m[1][1] - C64::new(1.0, 0.0)).norm() < 1e-10);

        // |10> -> |11>, |11> -> |10> (should swap with possible phase)
        // cos(pi/2) = 0, sin(pi/2) = 1
        // m[2][2] = e^(i*0) * cos(pi/2) = 0
        // m[2][3] = -e^(i*(0+pi)) * sin(pi/2) = -e^(i*pi) = 1
        // m[3][2] = e^(i*(0+0)) * sin(pi/2) = 1
        // m[3][3] = e^(i*(0+0+pi)) * cos(pi/2) = 0
        assert!(m[2][2].norm() < 1e-10, "CU(pi,0,pi,0) [2,2] should be ~0");
        assert!(
            (m[2][3] - C64::new(1.0, 0.0)).norm() < 1e-10,
            "CU(pi,0,pi,0) [2,3] should be ~1"
        );
        assert!(
            (m[3][2] - C64::new(1.0, 0.0)).norm() < 1e-10,
            "CU(pi,0,pi,0) [3,2] should be ~1"
        );
        assert!(m[3][3].norm() < 1e-10, "CU(pi,0,pi,0) [3,3] should be ~0");
    }
}

// ===========================================================================
// QASM3 Executor (Control Flow Support)
// ===========================================================================

/// Execution engine for OpenQASM 3.0 programs with control flow.
pub struct QASM3Executor {
    simulator: QuantumSimulator,
    /// Maps register name -> start index in state vector
    qubit_offsets: HashMap<String, usize>,
    /// Maps classical register name -> (start_bit_index, size)
    creg_offsets: HashMap<String, (usize, usize)>,
    /// Classical memory (bit array)
    creg_memory: Vec<bool>,
}

impl QASM3Executor {
    /// Create a new executor for a parsed program.
    pub fn new(program: &QASM3Program) -> Self {
        let mut qubit_offsets = HashMap::new();
        let mut q_offset = 0;

        let mut creg_offsets = HashMap::new();
        let mut c_offset = 0;

        // Allocation pass
        for stmt in &program.statements {
            match stmt {
                QASM3Statement::QubitDecl { name, size } => {
                    qubit_offsets.insert(name.clone(), q_offset);
                    q_offset += size;
                }
                QASM3Statement::CbitDecl { name, size } => {
                    creg_offsets.insert(name.clone(), (c_offset, *size));
                    c_offset += size;
                }
                _ => {}
            }
        }

        Self {
            simulator: QuantumSimulator::new(q_offset),
            qubit_offsets,
            creg_offsets,
            creg_memory: vec![false; c_offset],
        }
    }

    /// Run the program.
    pub fn run(&mut self, program: &QASM3Program) -> Result<(), String> {
        self.execute_block(&program.statements)
    }

    fn execute_block(&mut self, statements: &[QASM3Statement]) -> Result<(), String> {
        for stmt in statements {
            self.execute_statement(stmt)?;
        }
        Ok(())
    }

    fn execute_statement(&mut self, stmt: &QASM3Statement) -> Result<(), String> {
        match stmt {
            QASM3Statement::GateCall {
                name,
                params,
                qubits,
            } => {
                // Resolve qubits
                let mut resolved_qubits = Vec::new();
                for q in qubits {
                    let offset = *self
                        .qubit_offsets
                        .get(&q.reg)
                        .ok_or_else(|| format!("Unknown qubit register: {}", q.reg))?;
                    resolved_qubits.push(offset + q.index);
                }

                // Apply gate
                match name.as_str() {
                    "h" => self.simulator.h(resolved_qubits[0]),
                    "x" => self.simulator.x(resolved_qubits[0]),
                    "y" => self.simulator.y(resolved_qubits[0]),
                    "z" => self.simulator.z(resolved_qubits[0]),
                    "cx" | "cnot" => self.simulator.cnot(resolved_qubits[0], resolved_qubits[1]),
                    "rx" => self
                        .simulator
                        .rx(resolved_qubits[0], params.get(0).copied().unwrap_or(0.0)),
                    "ry" => self
                        .simulator
                        .ry(resolved_qubits[0], params.get(0).copied().unwrap_or(0.0)),
                    "rz" => self
                        .simulator
                        .rz(resolved_qubits[0], params.get(0).copied().unwrap_or(0.0)),
                    "measure" => { /* No-op, handled in assignment */ }
                    _ => return Err(format!("Unsupported gate in executor: {}", name)),
                }
            }
            QASM3Statement::Measure { qubit, cbit } => {
                let q_offset = *self
                    .qubit_offsets
                    .get(&qubit.reg)
                    .ok_or_else(|| format!("Unknown qubit register: {}", qubit.reg))?;
                let q_idx = q_offset + qubit.index;

                let result = self.simulator.measure_qubit(q_idx).0;

                let c_offset_info = self
                    .creg_offsets
                    .get(&cbit.reg)
                    .ok_or_else(|| format!("Unknown classical register: {}", cbit.reg))?;

                let bit_idx = c_offset_info.0 + cbit.index;
                if bit_idx < self.creg_memory.len() {
                    self.creg_memory[bit_idx] = result == 1;
                }
            }
            QASM3Statement::If { condition, body } => {
                if self.check_condition(condition)? {
                    self.execute_block(body)?;
                }
            }
            QASM3Statement::While { condition, body } => {
                while self.check_condition(condition)? {
                    self.execute_block(body)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn check_condition(&self, cond: &ClassicalCondition) -> Result<bool, String> {
        let (start, size) = *self
            .creg_offsets
            .get(&cond.register)
            .ok_or_else(|| format!("Unknown register: {}", cond.register))?;

        let mut val = 0u64;
        for i in 0..size {
            if self.creg_memory[start + i] {
                val |= 1 << i;
            }
        }

        Ok(val == cond.value)
    }
}
