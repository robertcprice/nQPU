//! QASM Language Server Protocol (LSP) Implementation
//!
//! Provides IDE integration for OpenQASM 2.0 and 3.0 programs via the Language
//! Server Protocol. This module implements the core LSP capabilities — diagnostics,
//! completions, hover, go-to-definition, and simulation-aware analysis — by
//! reusing the existing QASM parsers in [`crate::qasm`] and [`crate::qasm3`].
//!
//! # Features
//!
//! - **Diagnostics**: Syntax validation with line/column error locations
//! - **Completion**: Context-aware auto-complete for gates, keywords, and snippets
//! - **Hover**: Documentation on gates (matrix representations, arity) and keywords
//! - **Go-to-Definition**: Jump to custom `gate` definitions
//! - **Simulation-Aware Diagnostics**: Memory and optimization hints for simulation
//! - **Incremental Sync**: Efficient incremental document synchronization (mode 2)
//!
//! # Building
//!
//! Requires the `lsp` feature flag:
//!
//! ```bash
//! cargo build --release --features lsp
//! ```
//!
//! # Architecture
//!
//! The [`QasmLanguageServer`] struct is the central entry point. It does NOT depend
//! on any LSP framework crate — it provides pure-data request/response types that
//! the binary in `src/bin/nqpu_qasm_lsp.rs` wraps with JSON-RPC transport.

#[cfg(feature = "lsp")]
pub use self::qasm_lsp_impl::*;

#[cfg(feature = "lsp")]
mod qasm_lsp_impl {
    use crate::qasm::parse_qasm;
    use crate::qasm3::{QASM3Error, QASM3Parser};

    // ================================================================
    // Public types
    // ================================================================

    /// Diagnostic severity levels, mirroring the LSP specification.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum DiagnosticSeverity {
        Error,
        Warning,
        Information,
        Hint,
    }

    /// A diagnostic (error/warning) attached to a region of QASM source.
    #[derive(Clone, Debug)]
    pub struct QasmDiagnostic {
        /// 0-based line number.
        pub line: usize,
        /// 0-based start column.
        pub col: usize,
        /// 0-based end column (exclusive).
        pub end_col: usize,
        /// Severity of the diagnostic.
        pub severity: DiagnosticSeverity,
        /// Human-readable message.
        pub message: String,
        /// Optional machine-readable diagnostic code.
        pub code: Option<String>,
    }

    /// Kind of completion item, used for icon selection in editors.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum CompletionKind {
        Gate,
        Keyword,
        Type,
        Snippet,
    }

    /// A single auto-completion suggestion.
    #[derive(Clone, Debug)]
    pub struct CompletionItem {
        /// Short label displayed in the completion list.
        pub label: String,
        /// Additional detail shown beside the label.
        pub detail: String,
        /// Text to insert when the item is accepted.
        pub insert_text: String,
        /// Completion kind for categorization.
        pub kind: CompletionKind,
    }

    /// Hover information for a symbol at a given position.
    #[derive(Clone, Debug)]
    pub struct HoverInfo {
        /// Markdown-formatted documentation string.
        pub contents: String,
        /// Optional range `(start_line, start_col, end_line, end_col)`.
        pub range: Option<(usize, usize, usize, usize)>,
    }

    /// A location in a document (for go-to-definition).
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct DefinitionLocation {
        /// 0-based line number of the definition.
        pub line: usize,
        /// 0-based start column.
        pub col: usize,
        /// 0-based end column.
        pub end_col: usize,
    }

    /// A stored custom gate definition location for go-to-definition.
    #[derive(Clone, Debug)]
    struct GateDefLocation {
        /// The name of the gate.
        name: String,
        /// 0-based line number.
        line: usize,
        /// 0-based start column of the gate name within the line.
        col: usize,
        /// 0-based end column.
        end_col: usize,
    }

    // ================================================================
    // QasmLanguageServer
    // ================================================================

    /// Core QASM Language Server logic.
    ///
    /// Stateless — each method takes the document source as input and returns
    /// results without requiring incremental updates. This keeps the
    /// implementation simple and deterministic.
    pub struct QasmLanguageServer {
        _private: (),
    }

    impl QasmLanguageServer {
        /// Create a new language server instance.
        pub fn new() -> Self {
            Self { _private: () }
        }

        // ------------------------------------------------------------
        // Diagnostics
        // ------------------------------------------------------------

        /// Validate QASM source and return diagnostics.
        ///
        /// Attempts QASM 3.0 parsing first and falls back to QASM 2.0. Returns
        /// diagnostics for syntax errors, undefined gates, undeclared registers,
        /// and out-of-range qubit indices. Also includes simulation-aware
        /// information diagnostics.
        pub fn validate(&self, source: &str) -> Vec<QasmDiagnostic> {
            let mut diagnostics = Vec::new();

            // ---- QASM 3.0 parse attempt ----
            let mut parser = QASM3Parser::new();
            match parser.parse(source) {
                Ok(program) => {
                    // Parse succeeded — try lowering to gates which catches
                    // undefined gates, wrong arity, etc.
                    if let Err(e) = QASM3Parser::to_gates(&program) {
                        let (line, msg) = Self::error_to_line_message(source, &e);
                        diagnostics.push(QasmDiagnostic {
                            line,
                            col: 0,
                            end_col: Self::line_length(source, line),
                            severity: DiagnosticSeverity::Error,
                            message: msg,
                            code: Some(Self::error_code(&e)),
                        });
                    }

                    // Additional validation: qubit index bounds checking.
                    Self::check_qubit_bounds(source, &program, &mut diagnostics);

                    // Simulation-aware diagnostics (QASM 3.0 path).
                    Self::check_simulation_resources(source, &mut diagnostics);
                    Self::check_gate_fusion_opportunities(source, &mut diagnostics);
                }
                Err(e) => {
                    // QASM 3.0 parse failed — try QASM 2.0 as fallback.
                    match parse_qasm(source) {
                        Ok(_circuit) => {
                            // QASM 2.0 parsed OK — emit a hint about the 3.0 error
                            // in case the user intended 3.0 syntax.
                            let (line, msg) = Self::qasm3_error_to_line_message(source, &e);
                            diagnostics.push(QasmDiagnostic {
                                line,
                                col: 0,
                                end_col: Self::line_length(source, line),
                                severity: DiagnosticSeverity::Information,
                                message: format!(
                                    "Parsed as QASM 2.0 (QASM 3.0 parse note: {})",
                                    msg
                                ),
                                code: Some("qasm2-fallback".to_string()),
                            });

                            // QASM 2.0 register bounds checking.
                            Self::check_qasm2_register_bounds(source, &mut diagnostics);

                            // Simulation-aware diagnostics (QASM 2.0 path).
                            Self::check_simulation_resources(source, &mut diagnostics);
                            Self::check_gate_fusion_opportunities(source, &mut diagnostics);
                        }
                        Err(qasm2_err) => {
                            // Both parsers failed — report the more informative
                            // QASM 3.0 error as the primary diagnostic.
                            let (line, msg) = Self::qasm3_error_to_line_message(source, &e);
                            diagnostics.push(QasmDiagnostic {
                                line,
                                col: 0,
                                end_col: Self::line_length(source, line),
                                severity: DiagnosticSeverity::Error,
                                message: msg,
                                code: Some(Self::error_code(&e)),
                            });

                            // Also include the 2.0 error as secondary info.
                            diagnostics.push(QasmDiagnostic {
                                line: 0,
                                col: 0,
                                end_col: 0,
                                severity: DiagnosticSeverity::Information,
                                message: format!("QASM 2.0 parse also failed: {}", qasm2_err),
                                code: Some("qasm2-error".to_string()),
                            });
                        }
                    }
                }
            }

            // ---- Version header check ----
            Self::check_version_header(source, &mut diagnostics);

            diagnostics
        }

        // ------------------------------------------------------------
        // Completions
        // ------------------------------------------------------------

        /// Return completion items appropriate for the cursor position.
        ///
        /// Context rules:
        /// - After `OPENQASM`: version numbers
        /// - Statement-level (beginning of line): gates + keywords
        /// - After a gate name: qubit register references extracted from source
        pub fn complete(&self, source: &str, line: usize, col: usize) -> Vec<CompletionItem> {
            let current_line = source.lines().nth(line).unwrap_or("");
            let prefix = if col <= current_line.len() {
                &current_line[..col]
            } else {
                current_line
            };
            let prefix_trimmed = prefix.trim();

            // After OPENQASM keyword — suggest version numbers.
            if prefix_trimmed.starts_with("OPENQASM") {
                return vec![
                    CompletionItem {
                        label: "2.0".to_string(),
                        detail: "OpenQASM 2.0".to_string(),
                        insert_text: "2.0;".to_string(),
                        kind: CompletionKind::Keyword,
                    },
                    CompletionItem {
                        label: "3.0".to_string(),
                        detail: "OpenQASM 3.0".to_string(),
                        insert_text: "3.0;".to_string(),
                        kind: CompletionKind::Keyword,
                    },
                ];
            }

            // If the line already has a gate name and space, suggest qubit refs.
            if Self::cursor_after_gate_name(prefix) {
                return Self::qubit_ref_completions(source);
            }

            // Default: provide all gates, keywords, and snippets.
            let mut items = Vec::new();
            items.extend(Self::standard_gate_completions());
            items.extend(Self::keyword_completions());
            items.extend(Self::snippet_completions());

            // Filter by partial typing.
            let word = Self::word_at_cursor(prefix);
            if !word.is_empty() {
                items.retain(|item| {
                    item.label.starts_with(word) || item.insert_text.starts_with(word)
                });
            }

            items
        }

        // ------------------------------------------------------------
        // Hover
        // ------------------------------------------------------------

        /// Return hover documentation for the symbol at the given position.
        pub fn hover(&self, source: &str, line: usize, col: usize) -> Option<HoverInfo> {
            let current_line = source.lines().nth(line)?;
            let word = Self::word_around(current_line, col)?;

            // Gate documentation.
            if let Some(doc) = Self::gate_docs(&word) {
                return Some(HoverInfo {
                    contents: doc,
                    range: Some((
                        line,
                        col.saturating_sub(word.len()),
                        line,
                        col.saturating_sub(word.len()) + word.len(),
                    )),
                });
            }

            // Keyword documentation.
            if let Some(doc) = Self::keyword_docs(&word) {
                return Some(HoverInfo {
                    contents: doc,
                    range: Some((
                        line,
                        col.saturating_sub(word.len()),
                        line,
                        col.saturating_sub(word.len()) + word.len(),
                    )),
                });
            }

            None
        }

        // ------------------------------------------------------------
        // Go-to-Definition
        // ------------------------------------------------------------

        /// Return the definition location for a gate name at the given position.
        ///
        /// Looks for custom `gate` definitions in the source. If the word under
        /// the cursor matches a `gate foo(...)` definition earlier in the file,
        /// returns the location of that definition line.
        pub fn goto_definition(
            &self,
            source: &str,
            line: usize,
            col: usize,
        ) -> Option<DefinitionLocation> {
            let current_line = source.lines().nth(line)?;
            let word = Self::word_around(current_line, col)?;

            // Find all custom gate definitions.
            let defs = Self::find_gate_definitions(source);

            // Look for one matching the word under the cursor.
            for def in &defs {
                if def.name == word {
                    return Some(DefinitionLocation {
                        line: def.line,
                        col: def.col,
                        end_col: def.end_col,
                    });
                }
            }

            None
        }

        // ------------------------------------------------------------
        // Incremental Document Sync
        // ------------------------------------------------------------

        /// Apply an incremental text edit to a document.
        ///
        /// Takes the current document content and a range-based edit, returns
        /// the updated content. The range is specified as
        /// `(start_line, start_col, end_line, end_col)` with 0-based indices.
        pub fn apply_incremental_edit(
            document: &str,
            start_line: usize,
            start_col: usize,
            end_line: usize,
            end_col: usize,
            new_text: &str,
        ) -> String {
            let lines: Vec<&str> = document.lines().collect();
            let mut result = String::with_capacity(document.len());

            // Compute the byte offset of (line, col) in the document.
            let start_offset = Self::position_to_offset(&lines, start_line, start_col);
            let end_offset = Self::position_to_offset(&lines, end_line, end_col);

            // Rebuild the document: prefix + new_text + suffix.
            // We need to be careful with line endings. The original document
            // uses \n between lines (we reconstruct from lines() which strips).
            let full_text = Self::lines_to_string(&lines, document);
            result.push_str(&full_text[..start_offset]);
            result.push_str(new_text);
            if end_offset <= full_text.len() {
                result.push_str(&full_text[end_offset..]);
            }

            result
        }

        /// Convert a (line, col) position to a byte offset in the reconstructed text.
        fn position_to_offset(lines: &[&str], line: usize, col: usize) -> usize {
            let mut offset = 0;
            for (i, l) in lines.iter().enumerate() {
                if i == line {
                    return offset + col.min(l.len());
                }
                offset += l.len() + 1; // +1 for newline
            }
            // Past end of document.
            offset
        }

        /// Reconstruct the full text from lines, preserving a trailing newline
        /// if the original had one.
        fn lines_to_string(lines: &[&str], original: &str) -> String {
            let mut s = lines.join("\n");
            if original.ends_with('\n') && !s.ends_with('\n') {
                s.push('\n');
            }
            s
        }

        // ================================================================
        // Internal helpers — go-to-definition
        // ================================================================

        /// Scan source for custom `gate` definitions and return their locations.
        fn find_gate_definitions(source: &str) -> Vec<GateDefLocation> {
            let mut defs = Vec::new();

            for (line_num, line) in source.lines().enumerate() {
                let trimmed = line.trim();
                // Match `gate <name>` pattern (QASM 2.0 and 3.0).
                if let Some(rest) = trimmed.strip_prefix("gate") {
                    let rest = rest.trim_start();
                    // Extract the gate name (identifier before '(' or whitespace).
                    let name_end = rest
                        .find(|c: char| !c.is_alphanumeric() && c != '_')
                        .unwrap_or(rest.len());
                    if name_end > 0 {
                        let name = &rest[..name_end];
                        // Find the column of the gate name in the original line.
                        if let Some(name_start_in_line) = line.find(name) {
                            // Make sure we're finding the name AFTER "gate", not "gate" itself.
                            let gate_keyword_end = line.find("gate").unwrap_or(0) + 4;
                            let col = if name_start_in_line >= gate_keyword_end {
                                name_start_in_line
                            } else {
                                // Fallback: search after "gate " prefix.
                                line[gate_keyword_end..]
                                    .find(name)
                                    .map(|pos| gate_keyword_end + pos)
                                    .unwrap_or(name_start_in_line)
                            };
                            defs.push(GateDefLocation {
                                name: name.to_string(),
                                line: line_num,
                                col,
                                end_col: col + name_end,
                            });
                        }
                    }
                }
            }

            defs
        }

        // ================================================================
        // Internal helpers — completions
        // ================================================================

        /// All standard quantum gates from the OpenQASM standard library.
        fn standard_gate_completions() -> Vec<CompletionItem> {
            vec![
                Self::gate_item("h", "Hadamard gate", "h"),
                Self::gate_item("x", "Pauli-X (NOT) gate", "x"),
                Self::gate_item("y", "Pauli-Y gate", "y"),
                Self::gate_item("z", "Pauli-Z gate", "z"),
                Self::gate_item("s", "S (phase) gate", "s"),
                Self::gate_item("t", "T (pi/8) gate", "t"),
                Self::gate_item("sx", "Sqrt-X gate", "sx"),
                Self::gate_item("cx", "Controlled-X (CNOT) gate", "cx"),
                Self::gate_item("cz", "Controlled-Z gate", "cz"),
                Self::gate_item("swap", "SWAP gate", "swap"),
                Self::gate_item("iswap", "iSWAP gate", "iswap"),
                Self::gate_item("ccx", "Toffoli (CCX) gate", "ccx"),
                Self::gate_item("ccz", "Controlled-CZ gate", "ccz"),
                Self::gate_item("cswap", "Fredkin (CSWAP) gate", "cswap"),
                CompletionItem {
                    label: "rx".to_string(),
                    detail: "X-rotation gate".to_string(),
                    insert_text: "rx(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "ry".to_string(),
                    detail: "Y-rotation gate".to_string(),
                    insert_text: "ry(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "rz".to_string(),
                    detail: "Z-rotation gate".to_string(),
                    insert_text: "rz(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "rxx".to_string(),
                    detail: "XX-rotation gate".to_string(),
                    insert_text: "rxx(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "ryy".to_string(),
                    detail: "YY-rotation gate".to_string(),
                    insert_text: "ryy(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "rzz".to_string(),
                    detail: "ZZ-rotation gate".to_string(),
                    insert_text: "rzz(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "crx".to_string(),
                    detail: "Controlled RX gate".to_string(),
                    insert_text: "crx(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "cry".to_string(),
                    detail: "Controlled RY gate".to_string(),
                    insert_text: "cry(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "crz".to_string(),
                    detail: "Controlled RZ gate".to_string(),
                    insert_text: "crz(${1:angle})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "u".to_string(),
                    detail: "Generic U(theta,phi,lambda) gate".to_string(),
                    insert_text: "u(${1:theta}, ${2:phi}, ${3:lambda})".to_string(),
                    kind: CompletionKind::Gate,
                },
                CompletionItem {
                    label: "cu".to_string(),
                    detail: "Controlled-U gate".to_string(),
                    insert_text: "cu(${1:theta}, ${2:phi}, ${3:lambda}, ${4:gamma})".to_string(),
                    kind: CompletionKind::Gate,
                },
                Self::gate_item("measure", "Measurement", "measure"),
                Self::gate_item("reset", "Qubit reset to |0>", "reset"),
                Self::gate_item("barrier", "Optimization barrier", "barrier"),
            ]
        }

        /// QASM keywords.
        fn keyword_completions() -> Vec<CompletionItem> {
            vec![
                CompletionItem {
                    label: "OPENQASM".to_string(),
                    detail: "Version declaration".to_string(),
                    insert_text: "OPENQASM".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "include".to_string(),
                    detail: "Include file".to_string(),
                    insert_text: "include \"${1:qelib1.inc}\";".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "qubit".to_string(),
                    detail: "QASM 3.0 qubit declaration".to_string(),
                    insert_text: "qubit[${1:n}] ${2:q};".to_string(),
                    kind: CompletionKind::Type,
                },
                CompletionItem {
                    label: "bit".to_string(),
                    detail: "QASM 3.0 classical bit declaration".to_string(),
                    insert_text: "bit[${1:n}] ${2:c};".to_string(),
                    kind: CompletionKind::Type,
                },
                CompletionItem {
                    label: "qreg".to_string(),
                    detail: "QASM 2.0 qubit register".to_string(),
                    insert_text: "qreg ${1:q}[${2:n}];".to_string(),
                    kind: CompletionKind::Type,
                },
                CompletionItem {
                    label: "creg".to_string(),
                    detail: "QASM 2.0 classical register".to_string(),
                    insert_text: "creg ${1:c}[${2:n}];".to_string(),
                    kind: CompletionKind::Type,
                },
                CompletionItem {
                    label: "gate".to_string(),
                    detail: "Custom gate definition".to_string(),
                    insert_text: "gate ${1:name} ${2:q0} {\n  ${3}\n}".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "if".to_string(),
                    detail: "Conditional execution".to_string(),
                    insert_text: "if (${1:c} == ${2:val}) {\n  ${3}\n}".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "while".to_string(),
                    detail: "While loop".to_string(),
                    insert_text: "while (${1:c} == ${2:val}) {\n  ${3}\n}".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "for".to_string(),
                    detail: "For loop".to_string(),
                    insert_text: "for uint ${1:i} in [${2:0}:${3:n}] {\n  ${4}\n}".to_string(),
                    kind: CompletionKind::Keyword,
                },
                CompletionItem {
                    label: "def".to_string(),
                    detail: "Subroutine definition".to_string(),
                    insert_text: "def ${1:name}(${2:params}) qubit ${3:q} {\n  ${4}\n}".to_string(),
                    kind: CompletionKind::Keyword,
                },
            ]
        }

        /// Circuit template snippets.
        fn snippet_completions() -> Vec<CompletionItem> {
            vec![
                CompletionItem {
                    label: "bell-pair".to_string(),
                    detail: "Bell state template".to_string(),
                    insert_text: concat!(
                        "// Bell pair\n",
                        "qubit[2] q;\n",
                        "bit[2] c;\n",
                        "h q[0];\n",
                        "cx q[0], q[1];\n",
                        "c[0] = measure q[0];\n",
                        "c[1] = measure q[1];\n",
                    )
                    .to_string(),
                    kind: CompletionKind::Snippet,
                },
                CompletionItem {
                    label: "ghz-state".to_string(),
                    detail: "GHZ state template".to_string(),
                    insert_text: concat!(
                        "// GHZ state\n",
                        "qubit[3] q;\n",
                        "bit[3] c;\n",
                        "h q[0];\n",
                        "cx q[0], q[1];\n",
                        "cx q[1], q[2];\n",
                        "c[0] = measure q[0];\n",
                        "c[1] = measure q[1];\n",
                        "c[2] = measure q[2];\n",
                    )
                    .to_string(),
                    kind: CompletionKind::Snippet,
                },
                CompletionItem {
                    label: "qft-2qubit".to_string(),
                    detail: "2-qubit QFT template".to_string(),
                    insert_text: concat!(
                        "// 2-qubit Quantum Fourier Transform\n",
                        "qubit[2] q;\n",
                        "h q[0];\n",
                        "cp(pi/2) q[1], q[0];\n",
                        "h q[1];\n",
                        "swap q[0], q[1];\n",
                    )
                    .to_string(),
                    kind: CompletionKind::Snippet,
                },
            ]
        }

        /// Build qubit reference completions by scanning source for register declarations.
        fn qubit_ref_completions(source: &str) -> Vec<CompletionItem> {
            let mut items = Vec::new();

            for line in source.lines() {
                let trimmed = line.trim();

                // QASM 3.0: qubit[N] name;
                if trimmed.starts_with("qubit[") || trimmed.starts_with("qubit [") {
                    if let Some((size, name)) = Self::extract_qasm3_decl(trimmed, "qubit") {
                        for i in 0..size {
                            items.push(CompletionItem {
                                label: format!("{}[{}]", name, i),
                                detail: format!("Qubit {} of register {}", i, name),
                                insert_text: format!("{}[{}]", name, i),
                                kind: CompletionKind::Type,
                            });
                        }
                    }
                }

                // QASM 2.0: qreg name[N];
                if trimmed.starts_with("qreg") {
                    if let Some((name, size)) = Self::extract_qasm2_reg(trimmed, "qreg") {
                        for i in 0..size {
                            items.push(CompletionItem {
                                label: format!("{}[{}]", name, i),
                                detail: format!("Qubit {} of register {}", i, name),
                                insert_text: format!("{}[{}]", name, i),
                                kind: CompletionKind::Type,
                            });
                        }
                    }
                }
            }

            items
        }

        // ================================================================
        // Internal helpers — hover
        // ================================================================

        /// Gate documentation for hover display.
        fn gate_docs(gate_name: &str) -> Option<String> {
            let doc = match gate_name {
                "h" | "H" => concat!(
                    "**H (Hadamard)**: Single-qubit gate.\n\n",
                    "Matrix: `1/sqrt(2) * [[1, 1], [1, -1]]`\n\n",
                    "Creates equal superposition: `|0> -> (|0> + |1>)/sqrt(2)`\n\n",
                    "Arity: 1 qubit",
                ),
                "x" | "X" => concat!(
                    "**X (Pauli-X / NOT)**: Single-qubit gate.\n\n",
                    "Matrix: `[[0, 1], [1, 0]]`\n\n",
                    "Bit-flip: `|0> -> |1>`, `|1> -> |0>`\n\n",
                    "Arity: 1 qubit",
                ),
                "y" | "Y" => concat!(
                    "**Y (Pauli-Y)**: Single-qubit gate.\n\n",
                    "Matrix: `[[0, -i], [i, 0]]`\n\n",
                    "Arity: 1 qubit",
                ),
                "z" | "Z" => concat!(
                    "**Z (Pauli-Z)**: Single-qubit gate.\n\n",
                    "Matrix: `[[1, 0], [0, -1]]`\n\n",
                    "Phase-flip: `|1> -> -|1>`\n\n",
                    "Arity: 1 qubit",
                ),
                "s" | "S" => concat!(
                    "**S (Phase)**: Single-qubit gate.\n\n",
                    "Matrix: `[[1, 0], [0, i]]`\n\n",
                    "Quarter-turn around Z axis. S^2 = Z.\n\n",
                    "Arity: 1 qubit",
                ),
                "t" | "T" => concat!(
                    "**T (pi/8)**: Single-qubit gate.\n\n",
                    "Matrix: `[[1, 0], [0, e^(i*pi/4)]]`\n\n",
                    "Eighth-turn around Z axis. T^2 = S.\n\n",
                    "Arity: 1 qubit",
                ),
                "sx" | "SX" => concat!(
                    "**SX (Sqrt-X)**: Single-qubit gate.\n\n",
                    "SX^2 = X. Native gate on many IBM devices.\n\n",
                    "Arity: 1 qubit",
                ),
                "cx" | "CX" | "cnot" | "CNOT" => concat!(
                    "**CX (CNOT)**: Two-qubit controlled-NOT gate.\n\n",
                    "Flips the target qubit if the control is |1>.\n\n",
                    "Matrix (4x4): `diag(I, X)`\n\n",
                    "Arity: 2 qubits (control, target)",
                ),
                "cz" | "CZ" => concat!(
                    "**CZ (Controlled-Z)**: Two-qubit gate.\n\n",
                    "Applies Z to target if control is |1>. Symmetric.\n\n",
                    "Arity: 2 qubits",
                ),
                "swap" | "SWAP" => concat!(
                    "**SWAP**: Two-qubit gate.\n\n",
                    "Exchanges the states of two qubits.\n\n",
                    "SWAP = CX(a,b) CX(b,a) CX(a,b)\n\n",
                    "Arity: 2 qubits",
                ),
                "iswap" | "ISWAP" => concat!(
                    "**iSWAP**: Two-qubit gate.\n\n",
                    "Swaps and applies a phase. Native on superconducting devices.\n\n",
                    "Arity: 2 qubits",
                ),
                "rx" | "RX" => concat!(
                    "**RX(theta)**: X-axis rotation gate.\n\n",
                    "Matrix: `[[cos(t/2), -i*sin(t/2)], [-i*sin(t/2), cos(t/2)]]`\n\n",
                    "Arity: 1 qubit, 1 parameter (angle in radians)",
                ),
                "ry" | "RY" => concat!(
                    "**RY(theta)**: Y-axis rotation gate.\n\n",
                    "Matrix: `[[cos(t/2), -sin(t/2)], [sin(t/2), cos(t/2)]]`\n\n",
                    "Arity: 1 qubit, 1 parameter (angle in radians)",
                ),
                "rz" | "RZ" => concat!(
                    "**RZ(theta)**: Z-axis rotation gate.\n\n",
                    "Matrix: `[[e^(-it/2), 0], [0, e^(it/2)]]`\n\n",
                    "Arity: 1 qubit, 1 parameter (angle in radians)",
                ),
                "rxx" | "RXX" => concat!(
                    "**RXX(theta)**: XX-rotation (Ising coupling) gate.\n\n",
                    "exp(-i * theta/2 * X x X)\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "ryy" | "RYY" => concat!(
                    "**RYY(theta)**: YY-rotation (Ising coupling) gate.\n\n",
                    "exp(-i * theta/2 * Y x Y)\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "rzz" | "RZZ" => concat!(
                    "**RZZ(theta)**: ZZ-rotation (Ising coupling) gate.\n\n",
                    "exp(-i * theta/2 * Z x Z)\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "ccx" | "CCX" | "toffoli" | "Toffoli" => concat!(
                    "**CCX (Toffoli)**: Three-qubit gate.\n\n",
                    "Flips the target if both controls are |1>. Universal for classical computation.\n\n",
                    "Arity: 3 qubits (control1, control2, target)",
                ),
                "ccz" | "CCZ" => concat!(
                    "**CCZ (Controlled-CZ)**: Three-qubit gate.\n\n",
                    "Applies phase -1 to |111>.\n\n",
                    "Arity: 3 qubits",
                ),
                "cswap" | "CSWAP" | "fredkin" | "Fredkin" => concat!(
                    "**CSWAP (Fredkin)**: Three-qubit gate.\n\n",
                    "Swaps the two targets if the control is |1>.\n\n",
                    "Arity: 3 qubits (control, target1, target2)",
                ),
                "crx" | "CRX" => concat!(
                    "**CRX(theta)**: Controlled RX rotation.\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "cry" | "CRY" => concat!(
                    "**CRY(theta)**: Controlled RY rotation.\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "crz" | "CRZ" => concat!(
                    "**CRZ(theta)**: Controlled RZ rotation.\n\n",
                    "Arity: 2 qubits, 1 parameter",
                ),
                "u" | "u3" | "U" => concat!(
                    "**U(theta, phi, lambda)**: Generic single-qubit unitary.\n\n",
                    "Any single-qubit gate can be expressed as U(t, p, l).\n\n",
                    "Matrix: `[[cos(t/2), -e^(il)*sin(t/2)], [e^(ip)*sin(t/2), e^(i(p+l))*cos(t/2)]]`\n\n",
                    "Arity: 1 qubit, 3 parameters",
                ),
                "cu" | "CU" => concat!(
                    "**CU(theta, phi, lambda, gamma)**: Controlled-U gate.\n\n",
                    "Applies U(theta, phi, lambda) to target with global phase gamma when control is |1>.\n\n",
                    "Arity: 2 qubits, 4 parameters",
                ),
                "measure" => concat!(
                    "**measure**: Measurement in the computational basis.\n\n",
                    "QASM 2.0: `measure q[0] -> c[0];`\n",
                    "QASM 3.0: `c[0] = measure q[0];`\n\n",
                    "Collapses the qubit to |0> or |1> and stores the result.",
                ),
                "reset" => concat!(
                    "**reset**: Reset qubit to |0> state.\n\n",
                    "Syntax: `reset q[0];`\n\n",
                    "Non-unitary operation that projects the qubit to |0>.",
                ),
                "barrier" => concat!(
                    "**barrier**: Optimization barrier.\n\n",
                    "Syntax: `barrier q[0], q[1];`\n\n",
                    "Prevents the compiler from optimizing across this boundary.",
                ),
                _ => return None,
            };
            Some(doc.to_string())
        }

        /// Keyword documentation for hover display.
        fn keyword_docs(keyword: &str) -> Option<String> {
            let doc = match keyword {
                "OPENQASM" => concat!(
                    "**OPENQASM**: Version declaration header.\n\n",
                    "Must be the first non-comment line.\n\n",
                    "QASM 2.0: `OPENQASM 2.0;`\n",
                    "QASM 3.0: `OPENQASM 3.0;`",
                ),
                "include" => concat!(
                    "**include**: Include an external QASM library.\n\n",
                    "Standard: `include \"qelib1.inc\";`\n\n",
                    "Provides the standard gate library (h, cx, etc.).",
                ),
                "qubit" => concat!(
                    "**qubit**: QASM 3.0 qubit register declaration.\n\n",
                    "Syntax: `qubit[N] name;`\n\n",
                    "Declares N qubits, initialized to |0>.",
                ),
                "bit" => concat!(
                    "**bit**: QASM 3.0 classical bit declaration.\n\n",
                    "Syntax: `bit[N] name;`\n\n",
                    "Declares N classical bits for measurement results.",
                ),
                "qreg" => concat!(
                    "**qreg**: QASM 2.0 qubit register declaration.\n\n",
                    "Syntax: `qreg name[N];`\n\n",
                    "Declares N qubits, initialized to |0>.",
                ),
                "creg" => concat!(
                    "**creg**: QASM 2.0 classical register declaration.\n\n",
                    "Syntax: `creg name[N];`\n\n",
                    "Declares N classical bits for measurement results.",
                ),
                "gate" => concat!(
                    "**gate**: Custom gate definition.\n\n",
                    "Syntax: `gate name(params) qubits { body }`\n\n",
                    "Defines a new unitary gate from existing gates.",
                ),
                "if" => concat!(
                    "**if**: Conditional execution.\n\n",
                    "Syntax: `if (creg == value) { body }`\n\n",
                    "Executes body only when the classical register matches the value.",
                ),
                "while" => concat!(
                    "**while**: While loop.\n\n",
                    "Syntax: `while (creg == value) { body }`\n\n",
                    "Repeatedly executes body while the condition holds.",
                ),
                "for" => concat!(
                    "**for**: For loop.\n\n",
                    "Syntax: `for uint i in [start:end] { body }`\n\n",
                    "Iterates over an integer range.",
                ),
                "def" => concat!(
                    "**def**: Subroutine definition (QASM 3.0).\n\n",
                    "Syntax: `def name(params) qubit q { body }`\n\n",
                    "Defines a reusable subroutine.",
                ),
                "pi" => concat!(
                    "**pi**: Mathematical constant.\n\n",
                    "Value: 3.14159265358979...\n\n",
                    "Used in rotation gate parameters: `rx(pi/4) q[0];`",
                ),
                _ => return None,
            };
            Some(doc.to_string())
        }

        // ================================================================
        // Internal helpers — validation
        // ================================================================

        /// Check that qubit indices stay within declared register bounds (QASM 3.0).
        fn check_qubit_bounds(
            source: &str,
            program: &crate::qasm3::QASM3Program,
            diagnostics: &mut Vec<QasmDiagnostic>,
        ) {
            use crate::qasm3::QASM3Statement;

            // Build a register size map from declarations.
            let mut reg_sizes: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for stmt in &program.statements {
                if let QASM3Statement::QubitDecl { name, size } = stmt {
                    reg_sizes.insert(name.clone(), *size);
                }
            }

            // Walk statements looking for qubit references that exceed bounds.
            for stmt in &program.statements {
                if let QASM3Statement::GateCall { qubits, .. } = stmt {
                    for q in qubits {
                        if let Some(&size) = reg_sizes.get(&q.reg) {
                            if q.index >= size {
                                let line = Self::find_qubit_ref_line(source, &q.reg, q.index);
                                diagnostics.push(QasmDiagnostic {
                                    line,
                                    col: 0,
                                    end_col: Self::line_length(source, line),
                                    severity: DiagnosticSeverity::Error,
                                    message: format!(
                                        "Qubit index {}[{}] out of range (register size is {})",
                                        q.reg, q.index, size
                                    ),
                                    code: Some("qubit-out-of-range".to_string()),
                                });
                            }
                        }
                    }
                }
            }
        }

        /// Check register bounds for QASM 2.0 programs.
        ///
        /// Scans the source text for `qreg` declarations and gate applications,
        /// detecting undeclared registers and out-of-range qubit indices.
        fn check_qasm2_register_bounds(source: &str, diagnostics: &mut Vec<QasmDiagnostic>) {
            let mut qreg_sizes: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();

            // First pass: collect register declarations.
            for line in source.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("qreg") {
                    if let Some((name, size)) = Self::extract_qasm2_reg(trimmed, "qreg") {
                        qreg_sizes.insert(name, size);
                    }
                }
            }

            // If no registers declared, nothing to check.
            if qreg_sizes.is_empty() {
                return;
            }

            // Second pass: check qubit references in gate applications.
            let single_qubit_gates = [
                "h", "x", "y", "z", "s", "t", "sx", "sdg", "tdg", "id", "reset",
            ];
            let two_qubit_gates = ["cx", "cz", "swap", "iswap", "cy", "ch"];
            let three_qubit_gates = ["ccx", "ccz", "cswap"];
            let param_gates = [
                "rx", "ry", "rz", "rxx", "ryy", "rzz", "crx", "cry", "crz", "u1", "u2", "u3", "u",
                "cu", "cu1", "cu3", "p", "cp",
            ];

            for (line_num, line) in source.lines().enumerate() {
                let trimmed = line.trim();

                // Skip declarations, comments, empty lines.
                if trimmed.is_empty()
                    || trimmed.starts_with("//")
                    || trimmed.starts_with("OPENQASM")
                    || trimmed.starts_with("include")
                    || trimmed.starts_with("qreg")
                    || trimmed.starts_with("creg")
                    || trimmed.starts_with("gate")
                    || trimmed.starts_with("measure")
                    || trimmed.starts_with("barrier")
                    || trimmed.starts_with("if")
                    || trimmed.starts_with("}")
                    || trimmed.starts_with("{")
                {
                    continue;
                }

                // Extract the gate name from the line.
                let gate_name = {
                    let first_word_end = trimmed
                        .find(|c: char| !c.is_alphanumeric() && c != '_')
                        .unwrap_or(trimmed.len());
                    &trimmed[..first_word_end]
                };

                let is_gate = single_qubit_gates.contains(&gate_name)
                    || two_qubit_gates.contains(&gate_name)
                    || three_qubit_gates.contains(&gate_name)
                    || param_gates.contains(&gate_name);

                if !is_gate {
                    continue;
                }

                // Extract qubit references from this line.
                // Find the part after the gate name (and params if any).
                let after_gate = if let Some(paren_end) = trimmed.find(')') {
                    &trimmed[paren_end + 1..]
                } else {
                    &trimmed[gate_name.len()..]
                };

                // Parse qubit references like "q[0], q[1]" or "q[0],q[1]"
                let refs_str = after_gate.trim().trim_end_matches(';');
                for qubit_ref in refs_str.split(',') {
                    let qref = qubit_ref.trim();
                    if let Some((reg_name, index)) = Self::parse_qubit_ref(qref) {
                        // Check if register is declared.
                        if let Some(&size) = qreg_sizes.get(&reg_name) {
                            // Check bounds.
                            if index >= size {
                                diagnostics.push(QasmDiagnostic {
                                    line: line_num,
                                    col: 0,
                                    end_col: Self::line_length(source, line_num),
                                    severity: DiagnosticSeverity::Error,
                                    message: format!(
                                        "Qubit index {}[{}] out of range (register size is {})",
                                        reg_name, index, size
                                    ),
                                    code: Some("qubit-out-of-range".to_string()),
                                });
                            }
                        } else {
                            diagnostics.push(QasmDiagnostic {
                                line: line_num,
                                col: 0,
                                end_col: Self::line_length(source, line_num),
                                severity: DiagnosticSeverity::Error,
                                message: format!("Undeclared qubit register: '{}'", reg_name),
                                code: Some("undeclared-register".to_string()),
                            });
                        }
                    }
                }
            }
        }

        /// Parse a qubit reference like "q[2]" into (register_name, index).
        fn parse_qubit_ref(s: &str) -> Option<(String, usize)> {
            let open = s.find('[')?;
            let close = s.find(']')?;
            if open >= close {
                return None;
            }
            let name = s[..open].trim().to_string();
            if name.is_empty() {
                return None;
            }
            let index: usize = s[open + 1..close].trim().parse().ok()?;
            Some((name, index))
        }

        // ================================================================
        // Internal helpers — simulation-aware diagnostics
        // ================================================================

        /// Check simulation resource requirements based on total qubit count.
        fn check_simulation_resources(source: &str, diagnostics: &mut Vec<QasmDiagnostic>) {
            let total_qubits = Self::count_total_qubits(source);

            if total_qubits == 0 {
                return;
            }

            // Find the line of the last qubit declaration for diagnostic placement.
            let decl_line = Self::find_last_qubit_decl_line(source);

            if total_qubits > 40 {
                diagnostics.push(QasmDiagnostic {
                    line: decl_line,
                    col: 0,
                    end_col: Self::line_length(source, decl_line),
                    severity: DiagnosticSeverity::Information,
                    message: format!(
                        "Circuit uses {} qubits: requires distributed or tensor network simulation \
                         (statevector would need 2^{} * 16 bytes = {:.0} TB RAM)",
                        total_qubits,
                        total_qubits,
                        (2.0_f64).powi(total_qubits as i32) * 16.0 / (1024.0 * 1024.0 * 1024.0 * 1024.0)
                    ),
                    code: Some("sim-resource-distributed".to_string()),
                });
            } else if total_qubits > 30 {
                let memory_gb =
                    (2.0_f64).powi(total_qubits as i32) * 16.0 / (1024.0 * 1024.0 * 1024.0);
                diagnostics.push(QasmDiagnostic {
                    line: decl_line,
                    col: 0,
                    end_col: Self::line_length(source, decl_line),
                    severity: DiagnosticSeverity::Information,
                    message: format!(
                        "Circuit uses {} qubits: requires >{:.0} GB RAM for statevector simulation \
                         (2^{} * 16 bytes)",
                        total_qubits, memory_gb, total_qubits
                    ),
                    code: Some("sim-resource-memory".to_string()),
                });
            } else if total_qubits > 20 {
                let memory_bytes = (1u64 << total_qubits) * 16;
                let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);
                diagnostics.push(QasmDiagnostic {
                    line: decl_line,
                    col: 0,
                    end_col: Self::line_length(source, decl_line),
                    severity: DiagnosticSeverity::Information,
                    message: format!(
                        "Circuit uses {} qubits: statevector simulation requires {:.0} MB RAM \
                         (2^{} * 16 bytes for complex f64)",
                        total_qubits, memory_mb, total_qubits
                    ),
                    code: Some("sim-resource-warning".to_string()),
                });
            }
        }

        /// Detect consecutive single-qubit gates on the same qubit that could be fused.
        fn check_gate_fusion_opportunities(source: &str, diagnostics: &mut Vec<QasmDiagnostic>) {
            let single_qubit_gates = [
                "h", "x", "y", "z", "s", "t", "sx", "sdg", "tdg", "id", "rx", "ry", "rz", "u",
                "u1", "u2", "u3", "p",
            ];

            // Track: (gate_name, qubit_ref, line_number) for the previous gate line.
            let mut prev_gate: Option<(String, String, usize)> = None;
            // Track fusion runs to avoid duplicate diagnostics.
            let mut fusion_reported_at: Option<usize> = None;

            for (line_num, line) in source.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.is_empty()
                    || trimmed.starts_with("//")
                    || trimmed.starts_with("OPENQASM")
                    || trimmed.starts_with("include")
                    || trimmed.starts_with("qreg")
                    || trimmed.starts_with("creg")
                    || trimmed.starts_with("qubit")
                    || trimmed.starts_with("bit")
                    || trimmed.starts_with("gate")
                    || trimmed.starts_with("}")
                    || trimmed.starts_with("{")
                    || trimmed.starts_with("barrier")
                {
                    prev_gate = None;
                    continue;
                }

                // Extract gate name (handling parameterized gates).
                let gate_name_end = trimmed
                    .find(|c: char| !c.is_alphanumeric() && c != '_')
                    .unwrap_or(trimmed.len());
                let gate_name = &trimmed[..gate_name_end];

                if !single_qubit_gates.contains(&gate_name) {
                    prev_gate = None;
                    continue;
                }

                // Extract the qubit reference after the gate (and params).
                let after_gate = if let Some(paren_end) = trimmed.find(')') {
                    &trimmed[paren_end + 1..]
                } else {
                    &trimmed[gate_name.len()..]
                };
                let qubit_ref = after_gate.trim().trim_end_matches(';').trim().to_string();

                if let Some((ref prev_name, ref prev_qubit, prev_line)) = prev_gate {
                    if &qubit_ref == prev_qubit && fusion_reported_at != Some(prev_line) {
                        diagnostics.push(QasmDiagnostic {
                            line: line_num,
                            col: 0,
                            end_col: Self::line_length(source, line_num),
                            severity: DiagnosticSeverity::Hint,
                            message: format!(
                                "Consecutive single-qubit gates on {}: '{}' (line {}) and '{}' could be fused into a single U gate",
                                qubit_ref,
                                prev_name,
                                prev_line + 1,
                                gate_name
                            ),
                            code: Some("gate-fusion-opportunity".to_string()),
                        });
                        fusion_reported_at = Some(line_num);
                    }
                }

                prev_gate = Some((gate_name.to_string(), qubit_ref, line_num));
            }
        }

        /// Count total qubits declared in the source (both QASM 2.0 and 3.0 styles).
        fn count_total_qubits(source: &str) -> usize {
            let mut total = 0;
            for line in source.lines() {
                let trimmed = line.trim();
                // QASM 3.0: qubit[N] name;
                if trimmed.starts_with("qubit[") || trimmed.starts_with("qubit [") {
                    if let Some((size, _name)) = Self::extract_qasm3_decl(trimmed, "qubit") {
                        total += size;
                    }
                }
                // QASM 2.0: qreg name[N];
                if trimmed.starts_with("qreg") {
                    if let Some((_name, size)) = Self::extract_qasm2_reg(trimmed, "qreg") {
                        total += size;
                    }
                }
            }
            total
        }

        /// Find the line number of the last qubit declaration.
        fn find_last_qubit_decl_line(source: &str) -> usize {
            let mut last_line = 0;
            for (i, line) in source.lines().enumerate() {
                let trimmed = line.trim();
                if trimmed.starts_with("qubit[")
                    || trimmed.starts_with("qubit [")
                    || trimmed.starts_with("qreg")
                {
                    last_line = i;
                }
            }
            last_line
        }

        /// Check that the source begins with a valid OPENQASM version header.
        fn check_version_header(source: &str, diagnostics: &mut Vec<QasmDiagnostic>) {
            let has_header = source.lines().any(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with("//") && t.starts_with("OPENQASM")
            });

            if !has_header {
                diagnostics.push(QasmDiagnostic {
                    line: 0,
                    col: 0,
                    end_col: 0,
                    severity: DiagnosticSeverity::Warning,
                    message: "Missing OPENQASM version header (e.g. 'OPENQASM 3.0;')".to_string(),
                    code: Some("missing-version".to_string()),
                });
            }
        }

        // ================================================================
        // Internal helpers — utilities
        // ================================================================

        /// Map a QASM3Error to a (line, message) pair.
        fn error_to_line_message(source: &str, err: &QASM3Error) -> (usize, String) {
            match err {
                QASM3Error::UndefinedGate(name) => {
                    let line = Self::find_word_line(source, name);
                    (line, format!("Undefined gate: '{}'", name))
                }
                QASM3Error::UndefinedRegister(name) => {
                    let line = Self::find_word_line(source, name);
                    (line, format!("Undefined register: '{}'", name))
                }
                QASM3Error::QubitOutOfRange { reg, index, size } => {
                    let line = Self::find_qubit_ref_line(source, reg, *index);
                    (
                        line,
                        format!(
                            "Qubit {}[{}] out of range (register size {})",
                            reg, index, size
                        ),
                    )
                }
                QASM3Error::InvalidSyntax(msg) => (0, format!("Invalid syntax: {}", msg)),
                QASM3Error::ParseError(msg) => (0, format!("Parse error: {}", msg)),
            }
        }

        fn qasm3_error_to_line_message(source: &str, err: &QASM3Error) -> (usize, String) {
            Self::error_to_line_message(source, err)
        }

        fn error_code(err: &QASM3Error) -> String {
            match err {
                QASM3Error::ParseError(_) => "parse-error".to_string(),
                QASM3Error::UndefinedGate(_) => "undefined-gate".to_string(),
                QASM3Error::UndefinedRegister(_) => "undefined-register".to_string(),
                QASM3Error::QubitOutOfRange { .. } => "qubit-out-of-range".to_string(),
                QASM3Error::InvalidSyntax(_) => "invalid-syntax".to_string(),
            }
        }

        /// Find the 0-based line containing a word.
        fn find_word_line(source: &str, word: &str) -> usize {
            for (i, line) in source.lines().enumerate() {
                if line.contains(word) {
                    return i;
                }
            }
            0
        }

        /// Find the 0-based line containing `reg[index]`.
        fn find_qubit_ref_line(source: &str, reg: &str, index: usize) -> usize {
            let needle = format!("{}[{}]", reg, index);
            Self::find_word_line(source, &needle)
        }

        /// Length of a specific line in the source.
        fn line_length(source: &str, line: usize) -> usize {
            source.lines().nth(line).map(|l| l.len()).unwrap_or(0)
        }

        /// Check whether the cursor is positioned after a gate name (for qubit
        /// reference completions).
        fn cursor_after_gate_name(prefix: &str) -> bool {
            let words: Vec<&str> = prefix.split_whitespace().collect();
            if words.len() < 1 {
                return false;
            }
            let first = words[0];
            // Known single-word gate names (without trailing params).
            let gates = [
                "h", "x", "y", "z", "s", "t", "sx", "cx", "cz", "swap", "iswap", "ccx", "ccz",
                "cswap", "measure", "reset", "barrier",
            ];
            if gates.contains(&first) && words.len() >= 1 && prefix.ends_with(' ') {
                return true;
            }
            // Parameterized gates: rx(...) followed by space
            if first.contains('(') && prefix.contains(')') && prefix.ends_with(' ') {
                return true;
            }
            false
        }

        /// Extract the partial word at the cursor (for filtering completions).
        fn word_at_cursor(prefix: &str) -> &str {
            let bytes = prefix.as_bytes();
            let mut end = bytes.len();
            while end > 0 && (bytes[end - 1] as char).is_alphanumeric()
                || (end > 0 && bytes[end - 1] == b'_')
            {
                end -= 1;
            }
            &prefix[end..]
        }

        /// Extract the word surrounding a column position.
        fn word_around(line: &str, col: usize) -> Option<String> {
            let bytes = line.as_bytes();
            if col > bytes.len() {
                return None;
            }

            let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';

            let mut start = col;
            while start > 0 && is_word(bytes[start - 1]) {
                start -= 1;
            }

            let mut end = col;
            while end < bytes.len() && is_word(bytes[end]) {
                end += 1;
            }

            if start == end {
                return None;
            }

            Some(line[start..end].to_string())
        }

        /// Helper: create a simple gate completion item.
        fn gate_item(label: &str, detail: &str, insert: &str) -> CompletionItem {
            CompletionItem {
                label: label.to_string(),
                detail: detail.to_string(),
                insert_text: insert.to_string(),
                kind: CompletionKind::Gate,
            }
        }

        /// Extract (size, name) from a QASM 3.0 declaration like "qubit[2] q;"
        fn extract_qasm3_decl(line: &str, keyword: &str) -> Option<(usize, String)> {
            let rest = line.strip_prefix(keyword)?.trim();
            let open = rest.find('[')?;
            let close = rest.find(']')?;
            let size: usize = rest[open + 1..close].trim().parse().ok()?;
            let name = rest[close + 1..]
                .trim()
                .trim_end_matches(';')
                .trim()
                .to_string();
            if name.is_empty() {
                return None;
            }
            Some((size, name))
        }

        /// Extract (name, size) from a QASM 2.0 declaration like "qreg q[2];"
        fn extract_qasm2_reg(line: &str, keyword: &str) -> Option<(String, usize)> {
            let rest = line.strip_prefix(keyword)?.trim();
            let open = rest.find('[')?;
            let close = rest.find(']')?;
            let name = rest[..open].trim().to_string();
            let size: usize = rest[open + 1..close].trim().parse().ok()?;
            if name.is_empty() {
                return None;
            }
            Some((name, size))
        }
    }

    impl Default for QasmLanguageServer {
        fn default() -> Self {
            Self::new()
        }
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
#[cfg(feature = "lsp")]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Existing tests
    // ----------------------------------------------------------------

    #[test]
    fn test_validate_valid_qasm3() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"#;
        let diags = server.validate(src);
        // Valid program should produce no errors.
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_validate_syntax_error() {
        let server = QasmLanguageServer::new();
        // Missing register declaration makes gate lowering fail.
        let src = r#"
OPENQASM 3.0;
h q[0];
"#;
        let diags = server.validate(src);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(!errors.is_empty(), "Expected at least one error");
    }

    #[test]
    fn test_validate_valid_qasm2() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
"#;
        let diags = server.validate(src);
        // QASM 2.0 should parse fine (possibly with an informational note).
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(
            errors.is_empty(),
            "Expected no errors for valid QASM 2.0, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_undefined_gate() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 3.0;
qubit[1] q;
foobar q[0];
"#;
        let diags = server.validate(src);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.severity == DiagnosticSeverity::Error)
            .collect();
        assert!(!errors.is_empty(), "Expected error for undefined gate");
        assert!(
            errors[0].message.contains("foobar"),
            "Expected error about 'foobar', got: {}",
            errors[0].message
        );
    }

    #[test]
    fn test_complete_gates() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;\n";
        let completions = server.complete(src, 2, 0);
        // Should include standard gates.
        let labels: Vec<&str> = completions.iter().map(|c| c.label.as_str()).collect();
        assert!(labels.contains(&"h"), "Missing 'h' gate in completions");
        assert!(labels.contains(&"cx"), "Missing 'cx' gate in completions");
        assert!(labels.contains(&"rx"), "Missing 'rx' gate in completions");
    }

    #[test]
    fn test_complete_keywords() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\n";
        let completions = server.complete(src, 1, 0);
        let labels: Vec<&str> = completions.iter().map(|c| c.label.as_str()).collect();
        assert!(
            labels.contains(&"qubit"),
            "Missing 'qubit' keyword in completions"
        );
        assert!(
            labels.contains(&"gate"),
            "Missing 'gate' keyword in completions"
        );
    }

    #[test]
    fn test_complete_after_gate() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;\nh ";
        let completions = server.complete(src, 2, 2);
        // After "h " we should get qubit reference completions.
        let labels: Vec<&str> = completions.iter().map(|c| c.label.as_str()).collect();
        assert!(
            labels.contains(&"q[0]"),
            "Missing 'q[0]' in post-gate completions, got: {:?}",
            labels
        );
        assert!(
            labels.contains(&"q[1]"),
            "Missing 'q[1]' in post-gate completions"
        );
    }

    #[test]
    fn test_hover_gate() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];";
        let hover = server.hover(src, 2, 0);
        assert!(hover.is_some(), "Expected hover info for 'h' gate");
        let info = hover.unwrap();
        assert!(
            info.contents.contains("Hadamard"),
            "Expected Hadamard doc, got: {}",
            info.contents
        );
    }

    #[test]
    fn test_hover_keyword() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;";
        let hover = server.hover(src, 1, 2);
        assert!(hover.is_some(), "Expected hover info for 'qubit' keyword");
        let info = hover.unwrap();
        assert!(
            info.contents.contains("qubit register"),
            "Expected qubit doc, got: {}",
            info.contents
        );
    }

    #[test]
    fn test_hover_unknown() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;\n";
        let hover = server.hover(src, 2, 0);
        // Empty line should return None.
        assert!(hover.is_none(), "Expected no hover on empty line");
    }

    #[test]
    fn test_validate_qubit_out_of_range() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 3.0;
qubit[2] q;
h q[5];
"#;
        let diags = server.validate(src);
        let oob_errors: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.severity == DiagnosticSeverity::Error
                    && d.code.as_deref() == Some("qubit-out-of-range")
            })
            .collect();
        assert!(
            !oob_errors.is_empty(),
            "Expected out-of-range error for q[5] with register size 2"
        );
    }

    #[test]
    fn test_validate_missing_version_header() {
        let server = QasmLanguageServer::new();
        let src = "qubit[2] q;\nh q[0];\n";
        let diags = server.validate(src);
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("missing-version"))
            .collect();
        assert!(
            !warnings.is_empty(),
            "Expected warning for missing OPENQASM header"
        );
    }

    #[test]
    fn test_complete_version_after_openqasm() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM ";
        let completions = server.complete(src, 0, 9);
        let labels: Vec<&str> = completions.iter().map(|c| c.label.as_str()).collect();
        assert!(
            labels.contains(&"3.0"),
            "Missing '3.0' in version completions"
        );
        assert!(
            labels.contains(&"2.0"),
            "Missing '2.0' in version completions"
        );
    }

    #[test]
    fn test_snippet_completions_available() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\n";
        let completions = server.complete(src, 1, 0);
        let snippets: Vec<_> = completions
            .iter()
            .filter(|c| c.kind == CompletionKind::Snippet)
            .collect();
        assert!(
            snippets.len() >= 3,
            "Expected at least 3 snippet completions, got {}",
            snippets.len()
        );
    }

    // ----------------------------------------------------------------
    // New tests: Incremental Document Sync
    // ----------------------------------------------------------------

    #[test]
    fn test_incremental_edit_single_char_insert() {
        let doc = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n";
        // Insert "x " at the beginning of line 2 (before "h q[0];")
        let result = QasmLanguageServer::apply_incremental_edit(
            doc,
            2,
            0, // start
            2,
            0, // end (empty range = insertion)
            "x q[1];\n",
        );
        assert!(
            result.contains("x q[1];"),
            "Expected inserted text, got: {}",
            result
        );
        assert!(
            result.contains("h q[0];"),
            "Expected original text preserved, got: {}",
            result
        );
    }

    #[test]
    fn test_incremental_edit_replace_range() {
        let doc = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n";
        // Replace "h" with "x" on line 2 (col 0..1).
        let result = QasmLanguageServer::apply_incremental_edit(
            doc, 2, 0, // start
            2, 1, // end (replaces 1 char)
            "x",
        );
        assert!(
            result.contains("x q[0];"),
            "Expected 'x q[0];' after replacing 'h', got: {}",
            result
        );
        assert!(
            !result.contains("h q[0]"),
            "Original 'h q[0]' should be replaced, got: {}",
            result
        );
    }

    #[test]
    fn test_incremental_edit_multiline_replace() {
        let doc = "line0\nline1\nline2\nline3\n";
        // Replace from line 1, col 0 to line 2, col 5 (entire line1 + line2).
        let result = QasmLanguageServer::apply_incremental_edit(doc, 1, 0, 2, 5, "replaced");
        assert_eq!(result, "line0\nreplaced\nline3\n");
    }

    #[test]
    fn test_incremental_edit_delete() {
        let doc = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nx q[1];\n";
        // Delete line 3 entirely ("x q[1];\n").
        let result = QasmLanguageServer::apply_incremental_edit(
            doc, 3, 0, 3, 6, // "x q[1];" is 7 chars but we delete "x q[1];" part
            "",
        );
        assert!(
            !result.contains("x q[1]"),
            "Expected 'x q[1];' deleted, got: {}",
            result
        );
    }

    #[test]
    fn test_incremental_edit_preserves_surrounding() {
        let doc = "aaa\nbbb\nccc\n";
        // Replace "bbb" with "BBB".
        let result = QasmLanguageServer::apply_incremental_edit(doc, 1, 0, 1, 3, "BBB");
        assert_eq!(result, "aaa\nBBB\nccc\n");
    }

    // ----------------------------------------------------------------
    // New tests: Go-to-Definition
    // ----------------------------------------------------------------

    #[test]
    fn test_goto_definition_custom_gate() {
        let server = QasmLanguageServer::new();
        let src = r#"OPENQASM 3.0;
qubit[2] q;
gate mygate q0 {
  h q0;
  x q0;
}
mygate q[0];
"#;
        // Cursor on "mygate" at line 6, col 2.
        let def = server.goto_definition(src, 6, 2);
        assert!(def.is_some(), "Expected definition location for 'mygate'");
        let loc = def.unwrap();
        assert_eq!(loc.line, 2, "Expected definition on line 2");
    }

    #[test]
    fn test_goto_definition_no_match() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n";
        // 'h' is a built-in, not a custom gate definition.
        let def = server.goto_definition(src, 2, 0);
        assert!(
            def.is_none(),
            "Expected no definition for built-in gate 'h'"
        );
    }

    #[test]
    fn test_goto_definition_parameterized_gate() {
        let server = QasmLanguageServer::new();
        let src = r#"OPENQASM 3.0;
qubit[1] q;
gate myrx(theta) q0 {
  rx(theta) q0;
}
myrx(3.14) q[0];
"#;
        let def = server.goto_definition(src, 5, 2);
        assert!(def.is_some(), "Expected definition for 'myrx'");
        let loc = def.unwrap();
        assert_eq!(loc.line, 2, "Expected definition on line 2");
    }

    #[test]
    fn test_goto_definition_multiple_gates() {
        let server = QasmLanguageServer::new();
        let src = r#"OPENQASM 3.0;
qubit[2] q;
gate foo q0 { h q0; }
gate bar q0, q1 { cx q0, q1; }
bar q[0], q[1];
"#;
        // Look up "bar" at line 4.
        let def = server.goto_definition(src, 4, 1);
        assert!(def.is_some(), "Expected definition for 'bar'");
        let loc = def.unwrap();
        assert_eq!(loc.line, 3, "Expected 'bar' defined on line 3");

        // Look up "foo" is not used on line 4, test from a different position
        // if we were to hover on line 2 where "foo" is defined...
    }

    // ----------------------------------------------------------------
    // New tests: QASM 2.0 Register Bounds
    // ----------------------------------------------------------------

    #[test]
    fn test_qasm2_qubit_out_of_range() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[5];
"#;
        let diags = server.validate(src);
        let oob: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("qubit-out-of-range"))
            .collect();
        assert!(
            !oob.is_empty(),
            "Expected out-of-range error for q[5] in QASM 2.0 (qreg q[3]), got diags: {:?}",
            diags
                .iter()
                .map(|d| (&d.code, &d.message))
                .collect::<Vec<_>>()
        );
        assert!(
            oob[0].message.contains("q[5]") && oob[0].message.contains("3"),
            "Expected message about q[5] exceeding size 3, got: {}",
            oob[0].message
        );
    }

    #[test]
    fn test_qasm2_undeclared_register() {
        let server = QasmLanguageServer::new();
        // This source uses an undeclared register 'r'. The QASM 3.0 parser
        // catches this as UndefinedRegister, and when QASM 2.0 fallback
        // succeeds, our QASM 2.0 bounds checker also flags it. Either path
        // should report the problem.
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h r[0];
"#;
        let diags = server.validate(src);
        // The undeclared register should be caught -- either as
        // "undefined-register" (QASM3 path) or "undeclared-register" (QASM2 bounds).
        let undecl: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.code.as_deref() == Some("undeclared-register")
                    || d.code.as_deref() == Some("undefined-register")
                    || d.message.contains("ndeclared")
                    || d.message.contains("ndefined")
            })
            .collect();
        assert!(
            !undecl.is_empty(),
            "Expected undeclared/undefined register error for 'r' in QASM 2.0, got diags: {:?}",
            diags
                .iter()
                .map(|d| (&d.code, &d.message))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_qasm2_valid_bounds() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[2];
"#;
        let diags = server.validate(src);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.severity == DiagnosticSeverity::Error
                    && (d.code.as_deref() == Some("qubit-out-of-range")
                        || d.code.as_deref() == Some("undeclared-register"))
            })
            .collect();
        assert!(
            errors.is_empty(),
            "Expected no bounds errors for valid QASM 2.0, got: {:?}",
            errors
        );
    }

    // ----------------------------------------------------------------
    // New tests: Simulation-Aware Diagnostics
    // ----------------------------------------------------------------

    #[test]
    fn test_sim_resource_warning_25_qubits() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[25] q;\nh q[0];\n";
        let diags = server.validate(src);
        let sim_diags: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("sim-resource-warning"))
            .collect();
        assert!(
            !sim_diags.is_empty(),
            "Expected sim-resource-warning for 25 qubits"
        );
        assert!(
            sim_diags[0].message.contains("25 qubits"),
            "Expected message mentioning 25 qubits, got: {}",
            sim_diags[0].message
        );
    }

    #[test]
    fn test_sim_resource_memory_35_qubits() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[35] q;\nh q[0];\n";
        let diags = server.validate(src);
        let sim_diags: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("sim-resource-memory"))
            .collect();
        assert!(
            !sim_diags.is_empty(),
            "Expected sim-resource-memory for 35 qubits"
        );
        assert!(
            sim_diags[0].message.contains("GB RAM"),
            "Expected memory warning message, got: {}",
            sim_diags[0].message
        );
    }

    #[test]
    fn test_sim_resource_distributed_50_qubits() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[50] q;\nh q[0];\n";
        let diags = server.validate(src);
        let sim_diags: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("sim-resource-distributed"))
            .collect();
        assert!(
            !sim_diags.is_empty(),
            "Expected sim-resource-distributed for 50 qubits"
        );
        assert!(
            sim_diags[0]
                .message
                .contains("distributed or tensor network"),
            "Expected distributed simulation message, got: {}",
            sim_diags[0].message
        );
    }

    #[test]
    fn test_sim_no_warning_small_circuit() {
        let server = QasmLanguageServer::new();
        let src = "OPENQASM 3.0;\nqubit[5] q;\nh q[0];\n";
        let diags = server.validate(src);
        let sim_diags: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.code
                    .as_deref()
                    .map(|c| c.starts_with("sim-resource"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            sim_diags.is_empty(),
            "Expected no simulation warnings for 5-qubit circuit, got: {:?}",
            sim_diags
        );
    }

    #[test]
    fn test_gate_fusion_opportunity() {
        let server = QasmLanguageServer::new();
        let src = r#"OPENQASM 3.0;
qubit[2] q;
h q[0];
x q[0];
"#;
        let diags = server.validate(src);
        let fusion: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("gate-fusion-opportunity"))
            .collect();
        assert!(
            !fusion.is_empty(),
            "Expected gate fusion hint for consecutive h,x on q[0], got diags: {:?}",
            diags
                .iter()
                .map(|d| (&d.code, &d.message))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_no_gate_fusion_different_qubits() {
        let server = QasmLanguageServer::new();
        let src = r#"OPENQASM 3.0;
qubit[2] q;
h q[0];
x q[1];
"#;
        let diags = server.validate(src);
        let fusion: Vec<_> = diags
            .iter()
            .filter(|d| d.code.as_deref() == Some("gate-fusion-opportunity"))
            .collect();
        assert!(
            fusion.is_empty(),
            "Expected no gate fusion hint for gates on different qubits"
        );
    }

    #[test]
    fn test_sim_resource_qasm2() {
        let server = QasmLanguageServer::new();
        let src = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
h q[0];
"#;
        let diags = server.validate(src);
        let sim_diags: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.code
                    .as_deref()
                    .map(|c| c.starts_with("sim-resource"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !sim_diags.is_empty(),
            "Expected simulation resource warning for 25-qubit QASM 2.0 circuit"
        );
    }
}
