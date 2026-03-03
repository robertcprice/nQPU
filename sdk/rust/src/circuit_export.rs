//! Circuit Export: LaTeX (quantikz) and SVG visualization
//!
//! Generates publication-quality quantum circuit diagrams for papers,
//! presentations, and web applications.
//!
//! Supports three output formats:
//! - **LaTeX** (quantikz package): For academic papers and typeset documents
//! - **SVG**: For web applications and scalable graphics
//! - **ASCII**: For terminal output and plain-text environments
//!
//! # Example
//!
//! ```rust
//! use nqpu_metal::circuit_export::{VisCircuit, CircuitExporter, ExportConfig, ExportFormat};
//!
//! let mut circ = VisCircuit::new(2);
//! circ.h(0);
//! circ.cx(0, 1);
//! circ.measure(0);
//! circ.measure(1);
//!
//! let config = ExportConfig::default();
//! let exporter = CircuitExporter::new(config);
//! let latex = exporter.export(&circ).unwrap();
//! ```

use std::fmt;

// ============================================================
// ERROR TYPES
// ============================================================

/// Errors that can occur during circuit export.
#[derive(Debug, Clone)]
pub enum ExportError {
    /// The circuit description is invalid (e.g., qubit index out of range).
    InvalidCircuit(String),
    /// A rendering error occurred during export.
    RenderError(String),
    /// A gate type is not supported by the chosen export format.
    UnsupportedGate(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExportError::InvalidCircuit(msg) => write!(f, "Invalid circuit: {}", msg),
            ExportError::RenderError(msg) => write!(f, "Render error: {}", msg),
            ExportError::UnsupportedGate(msg) => write!(f, "Unsupported gate: {}", msg),
        }
    }
}

impl std::error::Error for ExportError {}

// ============================================================
// EXPORT FORMAT
// ============================================================

/// Output format for the circuit diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// LaTeX using the quantikz package.
    Latex,
    /// Scalable Vector Graphics.
    Svg,
    /// Plain-text ASCII art for terminal display.
    Ascii,
}

// ============================================================
// EXPORT CONFIG
// ============================================================

/// Configuration for circuit diagram export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output format selection.
    pub format: ExportFormat,
    /// Show rotation angles on parameterized gates.
    pub show_parameters: bool,
    /// Show measurement symbols.
    pub show_measurements: bool,
    /// Color-code gate families (SVG/LaTeX).
    pub color_gates: bool,
    /// Minimize whitespace in output.
    pub compact: bool,
    /// Gate box width in em units (LaTeX).
    pub gate_width_em: f64,
    /// Wire vertical spacing in em units (LaTeX).
    pub wire_spacing_em: f64,
    /// Font size in pixels (SVG).
    pub font_size: f64,
    /// Maximum gates per row before wrapping.
    pub max_gates_per_row: usize,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Latex,
            show_parameters: true,
            show_measurements: true,
            color_gates: false,
            compact: false,
            gate_width_em: 1.5,
            wire_spacing_em: 1.0,
            font_size: 14.0,
            max_gates_per_row: 20,
        }
    }
}

impl ExportConfig {
    /// Create a new config with the given format.
    pub fn with_format(mut self, format: ExportFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable or disable parameter display.
    pub fn with_show_parameters(mut self, show: bool) -> Self {
        self.show_parameters = show;
        self
    }

    /// Enable or disable measurement symbols.
    pub fn with_show_measurements(mut self, show: bool) -> Self {
        self.show_measurements = show;
        self
    }

    /// Enable or disable gate coloring.
    pub fn with_color_gates(mut self, color: bool) -> Self {
        self.color_gates = color;
        self
    }

    /// Enable or disable compact output.
    pub fn with_compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    /// Set the SVG font size.
    pub fn with_font_size(mut self, size: f64) -> Self {
        self.font_size = size;
        self
    }

    /// Set the maximum gates per row.
    pub fn with_max_gates_per_row(mut self, max: usize) -> Self {
        self.max_gates_per_row = max;
        self
    }
}

// ============================================================
// VISUALIZATION GATE
// ============================================================

/// Represents a gate for visualization purposes.
#[derive(Debug, Clone)]
pub enum VisGate {
    /// A single-qubit gate (e.g., H, X, Rz).
    Single {
        name: String,
        qubit: usize,
        params: Vec<f64>,
    },
    /// A controlled gate with one control and one target (e.g., CNOT, CZ).
    Controlled {
        name: String,
        control: usize,
        target: usize,
        params: Vec<f64>,
    },
    /// A multi-controlled gate (e.g., Toffoli = CCX).
    MultiControlled {
        name: String,
        controls: Vec<usize>,
        target: usize,
    },
    /// A SWAP gate between two qubits.
    Swap { q1: usize, q2: usize },
    /// A barrier across specified qubits (visual separator).
    Barrier { qubits: Vec<usize> },
    /// A measurement operation on a qubit, optionally writing to a classical bit.
    Measure {
        qubit: usize,
        cbit: Option<usize>,
    },
    /// A qubit reset to |0>.
    Reset { qubit: usize },
}

impl VisGate {
    /// Return all qubit indices touched by this gate.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            VisGate::Single { qubit, .. } => vec![*qubit],
            VisGate::Controlled {
                control, target, ..
            } => vec![*control, *target],
            VisGate::MultiControlled {
                controls, target, ..
            } => {
                let mut qs = controls.clone();
                qs.push(*target);
                qs
            }
            VisGate::Swap { q1, q2 } => vec![*q1, *q2],
            VisGate::Barrier { qubits } => qubits.clone(),
            VisGate::Measure { qubit, .. } => vec![*qubit],
            VisGate::Reset { qubit } => vec![*qubit],
        }
    }
}

// ============================================================
// VISUALIZATION CIRCUIT
// ============================================================

/// A quantum circuit description for visualization.
pub struct VisCircuit {
    /// Number of quantum wires.
    pub num_qubits: usize,
    /// Number of classical wires.
    pub num_cbits: usize,
    /// Ordered sequence of gates.
    pub gates: Vec<VisGate>,
    /// Labels for quantum wires (e.g., ["q_0", "q_1"]).
    pub qubit_labels: Vec<String>,
    /// Labels for classical wires.
    pub cbit_labels: Vec<String>,
}

impl VisCircuit {
    /// Create a new circuit with the given number of qubits and default labels.
    pub fn new(num_qubits: usize) -> Self {
        let qubit_labels = (0..num_qubits)
            .map(|i| format!("q_{}", i))
            .collect();
        Self {
            num_qubits,
            num_cbits: 0,
            gates: Vec::new(),
            qubit_labels,
            cbit_labels: Vec::new(),
        }
    }

    /// Create a circuit with both quantum and classical bits.
    pub fn with_classical(num_qubits: usize, num_cbits: usize) -> Self {
        let qubit_labels = (0..num_qubits)
            .map(|i| format!("q_{}", i))
            .collect();
        let cbit_labels = (0..num_cbits)
            .map(|i| format!("c_{}", i))
            .collect();
        Self {
            num_qubits,
            num_cbits,
            gates: Vec::new(),
            qubit_labels,
            cbit_labels,
        }
    }

    /// Set custom qubit labels.
    pub fn set_qubit_labels(&mut self, labels: Vec<String>) {
        self.qubit_labels = labels;
    }

    /// Append a Hadamard gate.
    pub fn h(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "H".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append a Pauli-X gate.
    pub fn x(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "X".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append a Pauli-Y gate.
    pub fn y(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "Y".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append a Pauli-Z gate.
    pub fn z(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "Z".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append an S gate (phase gate, sqrt(Z)).
    pub fn s(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "S".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append a T gate (pi/8 gate, sqrt(S)).
    pub fn t(&mut self, qubit: usize) {
        self.gates.push(VisGate::Single {
            name: "T".into(),
            qubit,
            params: vec![],
        });
    }

    /// Append an Rz rotation gate.
    pub fn rz(&mut self, qubit: usize, angle: f64) {
        self.gates.push(VisGate::Single {
            name: "R_z".into(),
            qubit,
            params: vec![angle],
        });
    }

    /// Append an Rx rotation gate.
    pub fn rx(&mut self, qubit: usize, angle: f64) {
        self.gates.push(VisGate::Single {
            name: "R_x".into(),
            qubit,
            params: vec![angle],
        });
    }

    /// Append an Ry rotation gate.
    pub fn ry(&mut self, qubit: usize, angle: f64) {
        self.gates.push(VisGate::Single {
            name: "R_y".into(),
            qubit,
            params: vec![angle],
        });
    }

    /// Append a CNOT (controlled-X) gate.
    pub fn cx(&mut self, control: usize, target: usize) {
        self.gates.push(VisGate::Controlled {
            name: "X".into(),
            control,
            target,
            params: vec![],
        });
    }

    /// Append a CZ (controlled-Z) gate.
    pub fn cz(&mut self, control: usize, target: usize) {
        self.gates.push(VisGate::Controlled {
            name: "Z".into(),
            control,
            target,
            params: vec![],
        });
    }

    /// Append a Toffoli (CCX) gate.
    pub fn ccx(&mut self, c1: usize, c2: usize, target: usize) {
        self.gates.push(VisGate::MultiControlled {
            name: "X".into(),
            controls: vec![c1, c2],
            target,
        });
    }

    /// Append a SWAP gate.
    pub fn swap(&mut self, q1: usize, q2: usize) {
        self.gates.push(VisGate::Swap { q1, q2 });
    }

    /// Append a measurement on the given qubit.
    pub fn measure(&mut self, qubit: usize) {
        self.gates.push(VisGate::Measure { qubit, cbit: None });
    }

    /// Append a measurement writing to a specific classical bit.
    pub fn measure_to(&mut self, qubit: usize, cbit: usize) {
        self.gates.push(VisGate::Measure {
            qubit,
            cbit: Some(cbit),
        });
    }

    /// Append a barrier across the given qubits.
    pub fn barrier(&mut self, qubits: &[usize]) {
        self.gates.push(VisGate::Barrier {
            qubits: qubits.to_vec(),
        });
    }

    /// Append a reset gate.
    pub fn reset(&mut self, qubit: usize) {
        self.gates.push(VisGate::Reset { qubit });
    }
}

// ============================================================
// CIRCUIT EXPORTER
// ============================================================

/// Exports a `VisCircuit` to LaTeX (quantikz), SVG, or ASCII format.
pub struct CircuitExporter {
    config: ExportConfig,
}

impl CircuitExporter {
    /// Create a new exporter with the given configuration.
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export the circuit in the format specified by config.
    pub fn export(&self, circuit: &VisCircuit) -> Result<String, ExportError> {
        self.validate(circuit)?;
        match self.config.format {
            ExportFormat::Latex => Ok(self.to_latex(circuit)),
            ExportFormat::Svg => Ok(self.to_svg(circuit)),
            ExportFormat::Ascii => Ok(self.to_ascii(circuit)),
        }
    }

    /// Validate that the circuit is well-formed.
    fn validate(&self, circuit: &VisCircuit) -> Result<(), ExportError> {
        for (i, gate) in circuit.gates.iter().enumerate() {
            for q in gate.qubits() {
                if q >= circuit.num_qubits {
                    return Err(ExportError::InvalidCircuit(format!(
                        "gate {} references qubit {} but circuit has only {} qubits",
                        i, q, circuit.num_qubits
                    )));
                }
            }
        }
        Ok(())
    }

    // --------------------------------------------------------
    // LaTeX (quantikz) export
    // --------------------------------------------------------

    /// Generate a LaTeX quantikz diagram string.
    pub fn to_latex(&self, circuit: &VisCircuit) -> String {
        let total_wires = circuit.num_qubits + circuit.num_cbits;
        if total_wires == 0 {
            return "\\begin{quantikz}\n\\end{quantikz}".to_string();
        }

        // Build a column-major grid: grid[col][wire] = latex_cell
        let columns = self.build_latex_columns(circuit);

        let mut lines: Vec<String> = Vec::new();
        lines.push("\\begin{quantikz}".to_string());

        for wire in 0..total_wires {
            let mut row_parts: Vec<String> = Vec::new();

            // Left label
            let label = if wire < circuit.num_qubits {
                let raw = &circuit.qubit_labels[wire];
                format!("\\lstick{{${}$}}", raw)
            } else {
                let cbit_idx = wire - circuit.num_qubits;
                let raw = if cbit_idx < circuit.cbit_labels.len() {
                    circuit.cbit_labels[cbit_idx].clone()
                } else {
                    format!("c_{}", cbit_idx)
                };
                format!("\\lstick{{${}$}}", raw)
            };
            row_parts.push(label);

            // Gate columns
            for col in &columns {
                row_parts.push(col[wire].clone());
            }

            let sep = if self.config.compact { " & " } else { " & " };
            let mut row = row_parts.join(sep);
            if wire < total_wires - 1 {
                row.push_str(" \\\\");
            }
            lines.push(row);
        }

        lines.push("\\end{quantikz}".to_string());
        lines.join("\n")
    }

    /// Build column data for the LaTeX grid. Each column is a Vec<String> of
    /// length `num_qubits + num_cbits`, one entry per wire.
    fn build_latex_columns(&self, circuit: &VisCircuit) -> Vec<Vec<String>> {
        let total_wires = circuit.num_qubits + circuit.num_cbits;
        let mut columns: Vec<Vec<String>> = Vec::new();

        for gate in &circuit.gates {
            let mut col = vec!["\\qw".to_string(); total_wires];
            // Classical wires default to \cw
            for w in circuit.num_qubits..total_wires {
                col[w] = "\\cw".to_string();
            }

            match gate {
                VisGate::Single {
                    name,
                    qubit,
                    params,
                } => {
                    let gate_text = self.format_latex_gate_label(name, params);
                    col[*qubit] = format!("\\gate{{{}}}", gate_text);
                }
                VisGate::Controlled {
                    name,
                    control,
                    target,
                    params,
                } => {
                    if name == "X" && params.is_empty() {
                        // CNOT: control dot + target oplus
                        let dist = *target as isize - *control as isize;
                        col[*control] = format!("\\ctrl{{{}}}", dist);
                        col[*target] = "\\targ{}".to_string();
                    } else {
                        // General controlled gate
                        let dist = *target as isize - *control as isize;
                        let gate_text = self.format_latex_gate_label(name, params);
                        col[*control] = format!("\\ctrl{{{}}}", dist);
                        col[*target] = format!("\\gate{{{}}}", gate_text);
                    }
                }
                VisGate::MultiControlled {
                    name,
                    controls,
                    target,
                } => {
                    // All controls get dots, target gets gate or oplus
                    for ctrl in controls {
                        let dist = *target as isize - *ctrl as isize;
                        col[*ctrl] = format!("\\ctrl{{{}}}", dist);
                    }
                    if name == "X" {
                        col[*target] = "\\targ{}".to_string();
                    } else {
                        col[*target] = format!("\\gate{{{}}}", name);
                    }
                }
                VisGate::Swap { q1, q2 } => {
                    let dist = *q2 as isize - *q1 as isize;
                    col[*q1] = format!("\\swap{{{}}}", dist);
                    col[*q2] = "\\targX{}".to_string();
                }
                VisGate::Barrier { qubits } => {
                    for &q in qubits {
                        col[q] = "\\qw \\barrier{}".to_string();
                    }
                }
                VisGate::Measure { qubit, .. } => {
                    if self.config.show_measurements {
                        col[*qubit] = "\\meter{}".to_string();
                    }
                }
                VisGate::Reset { qubit } => {
                    col[*qubit] = "\\gate{\\ket{0}}".to_string();
                }
            }
            columns.push(col);
        }

        // If no gates, produce at least one identity column
        if columns.is_empty() {
            let mut col = vec!["\\qw".to_string(); total_wires];
            for w in circuit.num_qubits..total_wires {
                col[w] = "\\cw".to_string();
            }
            columns.push(col);
        }

        columns
    }

    /// Format a gate label for LaTeX, appending parameters if enabled.
    fn format_latex_gate_label(&self, name: &str, params: &[f64]) -> String {
        if !self.config.show_parameters || params.is_empty() {
            return name.to_string();
        }
        let param_str: Vec<String> = params.iter().map(|p| format_angle(*p)).collect();
        format!("{}({})", name, param_str.join(", "))
    }

    // --------------------------------------------------------
    // SVG export
    // --------------------------------------------------------

    /// Generate an SVG diagram string.
    pub fn to_svg(&self, circuit: &VisCircuit) -> String {
        let font = self.config.font_size;
        let wire_gap = 40.0;
        let gate_gap = 60.0;
        let margin_left = 80.0;
        let margin_top = 40.0;
        let margin_right = 40.0;
        let margin_bottom = 20.0;

        let num_wires = circuit.num_qubits + circuit.num_cbits;
        let num_cols = circuit.gates.len().max(1);

        let width = margin_left + (num_cols as f64) * gate_gap + margin_right;
        let height = margin_top + (num_wires.max(1) as f64 - 1.0) * wire_gap + margin_bottom + 20.0;

        let mut svg = String::new();
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" \
             font-family=\"monospace\" font-size=\"{}\">\n",
            width, height, font
        ));
        svg.push_str("<style>\n");
        svg.push_str("  .wire { stroke: #333; stroke-width: 1.5; fill: none; }\n");
        svg.push_str("  .cwire { stroke: #333; stroke-width: 1.5; stroke-dasharray: 4,3; fill: none; }\n");
        svg.push_str("  .gate-box { fill: #e8f4fd; stroke: #2980b9; stroke-width: 1.5; rx: 4; }\n");
        svg.push_str("  .gate-text { text-anchor: middle; dominant-baseline: central; fill: #2c3e50; }\n");
        svg.push_str("  .ctrl-dot { fill: #2c3e50; }\n");
        svg.push_str("  .targ-circle { fill: none; stroke: #2c3e50; stroke-width: 1.5; }\n");
        svg.push_str("  .meas-box { fill: #fef9e7; stroke: #f39c12; stroke-width: 1.5; rx: 4; }\n");
        svg.push_str("  .label { text-anchor: end; dominant-baseline: central; fill: #2c3e50; }\n");
        svg.push_str("</style>\n");

        // Wire y-positions
        let wire_y = |w: usize| margin_top + (w as f64) * wire_gap;
        let col_x = |c: usize| margin_left + (c as f64) * gate_gap + gate_gap * 0.5;

        // Draw wires
        let wire_end_x = margin_left + (num_cols as f64) * gate_gap;
        for w in 0..num_wires {
            let y = wire_y(w);
            let class = if w < circuit.num_qubits {
                "wire"
            } else {
                "cwire"
            };
            svg.push_str(&format!(
                "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"{}\" />\n",
                margin_left, y, wire_end_x, y, class
            ));
        }

        // Draw labels
        for w in 0..circuit.num_qubits {
            let label = &circuit.qubit_labels[w];
            svg.push_str(&format!(
                "  <text x=\"{}\" y=\"{}\" class=\"label\">{}</text>\n",
                margin_left - 10.0,
                wire_y(w),
                svg_escape(label)
            ));
        }
        for w in 0..circuit.num_cbits {
            let label = if w < circuit.cbit_labels.len() {
                circuit.cbit_labels[w].clone()
            } else {
                format!("c_{}", w)
            };
            svg.push_str(&format!(
                "  <text x=\"{}\" y=\"{}\" class=\"label\">{}</text>\n",
                margin_left - 10.0,
                wire_y(circuit.num_qubits + w),
                svg_escape(&label)
            ));
        }

        // Draw gates
        let gate_w = 40.0;
        let gate_h = 28.0;

        for (col, gate) in circuit.gates.iter().enumerate() {
            let cx = col_x(col);

            match gate {
                VisGate::Single {
                    name,
                    qubit,
                    params,
                } => {
                    let y = wire_y(*qubit);
                    let label = self.format_svg_gate_label(name, params);
                    svg.push_str(&format!(
                        "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"gate-box\" />\n",
                        cx - gate_w / 2.0,
                        y - gate_h / 2.0,
                        gate_w,
                        gate_h
                    ));
                    svg.push_str(&format!(
                        "  <text x=\"{}\" y=\"{}\" class=\"gate-text\">{}</text>\n",
                        cx, y, svg_escape(&label)
                    ));
                }
                VisGate::Controlled {
                    name,
                    control,
                    target,
                    params,
                } => {
                    let cy = wire_y(*control);
                    let ty = wire_y(*target);
                    // Vertical connector line
                    svg.push_str(&format!(
                        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                        cx, cy, cx, ty
                    ));
                    // Control dot
                    svg.push_str(&format!(
                        "  <circle cx=\"{}\" cy=\"{}\" r=\"4\" class=\"ctrl-dot\" />\n",
                        cx, cy
                    ));
                    if name == "X" && params.is_empty() {
                        // CNOT target: circle with plus
                        let r = 10.0;
                        svg.push_str(&format!(
                            "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" class=\"targ-circle\" />\n",
                            cx, ty, r
                        ));
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx - r, ty, cx + r, ty
                        ));
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx,
                            ty - r,
                            cx,
                            ty + r
                        ));
                    } else {
                        // General controlled gate box on target
                        let label = self.format_svg_gate_label(name, params);
                        svg.push_str(&format!(
                            "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"gate-box\" />\n",
                            cx - gate_w / 2.0,
                            ty - gate_h / 2.0,
                            gate_w,
                            gate_h
                        ));
                        svg.push_str(&format!(
                            "  <text x=\"{}\" y=\"{}\" class=\"gate-text\">{}</text>\n",
                            cx, ty, svg_escape(&label)
                        ));
                    }
                }
                VisGate::MultiControlled {
                    name,
                    controls,
                    target,
                } => {
                    let ty = wire_y(*target);
                    for ctrl in controls {
                        let cy = wire_y(*ctrl);
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx, cy, cx, ty
                        ));
                        svg.push_str(&format!(
                            "  <circle cx=\"{}\" cy=\"{}\" r=\"4\" class=\"ctrl-dot\" />\n",
                            cx, cy
                        ));
                    }
                    if name == "X" {
                        let r = 10.0;
                        svg.push_str(&format!(
                            "  <circle cx=\"{}\" cy=\"{}\" r=\"{}\" class=\"targ-circle\" />\n",
                            cx, ty, r
                        ));
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx - r, ty, cx + r, ty
                        ));
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx,
                            ty - r,
                            cx,
                            ty + r
                        ));
                    } else {
                        svg.push_str(&format!(
                            "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"gate-box\" />\n",
                            cx - gate_w / 2.0,
                            ty - gate_h / 2.0,
                            gate_w,
                            gate_h
                        ));
                        svg.push_str(&format!(
                            "  <text x=\"{}\" y=\"{}\" class=\"gate-text\">{}</text>\n",
                            cx, ty, svg_escape(name)
                        ));
                    }
                }
                VisGate::Swap { q1, q2 } => {
                    let y1 = wire_y(*q1);
                    let y2 = wire_y(*q2);
                    let arm = 6.0;
                    // Vertical connector
                    svg.push_str(&format!(
                        "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                        cx, y1, cx, y2
                    ));
                    // X marks on each qubit
                    for y in [y1, y2] {
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx - arm,
                            y - arm,
                            cx + arm,
                            y + arm
                        ));
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" class=\"wire\" />\n",
                            cx - arm,
                            y + arm,
                            cx + arm,
                            y - arm
                        ));
                    }
                }
                VisGate::Barrier { qubits } => {
                    if !qubits.is_empty() {
                        let y_min = qubits.iter().map(|q| wire_y(*q)).fold(f64::INFINITY, f64::min);
                        let y_max = qubits.iter().map(|q| wire_y(*q)).fold(f64::NEG_INFINITY, f64::max);
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                             stroke=\"#999\" stroke-width=\"1\" stroke-dasharray=\"4,2\" />\n",
                            cx,
                            y_min - 10.0,
                            cx,
                            y_max + 10.0
                        ));
                    }
                }
                VisGate::Measure { qubit, .. } => {
                    if self.config.show_measurements {
                        let y = wire_y(*qubit);
                        let mw = 32.0;
                        let mh = 28.0;
                        svg.push_str(&format!(
                            "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"meas-box\" />\n",
                            cx - mw / 2.0,
                            y - mh / 2.0,
                            mw,
                            mh
                        ));
                        // Meter arc
                        let arc_y = y + 2.0;
                        let arc_r = 8.0;
                        svg.push_str(&format!(
                            "  <path d=\"M {} {} A {} {} 0 0 1 {} {}\" \
                             fill=\"none\" stroke=\"#f39c12\" stroke-width=\"1.2\" />\n",
                            cx - arc_r,
                            arc_y,
                            arc_r,
                            arc_r,
                            cx + arc_r,
                            arc_y
                        ));
                        // Meter needle
                        svg.push_str(&format!(
                            "  <line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" \
                             stroke=\"#f39c12\" stroke-width=\"1.2\" />\n",
                            cx,
                            arc_y,
                            cx + 5.0,
                            y - mh / 2.0 + 4.0
                        ));
                    }
                }
                VisGate::Reset { qubit } => {
                    let y = wire_y(*qubit);
                    svg.push_str(&format!(
                        "  <rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" class=\"gate-box\" />\n",
                        cx - gate_w / 2.0,
                        y - gate_h / 2.0,
                        gate_w,
                        gate_h
                    ));
                    svg.push_str(&format!(
                        "  <text x=\"{}\" y=\"{}\" class=\"gate-text\">|0&gt;</text>\n",
                        cx, y
                    ));
                }
            }
        }

        svg.push_str("</svg>\n");
        svg
    }

    /// Format a gate label for SVG output.
    fn format_svg_gate_label(&self, name: &str, params: &[f64]) -> String {
        if !self.config.show_parameters || params.is_empty() {
            return name.to_string();
        }
        let param_str: Vec<String> = params.iter().map(|p| format_angle(*p)).collect();
        format!("{}({})", name, param_str.join(","))
    }

    // --------------------------------------------------------
    // ASCII export
    // --------------------------------------------------------

    /// Generate a plain-text ASCII circuit diagram.
    pub fn to_ascii(&self, circuit: &VisCircuit) -> String {
        if circuit.num_qubits == 0 {
            return String::new();
        }

        // Compute label width for alignment
        let label_width = circuit
            .qubit_labels
            .iter()
            .map(|l| l.len())
            .max()
            .unwrap_or(2);

        // Build column strings per wire
        let num_cols = circuit.gates.len();
        // Each gate gets a fixed-width cell
        let cell_width = 5;

        // grid[wire][col] = cell string (fixed width)
        let mut grid: Vec<Vec<String>> =
            vec![vec![pad_center("-", cell_width); num_cols]; circuit.num_qubits];
        // connector[wire][col] = true if vertical connector needed
        let mut connector: Vec<Vec<bool>> =
            vec![vec![false; num_cols]; circuit.num_qubits];

        for (col, gate) in circuit.gates.iter().enumerate() {
            match gate {
                VisGate::Single { name, qubit, .. } => {
                    let short = ascii_gate_label(name);
                    grid[*qubit][col] = pad_center(&short, cell_width);
                }
                VisGate::Controlled {
                    name,
                    control,
                    target,
                    ..
                } => {
                    grid[*control][col] = pad_center("*", cell_width);
                    let tgt_label = if name == "X" { "X" } else { name.as_str() };
                    grid[*target][col] = pad_center(tgt_label, cell_width);
                    let lo = (*control).min(*target);
                    let hi = (*control).max(*target);
                    for w in (lo + 1)..hi {
                        connector[w][col] = true;
                        grid[w][col] = pad_center("|", cell_width);
                    }
                }
                VisGate::MultiControlled {
                    name,
                    controls,
                    target,
                } => {
                    for ctrl in controls {
                        grid[*ctrl][col] = pad_center("*", cell_width);
                    }
                    let tgt_label = if name == "X" { "X" } else { name.as_str() };
                    grid[*target][col] = pad_center(tgt_label, cell_width);
                    // Connect all involved wires
                    let all: Vec<usize> = controls
                        .iter()
                        .copied()
                        .chain(std::iter::once(*target))
                        .collect();
                    let lo = *all.iter().min().unwrap();
                    let hi = *all.iter().max().unwrap();
                    for w in (lo + 1)..hi {
                        if !all.contains(&w) {
                            connector[w][col] = true;
                            grid[w][col] = pad_center("|", cell_width);
                        }
                    }
                }
                VisGate::Swap { q1, q2 } => {
                    grid[*q1][col] = pad_center("x", cell_width);
                    grid[*q2][col] = pad_center("x", cell_width);
                    let lo = (*q1).min(*q2);
                    let hi = (*q1).max(*q2);
                    for w in (lo + 1)..hi {
                        connector[w][col] = true;
                        grid[w][col] = pad_center("|", cell_width);
                    }
                }
                VisGate::Barrier { qubits } => {
                    for &q in qubits {
                        grid[q][col] = pad_center("||", cell_width);
                    }
                }
                VisGate::Measure { qubit, .. } => {
                    if self.config.show_measurements {
                        grid[*qubit][col] = pad_center("M", cell_width);
                    }
                }
                VisGate::Reset { qubit } => {
                    grid[*qubit][col] = pad_center("|0>", cell_width);
                }
            }
        }

        // Assemble output
        let mut lines: Vec<String> = Vec::new();
        for w in 0..circuit.num_qubits {
            let label = &circuit.qubit_labels[w];
            let padded_label = format!("{:>width$}", label, width = label_width);
            let wire_str: String = grid[w].join("");
            let wire_str = wire_str.replace(' ', "-");
            lines.push(format!("{}: {}", padded_label, wire_str));
        }

        lines.join("\n")
    }
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/// Format an angle as a human-readable string. Recognizes multiples of pi.
fn format_angle(angle: f64) -> String {
    let pi = std::f64::consts::PI;
    // Check common fractions of pi
    let fracs: &[(f64, &str)] = &[
        (1.0, "\\pi"),
        (0.5, "\\pi/2"),
        (0.25, "\\pi/4"),
        (0.125, "\\pi/8"),
        (-1.0, "-\\pi"),
        (-0.5, "-\\pi/2"),
        (-0.25, "-\\pi/4"),
    ];
    for &(frac, label) in fracs {
        if (angle - frac * pi).abs() < 1e-10 {
            return label.to_string();
        }
    }
    format!("{:.4}", angle)
}

/// Escape special characters for SVG text content.
fn svg_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Produce a short ASCII label for a gate name.
fn ascii_gate_label(name: &str) -> String {
    match name {
        "H" | "X" | "Y" | "Z" | "S" | "T" => name.to_string(),
        "R_z" => "Rz".to_string(),
        "R_x" => "Rx".to_string(),
        "R_y" => "Ry".to_string(),
        other => {
            if other.len() > 3 {
                other[..3].to_string()
            } else {
                other.to_string()
            }
        }
    }
}

/// Center a string inside a field of the given width, padding with dashes.
fn pad_center(s: &str, width: usize) -> String {
    if s.len() >= width {
        return s.to_string();
    }
    let total_pad = width - s.len();
    let left = total_pad / 2;
    let right = total_pad - left;
    format!("{}{}{}", "-".repeat(left), s, "-".repeat(right))
}

// ============================================================
// TESTS
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config tests ----

    #[test]
    fn test_export_config_defaults() {
        let cfg = ExportConfig::default();
        assert!(matches!(cfg.format, ExportFormat::Latex));
        assert!(cfg.show_parameters);
        assert!(cfg.show_measurements);
        assert!(!cfg.color_gates);
        assert!(!cfg.compact);
        assert!((cfg.gate_width_em - 1.5).abs() < f64::EPSILON);
        assert!((cfg.wire_spacing_em - 1.0).abs() < f64::EPSILON);
        assert!((cfg.font_size - 14.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_gates_per_row, 20);
    }

    #[test]
    fn test_export_config_builder() {
        let cfg = ExportConfig::default()
            .with_format(ExportFormat::Svg)
            .with_show_parameters(false)
            .with_compact(true)
            .with_font_size(18.0)
            .with_max_gates_per_row(10);
        assert!(matches!(cfg.format, ExportFormat::Svg));
        assert!(!cfg.show_parameters);
        assert!(cfg.compact);
        assert!((cfg.font_size - 18.0).abs() < f64::EPSILON);
        assert_eq!(cfg.max_gates_per_row, 10);
    }

    // ---- Circuit construction tests ----

    #[test]
    fn test_vis_circuit_construction() {
        let mut circ = VisCircuit::new(3);
        assert_eq!(circ.num_qubits, 3);
        assert_eq!(circ.num_cbits, 0);
        assert_eq!(circ.gates.len(), 0);
        assert_eq!(circ.qubit_labels, vec!["q_0", "q_1", "q_2"]);

        circ.h(0);
        circ.cx(0, 1);
        assert_eq!(circ.gates.len(), 2);
    }

    #[test]
    fn test_vis_circuit_with_classical() {
        let circ = VisCircuit::with_classical(2, 2);
        assert_eq!(circ.num_qubits, 2);
        assert_eq!(circ.num_cbits, 2);
        assert_eq!(circ.cbit_labels, vec!["c_0", "c_1"]);
    }

    #[test]
    fn test_vis_gate_qubits() {
        let g = VisGate::Single {
            name: "H".into(),
            qubit: 0,
            params: vec![],
        };
        assert_eq!(g.qubits(), vec![0]);

        let g = VisGate::Controlled {
            name: "X".into(),
            control: 0,
            target: 2,
            params: vec![],
        };
        assert_eq!(g.qubits(), vec![0, 2]);

        let g = VisGate::MultiControlled {
            name: "X".into(),
            controls: vec![0, 1],
            target: 2,
        };
        assert_eq!(g.qubits(), vec![0, 1, 2]);

        let g = VisGate::Swap { q1: 1, q2: 3 };
        assert_eq!(g.qubits(), vec![1, 3]);
    }

    // ---- LaTeX tests ----

    #[test]
    fn test_latex_single_qubit_gates() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        for name in &["H", "X", "Y", "Z", "S", "T"] {
            let mut circ = VisCircuit::new(1);
            circ.gates.push(VisGate::Single {
                name: name.to_string(),
                qubit: 0,
                params: vec![],
            });
            let latex = exporter.to_latex(&circ);
            assert!(
                latex.contains(&format!("\\gate{{{}}}", name)),
                "LaTeX should contain \\gate{{{}}} but got: {}",
                name,
                latex
            );
        }
    }

    #[test]
    fn test_latex_parameterized_gate() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.rz(0, std::f64::consts::FRAC_PI_4);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("R_z") && latex.contains("\\pi/4"),
            "Expected Rz with pi/4 parameter, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_parameterized_arbitrary_angle() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.rz(0, 1.234);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("1.2340"),
            "Expected numeric angle 1.2340, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_params_hidden_when_disabled() {
        let config = ExportConfig::default().with_show_parameters(false);
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.rz(0, std::f64::consts::FRAC_PI_2);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\gate{R_z}"),
            "With params hidden, gate should be just R_z, got: {}",
            latex
        );
        assert!(
            !latex.contains("\\pi"),
            "Should not contain pi when params disabled, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_cnot() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.cx(0, 1);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\ctrl{1}"),
            "Expected \\ctrl{{1}}, got: {}",
            latex
        );
        assert!(
            latex.contains("\\targ{}"),
            "Expected \\targ{{}}, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_cnot_reverse_direction() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.cx(1, 0); // Control below target
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\ctrl{-1}"),
            "Expected \\ctrl{{-1}} for upward CNOT, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_multi_controlled() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(3);
        circ.ccx(0, 1, 2); // Toffoli
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\ctrl{2}"),
            "Expected \\ctrl{{2}} for first control, got: {}",
            latex
        );
        assert!(
            latex.contains("\\ctrl{1}"),
            "Expected \\ctrl{{1}} for second control, got: {}",
            latex
        );
        assert!(
            latex.contains("\\targ{}"),
            "Expected \\targ{{}} for Toffoli target, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_swap() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.swap(0, 1);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\swap{1}"),
            "Expected \\swap{{1}}, got: {}",
            latex
        );
        assert!(
            latex.contains("\\targX{}"),
            "Expected \\targX{{}}, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_measurement() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.measure(0);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\meter{}"),
            "Expected \\meter{{}}, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_measurement_hidden() {
        let config = ExportConfig::default().with_show_measurements(false);
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.measure(0);
        let latex = exporter.to_latex(&circ);
        assert!(
            !latex.contains("\\meter"),
            "Measurement should be hidden, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_barrier() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.barrier(&[0, 1]);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\barrier{}"),
            "Expected barrier command, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_reset() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.reset(0);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\gate{\\ket{0}}"),
            "Expected reset gate, got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_full_circuit_bell_state() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.h(0);
        circ.cx(0, 1);
        let latex = exporter.to_latex(&circ);

        assert!(latex.contains("\\begin{quantikz}"));
        assert!(latex.contains("\\end{quantikz}"));
        assert!(latex.contains("\\gate{H}"));
        assert!(latex.contains("\\ctrl{1}"));
        assert!(latex.contains("\\targ{}"));
        assert!(latex.contains("\\lstick"));
        // Second wire should have \qw in the first column (H is on qubit 0)
        assert!(latex.contains("\\qw"));
    }

    #[test]
    fn test_latex_ghz_circuit() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        // 4-qubit GHZ: H on q0, then CNOT chain
        let mut circ = VisCircuit::new(4);
        circ.h(0);
        circ.cx(0, 1);
        circ.cx(1, 2);
        circ.cx(2, 3);
        let latex = exporter.to_latex(&circ);

        assert!(latex.contains("\\gate{H}"));
        // Should have three CNOT columns
        let ctrl_count = latex.matches("\\ctrl{1}").count();
        assert_eq!(ctrl_count, 3, "GHZ should have 3 \\ctrl{{1}} entries");
        let targ_count = latex.matches("\\targ{}").count();
        assert_eq!(targ_count, 3, "GHZ should have 3 \\targ{{}} entries");
    }

    #[test]
    fn test_latex_qubit_labels() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.set_qubit_labels(vec!["a".into(), "b".into()]);
        circ.h(0);
        let latex = exporter.to_latex(&circ);

        assert!(
            latex.contains("\\lstick{$a$}"),
            "Expected custom label 'a', got: {}",
            latex
        );
        assert!(
            latex.contains("\\lstick{$b$}"),
            "Expected custom label 'b', got: {}",
            latex
        );
    }

    #[test]
    fn test_latex_cz_gate() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.cz(0, 1);
        let latex = exporter.to_latex(&circ);
        assert!(
            latex.contains("\\ctrl{1}"),
            "CZ should have control dot, got: {}",
            latex
        );
        assert!(
            latex.contains("\\gate{Z}"),
            "CZ target should be a Z gate box, got: {}",
            latex
        );
    }

    // ---- SVG tests ----

    #[test]
    fn test_svg_basic_circuit() {
        let config = ExportConfig::default().with_format(ExportFormat::Svg);
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.h(0);
        circ.cx(0, 1);
        let svg = exporter.to_svg(&circ);

        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("q_0"));
        assert!(svg.contains("q_1"));
    }

    #[test]
    fn test_svg_contains_rect_and_line() {
        let config = ExportConfig::default().with_format(ExportFormat::Svg);
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.h(0);
        circ.cx(0, 1);
        let svg = exporter.to_svg(&circ);

        assert!(
            svg.contains("<rect"),
            "SVG should contain rect elements for gate boxes"
        );
        assert!(
            svg.contains("<line"),
            "SVG should contain line elements for wires"
        );
        assert!(
            svg.contains("<circle"),
            "SVG should contain circle for control dot"
        );
        assert!(
            svg.contains("<text"),
            "SVG should contain text elements for labels"
        );
    }

    #[test]
    fn test_svg_measurement_box() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.measure(0);
        let svg = exporter.to_svg(&circ);
        assert!(
            svg.contains("meas-box"),
            "SVG should contain measurement box class"
        );
    }

    #[test]
    fn test_svg_swap_marks() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.swap(0, 1);
        let svg = exporter.to_svg(&circ);
        // SWAP draws X marks = diagonal lines, at least 4 extra lines
        let line_count = svg.matches("<line").count();
        // 2 wire lines + 1 connector + 4 cross arms = 7 minimum
        assert!(
            line_count >= 7,
            "SWAP SVG should have at least 7 lines, got {}",
            line_count
        );
    }

    // ---- ASCII tests ----

    #[test]
    fn test_ascii_bell_state() {
        let config = ExportConfig::default().with_format(ExportFormat::Ascii);
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.h(0);
        circ.cx(0, 1);
        let ascii = exporter.to_ascii(&circ);

        // Should have two lines, one per qubit
        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 2);

        // First line should contain H and control marker
        assert!(
            lines[0].contains("H"),
            "First wire should show H gate, got: {}",
            lines[0]
        );
        assert!(
            lines[0].contains("*"),
            "First wire should show control dot, got: {}",
            lines[0]
        );
        // Second line should contain target marker
        assert!(
            lines[1].contains("X"),
            "Second wire should show CNOT target, got: {}",
            lines[1]
        );
    }

    #[test]
    fn test_ascii_measurement() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(1);
        circ.h(0);
        circ.measure(0);
        let ascii = exporter.to_ascii(&circ);
        assert!(
            ascii.contains("M"),
            "ASCII should show M for measurement, got: {}",
            ascii
        );
    }

    #[test]
    fn test_ascii_toffoli() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(3);
        circ.ccx(0, 1, 2);
        let ascii = exporter.to_ascii(&circ);

        let lines: Vec<&str> = ascii.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("*"), "Control 0 should have *, got: {}", lines[0]);
        assert!(lines[1].contains("*"), "Control 1 should have *, got: {}", lines[1]);
        assert!(lines[2].contains("X"), "Target should have X, got: {}", lines[2]);
    }

    // ---- Circuit with classical bits ----

    #[test]
    fn test_circuit_with_classical_bits() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::with_classical(2, 2);
        circ.h(0);
        circ.cx(0, 1);
        circ.measure_to(0, 0);
        circ.measure_to(1, 1);
        let latex = exporter.to_latex(&circ);

        // Should have 4 wires total (2 quantum + 2 classical)
        let row_ends = latex.matches("\\\\").count();
        // 3 row separators for 4 wires (last wire has no \\)
        assert_eq!(
            row_ends, 3,
            "Expected 3 row separators for 4 wires, got: {}",
            row_ends
        );
        assert!(latex.contains("\\cw"), "Classical wires should use \\cw");
    }

    // ---- Empty circuit ----

    #[test]
    fn test_empty_circuit() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let circ = VisCircuit::new(0);
        let latex = exporter.to_latex(&circ);
        assert!(latex.contains("\\begin{quantikz}"));
        assert!(latex.contains("\\end{quantikz}"));
    }

    #[test]
    fn test_empty_circuit_no_gates() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let circ = VisCircuit::new(2);
        let latex = exporter.to_latex(&circ);
        assert!(latex.contains("\\begin{quantikz}"));
        // Should still show wire labels and identity columns
        assert!(latex.contains("\\lstick"));
        assert!(latex.contains("\\qw"));
    }

    // ---- Format selection ----

    #[test]
    fn test_export_format_selection() {
        let mut circ = VisCircuit::new(1);
        circ.h(0);

        // LaTeX
        let exp = CircuitExporter::new(ExportConfig::default().with_format(ExportFormat::Latex));
        let out = exp.export(&circ).unwrap();
        assert!(out.contains("\\begin{quantikz}"));

        // SVG
        let exp = CircuitExporter::new(ExportConfig::default().with_format(ExportFormat::Svg));
        let out = exp.export(&circ).unwrap();
        assert!(out.starts_with("<svg"));

        // ASCII
        let exp = CircuitExporter::new(ExportConfig::default().with_format(ExportFormat::Ascii));
        let out = exp.export(&circ).unwrap();
        assert!(out.contains("H"));
    }

    // ---- Error handling ----

    #[test]
    fn test_error_display() {
        let e = ExportError::InvalidCircuit("bad qubit".into());
        assert_eq!(format!("{}", e), "Invalid circuit: bad qubit");

        let e = ExportError::RenderError("SVG failed".into());
        assert_eq!(format!("{}", e), "Render error: SVG failed");

        let e = ExportError::UnsupportedGate("FooGate".into());
        assert_eq!(format!("{}", e), "Unsupported gate: FooGate");
    }

    #[test]
    fn test_validation_catches_bad_qubit() {
        let config = ExportConfig::default();
        let exporter = CircuitExporter::new(config);

        let mut circ = VisCircuit::new(2);
        circ.gates.push(VisGate::Single {
            name: "H".into(),
            qubit: 5, // out of range
            params: vec![],
        });

        let result = exporter.export(&circ);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("qubit 5"));
    }

    // ---- Angle formatting ----

    #[test]
    fn test_format_angle_pi_fractions() {
        assert_eq!(format_angle(std::f64::consts::PI), "\\pi");
        assert_eq!(format_angle(std::f64::consts::FRAC_PI_2), "\\pi/2");
        assert_eq!(format_angle(std::f64::consts::FRAC_PI_4), "\\pi/4");
        assert_eq!(format_angle(-std::f64::consts::PI), "-\\pi");
        assert_eq!(format_angle(-std::f64::consts::FRAC_PI_2), "-\\pi/2");
    }

    #[test]
    fn test_format_angle_arbitrary() {
        let s = format_angle(1.0);
        assert_eq!(s, "1.0000");
    }

    // ---- Helper tests ----

    #[test]
    fn test_svg_escape() {
        assert_eq!(svg_escape("<a&b>"), "&lt;a&amp;b&gt;");
        assert_eq!(svg_escape("normal"), "normal");
    }

    #[test]
    fn test_pad_center() {
        assert_eq!(pad_center("H", 5), "--H--");
        assert_eq!(pad_center("Rz", 5), "-Rz--");
        assert_eq!(pad_center("ABCDE", 5), "ABCDE");
        assert_eq!(pad_center("ABCDEF", 5), "ABCDEF");
    }
}
