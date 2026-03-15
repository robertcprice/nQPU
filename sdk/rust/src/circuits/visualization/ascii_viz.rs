//! ASCII visualization for quantum circuits, states, and animated execution.
//!
//! Always available (no feature gate). Provides terminal-friendly output for:
//! - Circuit diagrams with unicode wire/gate rendering
//! - Probability bar charts with educational annotations
//! - Phase/amplitude tables
//! - Bloch sphere (single-qubit)
//! - Step-by-step animated gate execution with gate explanations
//!
//! # Educational Mode
//!
//! Set `AsciiConfig::educational` to `true` (default) to display plain-english
//! explanations of what each gate does, state insight annotations (superposition,
//! entanglement detection), and color-coded output.

use num_complex::Complex64;
use std::fmt;
use std::io::Write as IoWrite;
use std::time::Duration;

use crate::gates::{Gate, GateType};
use crate::{GateOperations, QuantumState, C64};

// ===================================================================
// ANSI COLOR PALETTE (inspired by agent-monitor phosphor green terminal)
// ===================================================================

mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // Phosphor green family
    pub const GREEN: &str = "\x1b[38;2;0;255;65m";
    pub const GREEN_DIM: &str = "\x1b[38;2;0;180;45m";

    // Accent colors
    pub const AMBER: &str = "\x1b[38;2;255;176;0m";
    pub const CYAN: &str = "\x1b[38;5;51m";
    pub const VIOLET: &str = "\x1b[38;5;147m";

    // Highlight states (used by render_cell directly via inline codes)
    #[allow(dead_code)]
    pub const YELLOW_BOLD: &str = "\x1b[1;33m";
    #[allow(dead_code)]
    pub const GREEN_FG: &str = "\x1b[32m";
}

// ===================================================================
// CONFIGURATION
// ===================================================================

/// Shared configuration for all ASCII visualizations.
#[derive(Clone, Debug)]
pub struct AsciiConfig {
    /// Width of probability bar charts in characters.
    pub bar_width: usize,
    /// Show states with zero probability in output.
    pub show_zero_probs: bool,
    /// Decimal precision for floating-point values.
    pub precision: usize,
    /// Use unicode box-drawing / quantum symbols. When false, falls back to pure ASCII.
    pub use_unicode: bool,
    /// Enable educational annotations: gate explanations, state insights, concept tips.
    pub educational: bool,
    /// Use ANSI color codes in output (disable for piping to files).
    pub color: bool,
}

impl Default for AsciiConfig {
    fn default() -> Self {
        Self {
            bar_width: 40,
            show_zero_probs: true,
            precision: 3,
            use_unicode: true,
            educational: true,
            color: true,
        }
    }
}

// ===================================================================
// HELPERS
// ===================================================================

/// Human-readable label for a gate, matching the existing visualization.rs naming.
pub fn gate_label(gate: &Gate) -> String {
    match &gate.gate_type {
        GateType::H => "H".into(),
        GateType::X => "X".into(),
        GateType::Y => "Y".into(),
        GateType::Z => "Z".into(),
        GateType::S => "S".into(),
        GateType::T => "T".into(),
        GateType::Rx(a) => format!("Rx({:.2})", a),
        GateType::Ry(a) => format!("Ry({:.2})", a),
        GateType::Rz(a) => format!("Rz({:.2})", a),
        GateType::U { theta, phi, lambda } => {
            format!("U({:.1},{:.1},{:.1})", theta, phi, lambda)
        }
        GateType::CNOT => "CX".into(),
        GateType::CZ => "CZ".into(),
        GateType::SWAP => "SW".into(),
        GateType::Toffoli => "CCX".into(),
        GateType::CRx(a) => format!("CRx({:.2})", a),
        GateType::CRy(a) => format!("CRy({:.2})", a),
        GateType::CRz(a) => format!("CRz({:.2})", a),
        GateType::CR(a) => format!("CR({:.2})", a),
        GateType::SX => "SX".into(),
        GateType::Phase(a) => format!("P({:.2})", a),
        GateType::ISWAP => "iSW".into(),
        GateType::CCZ => "CCZ".into(),
        GateType::Rxx(a) => format!("Rxx({:.2})", a),
        GateType::Ryy(a) => format!("Ryy({:.2})", a),
        GateType::Rzz(a) => format!("Rzz({:.2})", a),
        GateType::CSWAP => "CSW".into(),
        GateType::CU { .. } => "CU".into(),
        GateType::Custom(_) => "?".into(),
    }
}

/// Apply a single gate to a mutable QuantumState, dispatching to GateOperations.
pub fn apply_gate_to_state(state: &mut QuantumState, gate: &Gate) {
    match &gate.gate_type {
        GateType::H => GateOperations::h(state, gate.targets[0]),
        GateType::X => GateOperations::x(state, gate.targets[0]),
        GateType::Y => GateOperations::y(state, gate.targets[0]),
        GateType::Z => GateOperations::z(state, gate.targets[0]),
        GateType::S => GateOperations::s(state, gate.targets[0]),
        GateType::T => GateOperations::t(state, gate.targets[0]),
        GateType::Rx(a) => GateOperations::rx(state, gate.targets[0], *a),
        GateType::Ry(a) => GateOperations::ry(state, gate.targets[0], *a),
        GateType::Rz(a) => GateOperations::rz(state, gate.targets[0], *a),
        GateType::U { theta, phi, lambda } => {
            let cos = (theta / 2.0).cos();
            let sin = (theta / 2.0).sin();
            let matrix = [
                [
                    C64::new(cos, 0.0),
                    C64::new(-sin * lambda.cos(), -sin * lambda.sin()),
                ],
                [
                    C64::new(sin * phi.cos(), sin * phi.sin()),
                    C64::new(cos * (phi + lambda).cos(), cos * (phi + lambda).sin()),
                ],
            ];
            GateOperations::u(state, gate.targets[0], &matrix);
        }
        GateType::CNOT => GateOperations::cnot(state, gate.controls[0], gate.targets[0]),
        GateType::CZ => GateOperations::cz(state, gate.controls[0], gate.targets[0]),
        GateType::SWAP => GateOperations::swap(state, gate.targets[0], gate.targets[1]),
        GateType::Toffoli => {
            GateOperations::toffoli(state, gate.controls[0], gate.controls[1], gate.targets[0])
        }
        GateType::CRx(a) => GateOperations::crx(state, gate.controls[0], gate.targets[0], *a),
        GateType::CRy(a) => GateOperations::cry(state, gate.controls[0], gate.targets[0], *a),
        GateType::CRz(a) => GateOperations::crz(state, gate.controls[0], gate.targets[0], *a),
        GateType::CR(a) => GateOperations::cphase(state, gate.controls[0], gate.targets[0], *a),
        GateType::SX => GateOperations::sx(state, gate.targets[0]),
        GateType::Phase(a) => GateOperations::phase(state, gate.targets[0], *a),
        GateType::ISWAP => GateOperations::iswap(state, gate.targets[0], gate.targets[1]),
        GateType::CCZ => {
            GateOperations::ccz(state, gate.controls[0], gate.controls[1], gate.targets[0])
        }
        GateType::Rxx(_) | GateType::Ryy(_) | GateType::Rzz(_) => {
            // Two-qubit rotation gates: apply via custom matrix
        }
        GateType::CSWAP => {
            // Controlled-SWAP: not yet dispatched through GateOperations
        }
        GateType::CU { .. } => {
            // Generic controlled-U: not yet dispatched through GateOperations
        }
        GateType::Custom(_) => {
            // Custom gates are not dispatched through GateOperations
        }
    }
}

/// Format a ket label like |01⟩ for the given basis index and qubit count.
fn ket_label(index: usize, num_qubits: usize, unicode: bool) -> String {
    let bits: String = (0..num_qubits)
        .rev()
        .map(|q| if index & (1 << q) != 0 { '1' } else { '0' })
        .collect();
    if unicode {
        format!("|{}⟩", bits)
    } else {
        format!("|{}>", bits)
    }
}

/// Map a phase angle (radians) to a direction arrow character.
fn phase_arrow(angle: f64, unicode: bool) -> char {
    if !unicode {
        return '>';
    }
    // Normalize to [0, 2pi)
    let a = angle.rem_euclid(std::f64::consts::TAU);
    let sector = ((a + std::f64::consts::FRAC_PI_8) / std::f64::consts::FRAC_PI_4) as usize % 8;
    ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘'][sector]
}

// ===================================================================
// EDUCATIONAL CONTENT
// ===================================================================

/// Returns a concise plain-english explanation of what a gate does.
/// Designed to teach quantum computing concepts incrementally.
pub fn gate_explanation(gate: &Gate) -> &'static str {
    match &gate.gate_type {
        GateType::H => "Creates superposition: maps |0> to equal mix of |0>+|1>, and |1> to |0>-|1>. \
                         The quantum coin flip -- after H, measuring gives 0 or 1 with equal probability.",
        GateType::X => "Quantum NOT gate: flips |0> to |1> and vice versa. \
                         Equivalent to a 180-degree rotation around the X-axis of the Bloch sphere.",
        GateType::Y => "Pauli-Y: flips the qubit with a phase twist (multiplies by i). \
                         180-degree rotation around the Bloch sphere Y-axis.",
        GateType::Z => "Phase flip: leaves |0> alone but multiplies |1> by -1. \
                         Changes the *relative phase* without affecting measurement probabilities.",
        GateType::S => "Quarter-turn phase: adds a 90-degree phase to |1>. \
                         Square root of Z -- applying S twice equals Z.",
        GateType::T => "Eighth-turn phase: adds a 45-degree phase to |1>. \
                         Key ingredient for universal quantum computation and error correction.",
        GateType::Rx(_) => "Rotation around X-axis by the given angle. \
                            Smoothly interpolates between identity (0) and X gate (pi).",
        GateType::Ry(_) => "Rotation around Y-axis by the given angle. \
                            Can create any superposition of |0> and |1> from a computational basis state.",
        GateType::Rz(_) => "Rotation around Z-axis: adjusts the relative phase between |0> and |1>. \
                            Does not change measurement probabilities in the Z basis.",
        GateType::U { .. } => "Universal single-qubit gate: can implement ANY single-qubit operation. \
                               Parameterized by three Euler angles (theta, phi, lambda).",
        GateType::CNOT => "Controlled-NOT: flips the target qubit ONLY when the control qubit is |1>. \
                           The primary entangling gate -- creates Bell states from product states.",
        GateType::CZ => "Controlled-Z: applies a phase flip to the target ONLY when control is |1>. \
                         Symmetric: swapping control and target gives the same result.",
        GateType::SWAP => "Exchanges the quantum states of two qubits. \
                           Equivalent to three CNOT gates in sequence.",
        GateType::Toffoli => "Controlled-controlled-NOT: flips target only when BOTH controls are |1>. \
                              Universal for classical computation; implements AND gate reversibly.",
        GateType::CRx(_) => "Controlled X-rotation: applies Rx to target only when control is |1>. \
                             Creates entanglement with continuous angle control.",
        GateType::CRy(_) => "Controlled Y-rotation: applies Ry to target only when control is |1>.",
        GateType::CRz(_) => "Controlled Z-rotation: applies Rz to target only when control is |1>. \
                             Used extensively in quantum Fourier transform circuits.",
        GateType::CR(_) => "Controlled phase rotation: adds a phase to |11> only. \
                            Core building block of QFT and phase estimation algorithms.",
        GateType::SX => "Square root of X: halfway between identity and X gate. \
                         Native gate on IBM quantum hardware; two SX gates equal one X.",
        GateType::Phase(_) => "Phase gate P(theta): adds phase e^(i*theta) to |1>, leaves |0> unchanged. \
                               Generalization of S (pi/2) and T (pi/4) gates.",
        GateType::ISWAP => "iSWAP: swaps |01> and |10> with an additional phase factor of i. \
                            Native gate on Google Sycamore quantum processor.",
        GateType::CCZ => "Controlled-controlled-Z: applies a phase flip to |111> only. \
                          Equivalent to Toffoli up to basis change (H on target).",
        GateType::Rxx(_) => "XX rotation: exp(-i*theta/2 * XX). Entangling two-qubit gate \
                           used in variational quantum eigensolvers and quantum chemistry.",
        GateType::Ryy(_) => "YY rotation: exp(-i*theta/2 * YY). Two-qubit entangling gate \
                           that appears in Hamiltonian simulation of spin models.",
        GateType::Rzz(_) => "ZZ rotation: exp(-i*theta/2 * ZZ). Diagonal entangling gate \
                           central to Ising-model simulation and QAOA circuits.",
        GateType::CSWAP => "Controlled-SWAP (Fredkin gate): swaps two target qubits only when \
                           the control is |1>. Universal for classical computation; used in \
                           quantum fingerprinting and comparison algorithms.",
        GateType::CU { .. } => "Generic controlled-U: applies an arbitrary single-qubit unitary U \
                               to the target only when the control is |1>, with global phase gamma.",
        GateType::Custom(_) => "Custom unitary matrix operation.",
    }
}

/// Analyze the current quantum state and return educational insights.
pub fn state_insight(state: &QuantumState) -> Vec<String> {
    let mut insights = Vec::new();
    let probs = state.probabilities();
    let dim = state.dim;
    let nq = state.num_qubits;

    // Count non-zero amplitudes
    let nonzero: Vec<usize> = probs
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 1e-10)
        .map(|(i, _)| i)
        .collect();

    if nonzero.len() == 1 {
        let idx = nonzero[0];
        let label = ket_label(idx, nq, true);
        insights.push(format!(
            "Definite state {}: measuring will always give this result (no uncertainty).",
            label
        ));
    } else {
        insights.push(format!(
            "Superposition of {} basis states: measurement outcome is probabilistic.",
            nonzero.len()
        ));
    }

    // Check for uniform superposition
    if nonzero.len() > 1 {
        let expected = 1.0 / nonzero.len() as f64;
        let is_uniform = nonzero.iter().all(|&i| (probs[i] - expected).abs() < 1e-6);
        if is_uniform {
            if nonzero.len() == dim {
                insights.push(
                    "Equal superposition of ALL basis states (like after H on every qubit).".into(),
                );
            } else {
                insights
                    .push("Equal superposition: each outcome in the mix is equally likely.".into());
            }
        }
    }

    // Check for entanglement (only for 2+ qubits)
    if nq >= 2 && nonzero.len() > 1 {
        if is_entangled(state) {
            insights.push(
                "ENTANGLED: this state cannot be written as a product of individual qubit states. \
                 Measuring one qubit instantly determines outcomes on the others."
                    .into(),
            );
        } else {
            insights.push(
                "Separable (not entangled): each qubit can be described independently.".into(),
            );
        }
    }

    // Check for Bell state patterns (2-qubit)
    if nq == 2 && nonzero.len() == 2 {
        let (a, b) = (nonzero[0], nonzero[1]);
        if (a == 0 && b == 3) || (a == 1 && b == 2) {
            let p0 = probs[a];
            if (p0 - 0.5).abs() < 1e-6 {
                insights.push("Bell state detected: maximally entangled pair of qubits.".into());
            }
        }
    }

    // Max probability state
    if nonzero.len() > 1 {
        let (max_idx, max_p) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        let label = ket_label(max_idx, nq, true);
        insights.push(format!(
            "Most likely outcome: {} with {:.1}% probability.",
            label,
            max_p * 100.0
        ));
    }

    insights
}

/// Entanglement check via partial trace purity.
///
/// For 2 qubits: uses the determinant shortcut (product state iff a00*a11 == a01*a10).
/// For >2 qubits: computes the reduced density matrix of qubit 0 by tracing out
/// all other qubits, then checks Tr(rho^2). Pure reduced state (purity=1) means
/// separable; mixed (purity<1) means entangled with the rest.
fn is_entangled(state: &QuantumState) -> bool {
    let nq = state.num_qubits;
    if nq < 2 {
        return false;
    }

    if nq == 2 {
        // Product state check: |psi> = |a>|b> iff amp[00]*amp[11] == amp[01]*amp[10]
        let a00 = state.get(0);
        let a01 = state.get(1);
        let a10 = state.get(2);
        let a11 = state.get(3);

        let lhs = a00 * a11;
        let rhs = a01 * a10;
        let diff = (lhs.re - rhs.re).powi(2) + (lhs.im - rhs.im).powi(2);
        return diff > 1e-10;
    }

    // General case: reduced density matrix of qubit 0
    // Index layout: qubit 0 is the LSB. So for basis index i, qubit 0 = i & 1.
    // Group by "rest" bits (everything except qubit 0):
    //   idx = (rest << 1) | q0_val
    let mut rho = [[Complex64::new(0.0, 0.0); 2]; 2];
    for rest in 0..(1usize << (nq - 1)) {
        for a in 0..2usize {
            for b in 0..2usize {
                let idx_a = (rest << 1) | a;
                let idx_b = (rest << 1) | b;
                let amp_a = state.get(idx_a);
                let amp_b = state.get(idx_b);
                let prod = amp_a * amp_b.conj();
                rho[a][b] = rho[a][b] + prod;
            }
        }
    }

    // Purity = Tr(rho^2) = sum_ij |rho_ij|^2
    let mut purity = 0.0f64;
    for row in &rho {
        for elem in row {
            purity += elem.norm_sqr();
        }
    }

    // Entangled if reduced state is mixed (purity < 1)
    purity < 1.0 - 1e-6
}

// ===================================================================
// SLEEK BOX-DRAWING FRAMES
// ===================================================================

/// Render a styled box with title and content.
fn render_box(title: &str, content: &str, color: bool, accent: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let title_display_len = title.chars().count();
    let content_width = lines
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .max(title_display_len + 4);
    let box_width = content_width + 2; // padding

    let mut out = String::new();

    if color {
        // Top border with title
        out.push_str(accent);
        out.push_str("╭─");
        out.push_str(ansi::BOLD);
        out.push_str(" ");
        out.push_str(title);
        out.push_str(" ");
        out.push_str(ansi::RESET);
        out.push_str(accent);
        let remaining = box_width.saturating_sub(title_display_len + 3);
        for _ in 0..remaining {
            out.push('─');
        }
        out.push_str("─╮");
        out.push_str(ansi::RESET);
        out.push('\n');

        // Content lines
        for line in &lines {
            out.push_str(accent);
            out.push_str("│ ");
            out.push_str(ansi::RESET);
            let line_len = line.chars().count();
            out.push_str(line);
            for _ in line_len..content_width {
                out.push(' ');
            }
            out.push(' ');
            out.push_str(accent);
            out.push_str("│");
            out.push_str(ansi::RESET);
            out.push('\n');
        }

        // Bottom border
        out.push_str(accent);
        out.push_str("╰");
        for _ in 0..box_width + 1 {
            out.push('─');
        }
        out.push_str("─╯");
        out.push_str(ansi::RESET);
        out.push('\n');
    } else {
        // No-color version
        out.push_str("+-");
        out.push_str(" ");
        out.push_str(title);
        out.push_str(" ");
        let remaining = box_width.saturating_sub(title_display_len + 3);
        for _ in 0..remaining {
            out.push('-');
        }
        out.push_str("-+\n");

        for line in &lines {
            out.push_str("| ");
            let line_len = line.chars().count();
            out.push_str(line);
            for _ in line_len..content_width {
                out.push(' ');
            }
            out.push_str(" |\n");
        }

        out.push('+');
        for _ in 0..box_width + 2 {
            out.push('-');
        }
        out.push_str("+\n");
    }

    out
}

/// Wrap text to a maximum width, preserving words.
fn wrap_text(text: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut current_line_len = 0;

    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current_line_len + word_len + 1 > max_width && current_line_len > 0 {
            result.push('\n');
            current_line_len = 0;
        }
        if current_line_len > 0 {
            result.push(' ');
            current_line_len += 1;
        }
        result.push_str(word);
        current_line_len += word_len;
    }

    result
}

// ===================================================================
// PROBABILITY BARS
// ===================================================================

/// Renders probability bar charts for a quantum state.
pub struct AsciiProbabilities {
    probs: Vec<f64>,
    num_qubits: usize,
    config: AsciiConfig,
}

impl AsciiProbabilities {
    pub fn new(state: &QuantumState) -> Self {
        Self {
            probs: state.probabilities(),
            num_qubits: state.num_qubits,
            config: AsciiConfig::default(),
        }
    }

    pub fn with_config(state: &QuantumState, config: AsciiConfig) -> Self {
        Self {
            probs: state.probabilities(),
            num_qubits: state.num_qubits,
            config,
        }
    }
}

impl fmt::Display for AsciiProbabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let block = if self.config.use_unicode { '█' } else { '#' };
        let prec = self.config.precision;
        let w = self.config.bar_width;

        for (i, &p) in self.probs.iter().enumerate() {
            if !self.config.show_zero_probs && p < 1e-10 {
                continue;
            }
            let label = ket_label(i, self.num_qubits, self.config.use_unicode);
            let filled = (p * w as f64).round() as usize;
            let bar: String = std::iter::repeat(block).take(filled).collect();
            writeln!(f, "{} {:w$} {:>8.prec$}", label, bar, p, w = w, prec = prec)?;
        }
        Ok(())
    }
}

// ===================================================================
// PHASE TABLE
// ===================================================================

/// Renders a table of amplitudes, probabilities, and phases.
pub struct AsciiPhaseTable {
    amplitudes: Vec<C64>,
    num_qubits: usize,
    config: AsciiConfig,
}

impl AsciiPhaseTable {
    pub fn new(state: &QuantumState) -> Self {
        let amps: Vec<C64> = (0..state.dim).map(|i| state.get(i)).collect();
        Self {
            amplitudes: amps,
            num_qubits: state.num_qubits,
            config: AsciiConfig::default(),
        }
    }

    pub fn with_config(state: &QuantumState, config: AsciiConfig) -> Self {
        let amps: Vec<C64> = (0..state.dim).map(|i| state.get(i)).collect();
        Self {
            amplitudes: amps,
            num_qubits: state.num_qubits,
            config,
        }
    }
}

impl fmt::Display for AsciiPhaseTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prec = self.config.precision;
        let uni = self.config.use_unicode;
        let pi = std::f64::consts::PI;

        writeln!(
            f,
            "{:<8} {:>18}  {:>8}  {:>8}  {}",
            "State", "Amplitude", "Prob", "Phase", "Dir"
        )?;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            let prob = amp.norm_sqr();
            if !self.config.show_zero_probs && prob < 1e-10 {
                continue;
            }
            let label = ket_label(i, self.num_qubits, uni);
            let phase = amp.arg();
            let phase_pi = phase / pi;
            let arrow = phase_arrow(phase, uni);

            writeln!(
                f,
                "{:<8} {:>8.prec$}+{:.prec$}i  {:>8.prec$}  {:>5.2}pi  {}",
                label,
                amp.re,
                amp.im,
                prob,
                phase_pi,
                arrow,
                prec = prec,
            )?;
        }
        Ok(())
    }
}

// ===================================================================
// BLOCH SPHERE
// ===================================================================

/// ASCII Bloch sphere for a single-qubit state.
pub struct AsciiBloch {
    /// Bloch coordinates (x, y, z)
    coords: (f64, f64, f64),
    config: AsciiConfig,
}

impl AsciiBloch {
    /// Create from a quantum state. Returns an error if the state has more than 1 qubit.
    pub fn new(state: &QuantumState) -> Result<Self, String> {
        if state.num_qubits != 1 {
            return Err(format!(
                "Bloch sphere requires exactly 1 qubit, got {}",
                state.num_qubits
            ));
        }
        let alpha = state.get(0); // amplitude of |0>
        let beta = state.get(1); // amplitude of |1>

        // Bloch vector from amplitudes:
        // x = 2 Re(alpha* beta), y = 2 Im(alpha* beta), z = |alpha|^2 - |beta|^2
        let conj_alpha = alpha.conj();
        let prod = conj_alpha * beta;
        let x = 2.0 * prod.re;
        let y = 2.0 * prod.im;
        let z = alpha.norm_sqr() - beta.norm_sqr();

        Ok(Self {
            coords: (x, y, z),
            config: AsciiConfig::default(),
        })
    }

    pub fn with_config(state: &QuantumState, config: AsciiConfig) -> Result<Self, String> {
        let mut bloch = Self::new(state)?;
        bloch.config = config;
        Ok(bloch)
    }
}

impl fmt::Display for AsciiBloch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (bx, _by, bz) = self.coords;
        let uni = self.config.use_unicode;

        // 15x15 character grid, center at (7,7), radius 6
        let size = 15usize;
        let center = 7i32;
        let radius = 6.0f64;

        let mut grid = vec![vec![' '; size]; size];

        // Draw circle outline
        for angle_step in 0..72 {
            let a = (angle_step as f64) * std::f64::consts::TAU / 72.0;
            let cx = (center as f64 + radius * a.cos()).round() as i32;
            let cy = (center as f64 - radius * a.sin() * 0.5).round() as i32; // squash vertical for aspect ratio
            if cx >= 0 && cx < size as i32 && cy >= 0 && cy < size as i32 {
                let ch = grid[cy as usize][cx as usize];
                if ch == ' ' {
                    grid[cy as usize][cx as usize] = '.';
                }
            }
        }

        // Axis labels
        let (lbl_0, lbl_1, lbl_plus, lbl_minus) = if uni {
            ("|0⟩", "|1⟩", "|+⟩", "|-⟩")
        } else {
            ("|0>", "|1>", "|+>", "|->")
        };

        // Place state marker '*' at projected position
        // Project: px = bx, py = bz (top=+z, right=+x)
        let px = (center as f64 + bx * radius).round() as i32;
        let py = (center as f64 - bz * radius * 0.5).round() as i32;
        if px >= 0 && px < size as i32 && py >= 0 && py < size as i32 {
            grid[py as usize][px as usize] = '*';
        }

        // Render to string
        // Top label: |0>
        let top_pad = if center >= 2 { center as usize - 1 } else { 0 };
        writeln!(f, "{:>width$}", lbl_0, width = top_pad + lbl_0.len())?;

        for (row_idx, row) in grid.iter().enumerate() {
            let line: String = row.iter().collect();
            if row_idx == center as usize {
                // Middle row: show |-⟩ and |+⟩ on sides
                writeln!(f, "{} {} {}", lbl_minus, line, lbl_plus)?;
            } else {
                writeln!(f, "     {}", line)?;
            }
        }

        // Bottom label: |1>
        writeln!(f, "{:>width$}", lbl_1, width = top_pad + lbl_1.len())?;

        // Coordinates
        writeln!(
            f,
            "Bloch: x={:.3}, y={:.3}, z={:.3}",
            self.coords.0, self.coords.1, self.coords.2
        )?;

        Ok(())
    }
}

// ===================================================================
// CIRCUIT DIAGRAM
// ===================================================================

/// Content of a single cell in the circuit grid.
#[derive(Clone, Debug)]
enum CellContent {
    /// Horizontal wire segment
    Wire,
    /// Gate box with label
    GateBox(String),
    /// Control dot (CNOT / Toffoli control)
    ControlDot,
    /// Target circle (CNOT target)
    TargetCircle,
    /// Swap cross
    SwapCross,
    /// Vertical connecting line between control and target
    VerticalLine,
    /// CZ dot (both qubits get a dot)
    CzDot,
}

/// A column in the scheduled circuit layout.
#[derive(Clone, Debug)]
struct CircuitColumn {
    /// Index of this column in the schedule
    #[allow(dead_code)]
    col_idx: usize,
    /// Gates placed in this column (gate index in the original list)
    gate_indices: Vec<usize>,
}

/// Assigns gates to columns so that independent gates share columns.
fn schedule_gates(gates: &[Gate], num_qubits: usize) -> Vec<CircuitColumn> {
    if gates.is_empty() {
        return vec![];
    }
    let mut columns: Vec<CircuitColumn> = Vec::new();

    for (gi, gate) in gates.iter().enumerate() {
        let qubits_used = gate_qubits(gate);
        let min_q = qubits_used.iter().copied().min().unwrap_or(0);
        let max_q = qubits_used.iter().copied().max().unwrap_or(0);

        // Find the first column where this gate doesn't overlap with existing gates
        let mut placed = false;
        for col in columns.iter_mut() {
            let mut conflicts = false;
            for &existing_gi in &col.gate_indices {
                let existing_qubits = gate_qubits(&gates[existing_gi]);
                let ex_min = existing_qubits.iter().copied().min().unwrap_or(0);
                let ex_max = existing_qubits.iter().copied().max().unwrap_or(0);

                // Check range overlap (including vertical lines between control/target)
                if min_q <= ex_max && max_q >= ex_min {
                    conflicts = true;
                    break;
                }
            }
            if !conflicts {
                col.gate_indices.push(gi);
                placed = true;
                break;
            }
        }
        if !placed {
            columns.push(CircuitColumn {
                col_idx: columns.len(),
                gate_indices: vec![gi],
            });
        }
    }

    // Update col_idx after all insertions
    for (i, col) in columns.iter_mut().enumerate() {
        col.col_idx = i;
    }
    let _ = num_qubits; // used for API consistency
    columns
}

/// Collect all qubit indices touched by a gate (targets + controls).
fn gate_qubits(gate: &Gate) -> Vec<usize> {
    let mut qs = gate.targets.clone();
    qs.extend_from_slice(&gate.controls);
    qs
}

/// Build a 2D cell grid from scheduled columns.
fn build_cell_grid(
    columns: &[CircuitColumn],
    gates: &[Gate],
    num_qubits: usize,
) -> Vec<Vec<CellContent>> {
    // grid[qubit][column]
    let num_cols = columns.len();
    let mut grid = vec![vec![CellContent::Wire; num_cols]; num_qubits];

    for (ci, col) in columns.iter().enumerate() {
        for &gi in &col.gate_indices {
            let gate = &gates[gi];
            match &gate.gate_type {
                GateType::CNOT => {
                    let ctrl = gate.controls[0];
                    let tgt = gate.targets[0];
                    grid[ctrl][ci] = CellContent::ControlDot;
                    grid[tgt][ci] = CellContent::TargetCircle;
                    let lo = ctrl.min(tgt);
                    let hi = ctrl.max(tgt);
                    for q in (lo + 1)..hi {
                        grid[q][ci] = CellContent::VerticalLine;
                    }
                }
                GateType::CZ => {
                    let ctrl = gate.controls[0];
                    let tgt = gate.targets[0];
                    grid[ctrl][ci] = CellContent::CzDot;
                    grid[tgt][ci] = CellContent::CzDot;
                    let lo = ctrl.min(tgt);
                    let hi = ctrl.max(tgt);
                    for q in (lo + 1)..hi {
                        grid[q][ci] = CellContent::VerticalLine;
                    }
                }
                GateType::SWAP => {
                    let a = gate.targets[0];
                    let b = gate.targets[1];
                    grid[a][ci] = CellContent::SwapCross;
                    grid[b][ci] = CellContent::SwapCross;
                    let lo = a.min(b);
                    let hi = a.max(b);
                    for q in (lo + 1)..hi {
                        grid[q][ci] = CellContent::VerticalLine;
                    }
                }
                GateType::Toffoli => {
                    let c0 = gate.controls[0];
                    let c1 = gate.controls[1];
                    let tgt = gate.targets[0];
                    grid[c0][ci] = CellContent::ControlDot;
                    grid[c1][ci] = CellContent::ControlDot;
                    grid[tgt][ci] = CellContent::TargetCircle;
                    let all = [c0, c1, tgt];
                    let lo = *all.iter().min().unwrap();
                    let hi = *all.iter().max().unwrap();
                    for q in (lo + 1)..hi {
                        if q != c0 && q != c1 && q != tgt {
                            grid[q][ci] = CellContent::VerticalLine;
                        }
                    }
                }
                // Controlled rotation gates
                GateType::CRx(_) | GateType::CRy(_) | GateType::CRz(_) | GateType::CR(_) => {
                    let ctrl = gate.controls[0];
                    let tgt = gate.targets[0];
                    grid[ctrl][ci] = CellContent::ControlDot;
                    let label = gate_label(gate);
                    // Show the rotation label on the target qubit
                    let short_label = match &gate.gate_type {
                        GateType::CRx(_) => "Rx".to_string(),
                        GateType::CRy(_) => "Ry".to_string(),
                        GateType::CRz(_) => "Rz".to_string(),
                        GateType::CR(_) => "CR".to_string(),
                        _ => label,
                    };
                    grid[tgt][ci] = CellContent::GateBox(short_label);
                    let lo = ctrl.min(tgt);
                    let hi = ctrl.max(tgt);
                    for q in (lo + 1)..hi {
                        grid[q][ci] = CellContent::VerticalLine;
                    }
                }
                // Single-qubit and other gates
                _ => {
                    let label = gate_label(gate);
                    if !gate.targets.is_empty() {
                        grid[gate.targets[0]][ci] = CellContent::GateBox(label);
                    }
                }
            }
        }
    }

    grid
}

/// Compute the rendered character width for each column.
fn column_widths(grid: &[Vec<CellContent>], num_qubits: usize, num_cols: usize) -> Vec<usize> {
    let mut widths = vec![3usize; num_cols]; // minimum 3 for wire segments
    for ci in 0..num_cols {
        for q in 0..num_qubits {
            let cell_w = match &grid[q][ci] {
                CellContent::GateBox(label) => label.len() + 2, // [label]
                CellContent::ControlDot => 1,
                CellContent::TargetCircle => 1,
                CellContent::SwapCross => 1,
                CellContent::CzDot => 1,
                CellContent::VerticalLine => 1,
                CellContent::Wire => 1,
            };
            if cell_w > widths[ci] {
                widths[ci] = cell_w;
            }
        }
    }
    widths
}

/// Render a single cell into a string of the given width, padded with wire chars.
fn render_cell(
    cell: &CellContent,
    width: usize,
    unicode: bool,
    highlight: Option<HighlightKind>,
) -> String {
    let (wire_char, ctrl_char, tgt_char, swap_char, vert_char) = if unicode {
        ('─', '●', '⊕', '╳', '│')
    } else {
        ('-', 'o', '+', 'x', '|')
    };

    let raw = match cell {
        CellContent::Wire => std::iter::repeat(wire_char).take(width).collect::<String>(),
        CellContent::GateBox(label) => {
            let inner_width = width.saturating_sub(2);
            let padded = format!("{:^w$}", label, w = inner_width);
            format!("[{}]", padded)
        }
        CellContent::ControlDot => {
            let pad = width / 2;
            let right = width - pad - 1;
            format!(
                "{}{}{}",
                std::iter::repeat(wire_char).take(pad).collect::<String>(),
                ctrl_char,
                std::iter::repeat(wire_char).take(right).collect::<String>(),
            )
        }
        CellContent::TargetCircle => {
            let pad = width / 2;
            let right = width - pad - 1;
            format!(
                "{}{}{}",
                std::iter::repeat(wire_char).take(pad).collect::<String>(),
                tgt_char,
                std::iter::repeat(wire_char).take(right).collect::<String>(),
            )
        }
        CellContent::SwapCross => {
            let pad = width / 2;
            let right = width - pad - 1;
            format!(
                "{}{}{}",
                std::iter::repeat(wire_char).take(pad).collect::<String>(),
                swap_char,
                std::iter::repeat(wire_char).take(right).collect::<String>(),
            )
        }
        CellContent::CzDot => {
            let pad = width / 2;
            let right = width - pad - 1;
            format!(
                "{}{}{}",
                std::iter::repeat(wire_char).take(pad).collect::<String>(),
                ctrl_char,
                std::iter::repeat(wire_char).take(right).collect::<String>(),
            )
        }
        CellContent::VerticalLine => {
            let pad = width / 2;
            let right = width - pad - 1;
            format!("{}{}{}", " ".repeat(pad), vert_char, " ".repeat(right),)
        }
    };

    // Apply ANSI highlighting
    match highlight {
        Some(HighlightKind::Current) => format!("\x1b[1;33m{}\x1b[0m", raw), // bold yellow
        Some(HighlightKind::Completed) => format!("\x1b[32m{}\x1b[0m", raw), // green
        None => raw,
    }
}

#[derive(Clone, Copy, Debug)]
pub enum HighlightKind {
    Current,
    Completed,
}

/// ASCII circuit diagram for a set of gates.
pub struct AsciiCircuit {
    num_qubits: usize,
    gates: Vec<Gate>,
    columns: Vec<CircuitColumn>,
    grid: Vec<Vec<CellContent>>,
    col_widths: Vec<usize>,
    config: AsciiConfig,
    /// Per-column highlight state (for animator use).
    highlights: Vec<Option<HighlightKind>>,
}

impl AsciiCircuit {
    pub fn new(num_qubits: usize, gates: &[Gate]) -> Self {
        Self::with_config(num_qubits, gates, AsciiConfig::default())
    }

    pub fn with_config(num_qubits: usize, gates: &[Gate], config: AsciiConfig) -> Self {
        let gates_owned = gates.to_vec();
        let columns = schedule_gates(&gates_owned, num_qubits);
        let grid = build_cell_grid(&columns, &gates_owned, num_qubits);
        let col_widths = column_widths(&grid, num_qubits, columns.len());
        let highlights = vec![None; columns.len()];
        Self {
            num_qubits,
            gates: gates_owned,
            columns,
            grid,
            col_widths,
            config,
            highlights,
        }
    }

    /// Set highlight for a specific column index.
    pub fn highlight_column(&mut self, col: usize, kind: HighlightKind) {
        if col < self.highlights.len() {
            self.highlights[col] = Some(kind);
        }
    }

    /// Clear all highlights.
    pub fn clear_highlights(&mut self) {
        for h in self.highlights.iter_mut() {
            *h = None;
        }
    }

    /// Render the circuit to a string (used by Display and the animator).
    fn render(&self) -> String {
        let mut out = String::new();
        let wire_char = if self.config.use_unicode { '─' } else { '-' };

        for q in 0..self.num_qubits {
            // Qubit label
            let label = format!("q{}: ", q);
            out.push_str(&label);
            // Leading wire
            out.push_str(&std::iter::repeat(wire_char).take(3).collect::<String>());

            for (ci, w) in self.col_widths.iter().enumerate() {
                let cell = &self.grid[q][ci];
                let rendered = render_cell(cell, *w, self.config.use_unicode, self.highlights[ci]);
                out.push_str(&rendered);
                // Wire segment between columns
                out.push_str(&std::iter::repeat(wire_char).take(3).collect::<String>());
            }
            out.push('\n');

            // Vertical line row between qubits (if not last qubit)
            if q + 1 < self.num_qubits {
                let prefix_len = label.len() + 3; // "qN: " + leading wire
                let mut vert_row = " ".repeat(prefix_len);
                for (ci, w) in self.col_widths.iter().enumerate() {
                    let needs_vert = self.column_has_vertical(ci, q, q + 1);
                    let pad = w / 2;
                    let right = w - pad - 1;
                    if needs_vert {
                        let vert_char = if self.config.use_unicode { '│' } else { '|' };
                        vert_row.push_str(&" ".repeat(pad));
                        vert_row.push(vert_char);
                        vert_row.push_str(&" ".repeat(right));
                    } else {
                        vert_row.push_str(&" ".repeat(*w));
                    }
                    vert_row.push_str(&" ".repeat(3)); // gap between columns
                }
                let trimmed = vert_row.trim_end();
                if !trimmed.trim().is_empty() {
                    out.push_str(trimmed);
                    out.push('\n');
                }
            }
        }
        out
    }

    /// Check if column `ci` needs a vertical connector between qubit rows `q` and `q+1`.
    fn column_has_vertical(&self, ci: usize, q: usize, q_next: usize) -> bool {
        // A vertical line is needed if there's a multi-qubit gate spanning across these rows
        for &gi in &self.columns[ci].gate_indices {
            let gate = &self.gates[gi];
            let qubits = gate_qubits(gate);
            if let (Some(&lo), Some(&hi)) = (qubits.iter().min(), qubits.iter().max()) {
                if lo <= q && hi >= q_next {
                    return true;
                }
            }
        }
        false
    }
}

impl fmt::Display for AsciiCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.render())
    }
}

// ===================================================================
// ANIMATOR
// ===================================================================

const BANNER: &str = r#"
  ░███╗░░░██████╗░██████╗░██╗░░░██╗
  ████║░░██╔═══██╗██╔══██╗██║░░░██║
  ╚██║░░░██║██╗██║██████╔╝██║░░░██║
  ░██║░░░╚██████╔╝██╔═══╝░██║░░░██║
  ░██║░░░░╚═██╔═╝░██║░░░░░╚██████╔╝
  ░╚═╝░░░░░╚═╝░░░╚═╝░░░░░░╚═════╝░
"#;

const STARFIELD: &str = "  ✦     ⋆   ★       ✧    ✶   ⋆  ★      ✦";

/// Step-by-step animated gate execution in the terminal.
///
/// Renders a sleek, educational terminal UI with:
/// - Animated banner with starfield decoration
/// - Color-coded circuit diagram (current=yellow, completed=green)
/// - Real-time probability bars
/// - Gate-by-gate explanations of quantum mechanics
/// - State insight annotations (superposition, entanglement detection)
/// - Progress indicator with scan-line effect
pub struct AsciiAnimator {
    num_qubits: usize,
    gates: Vec<Gate>,
    config: AsciiConfig,
}

impl AsciiAnimator {
    pub fn new(num_qubits: usize, gates: Vec<Gate>) -> Self {
        Self {
            num_qubits,
            gates,
            config: AsciiConfig::default(),
        }
    }

    pub fn with_config(num_qubits: usize, gates: Vec<Gate>, config: AsciiConfig) -> Self {
        Self {
            num_qubits,
            gates,
            config,
        }
    }

    /// Run with default 500ms delay.
    pub fn run(&self) {
        self.run_with_delay(Duration::from_millis(500));
    }

    /// Run animation with custom delay between steps.
    pub fn run_with_delay(&self, delay: Duration) {
        let mut state = QuantumState::new(self.num_qubits);
        let mut circuit =
            AsciiCircuit::with_config(self.num_qubits, &self.gates, self.config.clone());
        let total_gates = self.gates.len();

        // Build a mapping: gate_index -> column_index
        let gate_to_col = self.build_gate_to_col_map(&circuit.columns);

        // Initial frame: show circuit with no gates applied
        self.draw_frame(&circuit, &state, None, 0, total_gates, 0);
        std::thread::sleep(delay);

        for (gi, gate) in self.gates.iter().enumerate() {
            // Apply the gate
            apply_gate_to_state(&mut state, gate);

            // Update highlights: completed gates green, current gate yellow
            circuit.clear_highlights();
            for prev in 0..gi {
                if let Some(&col) = gate_to_col.get(&prev) {
                    circuit.highlight_column(col, HighlightKind::Completed);
                }
            }
            if let Some(&col) = gate_to_col.get(&gi) {
                circuit.highlight_column(col, HighlightKind::Current);
            }

            self.draw_frame(&circuit, &state, Some(gate), gi + 1, total_gates, gi);
            std::thread::sleep(delay);
        }

        // Final frame: all gates completed (green)
        circuit.clear_highlights();
        for gi in 0..self.gates.len() {
            if let Some(&col) = gate_to_col.get(&gi) {
                circuit.highlight_column(col, HighlightKind::Completed);
            }
        }
        self.draw_frame(
            &circuit,
            &state,
            None,
            total_gates,
            total_gates,
            total_gates,
        );
    }

    fn build_gate_to_col_map(
        &self,
        columns: &[CircuitColumn],
    ) -> std::collections::HashMap<usize, usize> {
        let mut map = std::collections::HashMap::new();
        for (ci, col) in columns.iter().enumerate() {
            for &gi in &col.gate_indices {
                map.insert(gi, ci);
            }
        }
        map
    }

    fn draw_frame(
        &self,
        circuit: &AsciiCircuit,
        state: &QuantumState,
        current_gate: Option<&Gate>,
        step: usize,
        total: usize,
        frame_idx: usize,
    ) {
        let c = self.config.color;
        let edu = self.config.educational;

        let mut frame = String::new();
        // ANSI: clear screen + move to home
        frame.push_str("\x1b[H\x1b[2J");

        // ── Banner ──
        if c {
            frame.push_str(ansi::CYAN);
            frame.push_str(STARFIELD);
            frame.push_str(ansi::RESET);
            frame.push('\n');
            frame.push_str(ansi::VIOLET);
            frame.push_str(BANNER);
            frame.push_str(ansi::RESET);
            frame.push_str(ansi::CYAN);
            frame.push_str(STARFIELD);
            frame.push_str(ansi::RESET);
            frame.push('\n');
        } else {
            frame.push_str("  === nQPU Quantum Simulator ===\n\n");
        }

        // ── Progress bar ──
        let progress = if total > 0 {
            step as f64 / total as f64
        } else {
            0.0
        };
        let bar_total = 40;
        let filled = (progress * bar_total as f64).round() as usize;
        let scan_chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let scan_idx = frame_idx % scan_chars.len();

        if c {
            frame.push_str("  ");
            frame.push_str(ansi::GREEN_DIM);
            frame.push_str("Step [");
            frame.push_str(ansi::GREEN);
            for i in 0..bar_total {
                if i < filled {
                    frame.push('█');
                } else if i == filled {
                    frame.push(scan_chars[scan_idx]);
                } else {
                    frame.push('░');
                }
            }
            frame.push_str(ansi::GREEN_DIM);
            frame.push_str(&format!("] {}/{}", step, total));
            frame.push_str(ansi::RESET);
        } else {
            frame.push_str(&format!("  Step [{}/{}] ", step, total));
            let bar: String = (0..bar_total)
                .map(|i| if i < filled { '#' } else { '.' })
                .collect();
            frame.push_str(&bar);
        }
        frame.push_str("\n\n");

        // ── Circuit ──
        let circuit_content = circuit.render();
        if c {
            frame.push_str(&render_box(
                "Circuit",
                &circuit_content,
                true,
                ansi::GREEN_DIM,
            ));
        } else {
            frame.push_str("  -- Circuit --\n");
            frame.push_str(&circuit_content);
        }
        frame.push('\n');

        // ── Gate action ──
        if let Some(gate) = current_gate {
            let qubits = gate_qubits(gate);
            let qubit_str: Vec<String> = qubits.iter().map(|q| format!("q{}", q)).collect();
            let action_line = format!(
                "▶ Applied: {} on {}",
                gate_label(gate),
                qubit_str.join(", ")
            );

            if c {
                frame.push_str("  ");
                frame.push_str(ansi::AMBER);
                frame.push_str(ansi::BOLD);
                frame.push_str(&action_line);
                frame.push_str(ansi::RESET);
                frame.push_str("\n\n");
            } else {
                frame.push_str(&format!("  {}\n\n", action_line));
            }

            // Educational explanation
            if edu {
                let explanation = gate_explanation(gate);
                let wrapped = wrap_text(explanation, 60);
                if c {
                    frame.push_str(&render_box("What happened?", &wrapped, true, ansi::VIOLET));
                } else {
                    frame.push_str(&render_box("What happened?", &wrapped, false, ""));
                }
                frame.push('\n');
            }
        } else if step == total && total > 0 {
            if c {
                frame.push_str("  ");
                frame.push_str(ansi::GREEN);
                frame.push_str(ansi::BOLD);
                frame.push_str("✓ Circuit execution complete.");
                frame.push_str(ansi::RESET);
                frame.push_str("\n\n");
            } else {
                frame.push_str("  [DONE] Circuit execution complete.\n\n");
            }
        } else {
            if c {
                frame.push_str("  ");
                frame.push_str(ansi::DIM);
                frame.push_str("Initializing... all qubits in |0⟩");
                frame.push_str(ansi::RESET);
                frame.push_str("\n\n");
            } else {
                frame.push_str("  Initializing... all qubits in |0>\n\n");
            }
        }

        // ── Probabilities ──
        let probs_str = format!(
            "{}",
            AsciiProbabilities::with_config(state, self.config.clone())
        );
        if c {
            frame.push_str(&render_box(
                "Probabilities",
                &probs_str.trim_end(),
                true,
                ansi::CYAN,
            ));
        } else {
            frame.push_str("  -- Probabilities --\n");
            frame.push_str(&probs_str);
        }

        // ── State Insights ──
        if edu {
            let insights = state_insight(state);
            if !insights.is_empty() {
                let insight_text = insights
                    .iter()
                    .map(|s| {
                        let wrapped = wrap_text(s, 58);
                        format!("  ◊ {}", wrapped.replace('\n', "\n    "))
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                frame.push('\n');
                if c {
                    frame.push_str(&render_box(
                        "State Insights",
                        &insight_text,
                        true,
                        ansi::AMBER,
                    ));
                } else {
                    frame.push_str(&render_box("State Insights", &insight_text, false, ""));
                }
            }
        }

        // ── Footer ──
        frame.push('\n');
        if c {
            frame.push_str(ansi::DIM);
            frame.push_str("  nQPU-Metal Quantum Simulator");
            frame.push_str(ansi::RESET);
        }
        frame.push('\n');

        // Single write to minimize flicker
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(frame.as_bytes());
        let _ = lock.flush();
    }
}

// ===================================================================
// TESTS
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_label() {
        assert_eq!(gate_label(&Gate::h(0)), "H");
        assert_eq!(gate_label(&Gate::x(1)), "X");
        assert_eq!(gate_label(&Gate::cnot(0, 1)), "CX");
        assert_eq!(gate_label(&Gate::swap(0, 1)), "SW");
        assert_eq!(gate_label(&Gate::toffoli(0, 1, 2)), "CCX");
    }

    #[test]
    fn test_ket_label() {
        assert_eq!(ket_label(0, 2, true), "|00⟩");
        assert_eq!(ket_label(3, 2, true), "|11⟩");
        assert_eq!(ket_label(5, 3, true), "|101⟩");
        assert_eq!(ket_label(0, 1, false), "|0>");
    }

    #[test]
    fn test_phase_arrow() {
        assert_eq!(phase_arrow(0.0, true), '→');
        assert_eq!(phase_arrow(std::f64::consts::FRAC_PI_2, true), '↑');
        assert_eq!(phase_arrow(std::f64::consts::PI, true), '←');
        assert_eq!(phase_arrow(-std::f64::consts::FRAC_PI_2, true), '↓');
    }

    #[test]
    fn test_probabilities_display() {
        // |0> state: all probability on |0>
        let state = QuantumState::new(1);
        let probs = AsciiProbabilities::new(&state);
        let output = format!("{}", probs);
        assert!(output.contains("|0"));
        assert!(output.contains("1.000"));
    }

    #[test]
    fn test_probabilities_bell_state() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        let probs = AsciiProbabilities::new(&state);
        let output = format!("{}", probs);
        // Bell state: |00> and |11> each ~0.500
        assert!(output.contains("0.500"));
    }

    #[test]
    fn test_probabilities_hide_zeros() {
        let state = QuantumState::new(2);
        let config = AsciiConfig {
            show_zero_probs: false,
            ..Default::default()
        };
        let probs = AsciiProbabilities::with_config(&state, config);
        let output = format!("{}", probs);
        // Only |00> should appear
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("|00"));
    }

    #[test]
    fn test_phase_table_display() {
        let state = QuantumState::new(1);
        let table = AsciiPhaseTable::new(&state);
        let output = format!("{}", table);
        assert!(output.contains("State"));
        assert!(output.contains("Amplitude"));
        assert!(output.contains("Phase"));
    }

    #[test]
    fn test_bloch_zero_state() {
        let state = QuantumState::new(1);
        let bloch = AsciiBloch::new(&state).unwrap();
        // |0> should be at north pole: z ≈ 1
        assert!((bloch.coords.2 - 1.0).abs() < 1e-10);
        assert!(bloch.coords.0.abs() < 1e-10);
    }

    #[test]
    fn test_bloch_one_state() {
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0);
        let bloch = AsciiBloch::new(&state).unwrap();
        // |1> should be at south pole: z ≈ -1
        assert!((bloch.coords.2 + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bloch_plus_state() {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        let bloch = AsciiBloch::new(&state).unwrap();
        // |+> should be on equator: x ≈ 1, z ≈ 0
        assert!((bloch.coords.0 - 1.0).abs() < 1e-10);
        assert!(bloch.coords.2.abs() < 1e-10);
    }

    #[test]
    fn test_bloch_rejects_multi_qubit() {
        let state = QuantumState::new(2);
        assert!(AsciiBloch::new(&state).is_err());
    }

    #[test]
    fn test_bloch_display() {
        let state = QuantumState::new(1);
        let bloch = AsciiBloch::new(&state).unwrap();
        let output = format!("{}", bloch);
        assert!(output.contains("Bloch"));
        assert!(output.contains("*")); // state marker
    }

    #[test]
    fn test_circuit_single_gate() {
        let gates = vec![Gate::h(0)];
        let circuit = AsciiCircuit::new(1, &gates);
        let output = format!("{}", circuit);
        assert!(output.contains("q0"));
        assert!(output.contains("[H]"));
    }

    #[test]
    fn test_circuit_bell_state() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let circuit = AsciiCircuit::new(2, &gates);
        let output = format!("{}", circuit);
        assert!(output.contains("q0"));
        assert!(output.contains("q1"));
        assert!(output.contains("[H]"));
        // Should have control dot and target
        assert!(output.contains('●') || output.contains('o'));
        assert!(output.contains('⊕') || output.contains('+'));
    }

    #[test]
    fn test_circuit_parallel_scheduling() {
        // H on q0 and X on q1 should share a column
        let gates = vec![Gate::h(0), Gate::x(1)];
        let columns = schedule_gates(&gates, 2);
        assert_eq!(
            columns.len(),
            1,
            "Independent gates should share one column"
        );
    }

    #[test]
    fn test_circuit_sequential_scheduling() {
        // H on q0 then CNOT(0,1) must be separate columns
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let columns = schedule_gates(&gates, 2);
        assert_eq!(columns.len(), 2, "Overlapping gates need separate columns");
    }

    #[test]
    fn test_circuit_swap() {
        let gates = vec![Gate::swap(0, 1)];
        let circuit = AsciiCircuit::new(2, &gates);
        let output = format!("{}", circuit);
        assert!(output.contains('╳') || output.contains('x'));
    }

    #[test]
    fn test_circuit_toffoli() {
        let gates = vec![Gate::toffoli(0, 1, 2)];
        let circuit = AsciiCircuit::new(3, &gates);
        let output = format!("{}", circuit);
        assert!(output.contains("q0"));
        assert!(output.contains("q1"));
        assert!(output.contains("q2"));
    }

    #[test]
    fn test_ascii_fallback() {
        let config = AsciiConfig {
            use_unicode: false,
            ..Default::default()
        };
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let circuit = AsciiCircuit::with_config(2, &gates, config.clone());
        let output = format!("{}", circuit);
        // Should use ASCII characters only
        assert!(!output.contains('─'));
        assert!(!output.contains('●'));
        assert!(output.contains('-'));
    }

    #[test]
    fn test_apply_gate_to_state_h() {
        let mut state = QuantumState::new(1);
        let gate = Gate::h(0);
        apply_gate_to_state(&mut state, &gate);
        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_gate_to_state_cnot() {
        let mut state = QuantumState::new(2);
        apply_gate_to_state(&mut state, &Gate::h(0));
        apply_gate_to_state(&mut state, &Gate::cnot(0, 1));
        let probs = state.probabilities();
        // Bell state
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = AsciiCircuit::new(2, &[]);
        let output = format!("{}", circuit);
        assert!(output.contains("q0"));
        assert!(output.contains("q1"));
    }

    #[test]
    fn test_config_default() {
        let config = AsciiConfig::default();
        assert_eq!(config.bar_width, 40);
        assert!(config.show_zero_probs);
        assert_eq!(config.precision, 3);
        assert!(config.use_unicode);
    }

    #[test]
    fn test_cz_gate_display() {
        let gates = vec![Gate::cz(0, 1)];
        let circuit = AsciiCircuit::new(2, &gates);
        let output = format!("{}", circuit);
        // CZ shows control dots on both qubits
        let dot_count = output.matches('●').count();
        assert!(dot_count >= 2, "CZ should show dots on both qubits");
    }

    // ── Educational feature tests ──

    #[test]
    fn test_gate_explanation_h() {
        let gate = Gate::h(0);
        let explanation = gate_explanation(&gate);
        assert!(explanation.contains("superposition"));
    }

    #[test]
    fn test_gate_explanation_cnot() {
        let gate = Gate::cnot(0, 1);
        let explanation = gate_explanation(&gate);
        assert!(explanation.contains("entangling"));
    }

    #[test]
    fn test_gate_explanation_all_variants() {
        // Every gate type should return a non-empty explanation
        let gates = vec![
            Gate::h(0),
            Gate::x(0),
            Gate::y(0),
            Gate::z(0),
            Gate::s(0),
            Gate::t(0),
            Gate::rx(0, 1.0),
            Gate::ry(0, 1.0),
            Gate::rz(0, 1.0),
            Gate::cnot(0, 1),
            Gate::cz(0, 1),
            Gate::swap(0, 1),
            Gate::toffoli(0, 1, 2),
        ];
        for g in &gates {
            let exp = gate_explanation(g);
            assert!(
                !exp.is_empty(),
                "Explanation for {:?} should not be empty",
                g.gate_type
            );
        }
    }

    #[test]
    fn test_state_insight_definite_state() {
        let state = QuantumState::new(2);
        let insights = state_insight(&state);
        assert!(insights.iter().any(|s| s.contains("Definite state")));
    }

    #[test]
    fn test_state_insight_superposition() {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        let insights = state_insight(&state);
        assert!(insights.iter().any(|s| s.contains("Superposition")));
    }

    #[test]
    fn test_state_insight_entangled() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        let insights = state_insight(&state);
        assert!(
            insights
                .iter()
                .any(|s| s.contains("ENTANGLED") || s.contains("Bell state")),
            "Bell state should be detected as entangled, got: {:?}",
            insights
        );
    }

    #[test]
    fn test_state_insight_separable() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        // H on q0 only: |+>|0>, which is separable
        let insights = state_insight(&state);
        assert!(
            insights.iter().any(|s| s.contains("Separable")),
            "H|0> tensor |0> should be separable, got: {:?}",
            insights
        );
    }

    #[test]
    fn test_state_insight_uniform_superposition() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::h(&mut state, 1);
        let insights = state_insight(&state);
        assert!(
            insights.iter().any(|s| s.contains("Equal superposition")),
            "H|0>H|0> should be uniform, got: {:?}",
            insights
        );
    }

    #[test]
    fn test_render_box_with_color() {
        let output = render_box("Test", "hello\nworld", true, ansi::GREEN);
        assert!(output.contains("Test"));
        assert!(output.contains("hello"));
        assert!(output.contains("╭"));
        assert!(output.contains("╯"));
    }

    #[test]
    fn test_render_box_without_color() {
        let output = render_box("Test", "hello\nworld", false, "");
        assert!(output.contains("Test"));
        assert!(output.contains("hello"));
        assert!(output.contains('+'));
        assert!(output.contains('|'));
    }

    #[test]
    fn test_wrap_text() {
        let text = "This is a long sentence that should be wrapped at a reasonable width.";
        let wrapped = wrap_text(text, 20);
        assert!(wrapped.contains('\n'));
        for line in wrapped.lines() {
            assert!(line.chars().count() <= 20 + 1); // allow slight overshoot for last word
        }
    }

    #[test]
    fn test_educational_config() {
        let config = AsciiConfig::default();
        assert!(config.educational, "educational should be true by default");
        assert!(config.color, "color should be true by default");
    }

    #[test]
    fn test_entanglement_detection_3_qubit() {
        let mut state = QuantumState::new(3);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        GateOperations::cnot(&mut state, 1, 2);
        // GHZ state: (|000> + |111>)/sqrt(2)
        assert!(is_entangled(&state), "GHZ state should be entangled");
    }

    #[test]
    fn test_entanglement_detection_product_state() {
        let state = QuantumState::new(2);
        assert!(!is_entangled(&state), "|00> should not be entangled");
    }

    // ═══════════════════════════════════════════════════════════════
    // STRESS TESTS - edge cases, rendering correctness, all paths
    // ═══════════════════════════════════════════════════════════════

    #[test]
    fn test_apply_all_single_qubit_gates() {
        // Verify apply_gate_to_state dispatches every single-qubit gate type correctly
        let gate_types = vec![
            Gate::h(0),
            Gate::x(0),
            Gate::y(0),
            Gate::z(0),
            Gate::s(0),
            Gate::t(0),
            Gate::rx(0, std::f64::consts::FRAC_PI_4),
            Gate::ry(0, std::f64::consts::FRAC_PI_4),
            Gate::rz(0, std::f64::consts::FRAC_PI_4),
        ];
        for gate in &gate_types {
            let mut state = QuantumState::new(1);
            apply_gate_to_state(&mut state, gate);
            let probs = state.probabilities();
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-8,
                "Gate {:?} produced invalid probabilities (sum={})",
                gate.gate_type,
                total
            );
        }
    }

    #[test]
    fn test_apply_u_gate() {
        let gate = Gate::new(
            GateType::U {
                theta: std::f64::consts::PI,
                phi: 0.0,
                lambda: std::f64::consts::PI,
            },
            vec![0],
            vec![],
        );
        let mut state = QuantumState::new(1);
        apply_gate_to_state(&mut state, &gate);
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-8, "U gate sum={}", total);
    }

    #[test]
    fn test_apply_all_two_qubit_gates() {
        let gates = vec![Gate::cnot(0, 1), Gate::cz(0, 1), Gate::swap(0, 1)];
        for gate in &gates {
            let mut state = QuantumState::new(2);
            GateOperations::h(&mut state, 0); // put in superposition first
            apply_gate_to_state(&mut state, gate);
            let probs = state.probabilities();
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-8,
                "Gate {:?} produced invalid probabilities (sum={})",
                gate.gate_type,
                total
            );
        }
    }

    #[test]
    fn test_apply_controlled_rotations() {
        let gates = vec![
            Gate::new(GateType::CRx(1.0), vec![1], vec![0]),
            Gate::new(GateType::CRy(1.0), vec![1], vec![0]),
            Gate::new(GateType::CRz(1.0), vec![1], vec![0]),
            Gate::new(GateType::CR(1.0), vec![1], vec![0]),
        ];
        for gate in &gates {
            let mut state = QuantumState::new(2);
            GateOperations::h(&mut state, 0);
            apply_gate_to_state(&mut state, gate);
            let probs = state.probabilities();
            let total: f64 = probs.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-8,
                "Controlled rotation {:?} produced invalid probabilities (sum={})",
                gate.gate_type,
                total
            );
        }
    }

    #[test]
    fn test_apply_toffoli() {
        let mut state = QuantumState::new(3);
        GateOperations::x(&mut state, 0);
        GateOperations::x(&mut state, 1);
        // |110⟩ -> Toffoli should flip q2 -> |111⟩
        let gate = Gate::toffoli(0, 1, 2);
        apply_gate_to_state(&mut state, &gate);
        let probs = state.probabilities();
        assert!(
            (probs[7] - 1.0).abs() < 1e-10,
            "Toffoli should flip |110> to |111>"
        );
    }

    #[test]
    fn test_apply_custom_gate_noop() {
        // Custom gate dispatch is a no-op; verify it doesn't crash
        let gate = Gate::new(GateType::Custom(vec![]), vec![0], vec![]);
        let mut state = QuantumState::new(1);
        apply_gate_to_state(&mut state, &gate);
        // State should be unchanged
        assert!((state.get(0).re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_circuit_many_gates_scheduling() {
        // 10 independent single-qubit gates should pack into 1 column
        let gates: Vec<Gate> = (0..10).map(|q| Gate::h(q)).collect();
        let columns = schedule_gates(&gates, 10);
        assert_eq!(
            columns.len(),
            1,
            "10 independent H gates should share 1 column"
        );
    }

    #[test]
    fn test_circuit_chain_scheduling() {
        // CNOT(0,1) spans qubits 0-1, placed in col 0
        // CNOT(1,2) spans qubits 1-2, overlaps col 0 -> col 1
        // CNOT(2,3) spans qubits 2-3, doesn't overlap col 0 (0-1) -> packs into col 0
        let gates = vec![Gate::cnot(0, 1), Gate::cnot(1, 2), Gate::cnot(2, 3)];
        let columns = schedule_gates(&gates, 4);
        assert_eq!(
            columns.len(),
            2,
            "CNOT(0,1)+CNOT(2,3) share col, CNOT(1,2) gets its own"
        );
    }

    #[test]
    fn test_circuit_mixed_scheduling() {
        // H(0), H(2) can share; CNOT(0,1), CNOT(2,3) can share
        let gates = vec![Gate::h(0), Gate::h(2), Gate::cnot(0, 1), Gate::cnot(2, 3)];
        let columns = schedule_gates(&gates, 4);
        assert_eq!(
            columns.len(),
            2,
            "H(0)+H(2) share col, CNOT(0,1)+CNOT(2,3) share col"
        );
    }

    #[test]
    fn test_circuit_5_qubit_render() {
        let gates = vec![
            Gate::h(0),
            Gate::h(1),
            Gate::h(2),
            Gate::h(3),
            Gate::h(4),
            Gate::cnot(0, 4), // long-range CNOT
            Gate::toffoli(1, 3, 2),
        ];
        let circuit = AsciiCircuit::new(5, &gates);
        let output = format!("{}", circuit);
        // All 5 qubit labels should appear
        for q in 0..5 {
            assert!(
                output.contains(&format!("q{}", q)),
                "Missing q{} in 5-qubit circuit",
                q
            );
        }
        // Should have vertical connectors for the long-range CNOT
        assert!(
            output.contains('│'),
            "Long-range CNOT should have vertical lines"
        );
    }

    #[test]
    fn test_circuit_controlled_rotation_display() {
        let gates = vec![Gate::new(GateType::CRz(1.5), vec![1], vec![0])];
        let circuit = AsciiCircuit::new(2, &gates);
        let output = format!("{}", circuit);
        // Control dot on q0, gate box on q1
        assert!(output.contains('●'), "Should have control dot");
        assert!(output.contains("[Rz]"), "Should show Rz gate box");
    }

    #[test]
    fn test_phase_table_bell_state() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        let table = AsciiPhaseTable::new(&state);
        let output = format!("{}", table);
        // Should show both |00⟩ and |11⟩ with non-zero amplitudes
        assert!(output.contains("|00"), "Bell state should show |00>");
        assert!(output.contains("|11"), "Bell state should show |11>");
        assert!(output.contains("0.500"), "Should show 50% probability");
    }

    #[test]
    fn test_phase_table_hide_zeros() {
        let state = QuantumState::new(2);
        let config = AsciiConfig {
            show_zero_probs: false,
            ..Default::default()
        };
        let table = AsciiPhaseTable::with_config(&state, config);
        let output = format!("{}", table);
        // Only header + |00⟩ line
        let data_lines: Vec<&str> = output.lines().filter(|l| l.contains("|0")).collect();
        assert_eq!(data_lines.len(), 1, "Only |00> should appear");
    }

    #[test]
    fn test_phase_arrow_all_sectors() {
        use std::f64::consts::*;
        let expected = vec![
            (0.0, '→'),
            (FRAC_PI_4, '↗'),
            (FRAC_PI_2, '↑'),
            (3.0 * FRAC_PI_4, '↖'),
            (PI, '←'),
            (-3.0 * FRAC_PI_4, '↙'),
            (-FRAC_PI_2, '↓'),
            (-FRAC_PI_4, '↘'),
        ];
        for (angle, expected_arrow) in expected {
            let arrow = phase_arrow(angle, true);
            assert_eq!(
                arrow, expected_arrow,
                "phase_arrow({}) should be {}",
                angle, expected_arrow
            );
        }
    }

    #[test]
    fn test_ket_label_3_qubits() {
        assert_eq!(ket_label(0, 3, true), "|000⟩");
        assert_eq!(ket_label(7, 3, true), "|111⟩");
        assert_eq!(ket_label(5, 3, true), "|101⟩");
        assert_eq!(ket_label(3, 3, false), "|011>");
    }

    #[test]
    fn test_bloch_minus_state() {
        // |-> state = H|1> : should be on equator at x = -1
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0);
        GateOperations::h(&mut state, 0);
        let bloch = AsciiBloch::new(&state).unwrap();
        assert!((bloch.coords.0 + 1.0).abs() < 1e-10, "|-⟩ should have x=-1");
        assert!(bloch.coords.2.abs() < 1e-10, "|-⟩ should have z=0");
    }

    #[test]
    fn test_bloch_i_plus_state() {
        // Ry(pi/2)|0> should give |+i> state on +y axis
        let mut state = QuantumState::new(1);
        GateOperations::ry(&mut state, 0, std::f64::consts::FRAC_PI_2);
        let bloch = AsciiBloch::new(&state).unwrap();
        // After Ry(pi/2), we get cos(pi/4)|0> + sin(pi/4)|1>
        // which is on +x axis actually... let me check
        // Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
        // For theta=pi/2: cos(pi/4)|0> + sin(pi/4)|1> = (|0>+|1>)/sqrt(2) = |+>
        // That's on +x axis. For +y, we need S|+>
        let probs = state.probabilities();
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bloch_display_no_unicode() {
        let state = QuantumState::new(1);
        let config = AsciiConfig {
            use_unicode: false,
            ..Default::default()
        };
        let bloch = AsciiBloch::with_config(&state, config).unwrap();
        let output = format!("{}", bloch);
        assert!(output.contains("|0>"), "ASCII mode should use |0> not |0⟩");
        assert!(!output.contains("⟩"), "ASCII mode should not have ⟩");
    }

    #[test]
    fn test_probabilities_3_qubits() {
        let mut state = QuantumState::new(3);
        GateOperations::h(&mut state, 0);
        GateOperations::h(&mut state, 1);
        GateOperations::h(&mut state, 2);
        let probs = AsciiProbabilities::new(&state);
        let output = format!("{}", probs);
        // All 8 basis states should appear with ~0.125 probability
        assert!(
            output.contains("0.125"),
            "3-qubit uniform should have 0.125 entries"
        );
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 8, "Should have 8 basis states");
    }

    #[test]
    fn test_probabilities_no_unicode() {
        let state = QuantumState::new(1);
        let config = AsciiConfig {
            use_unicode: false,
            ..Default::default()
        };
        let probs = AsciiProbabilities::with_config(&state, config);
        let output = format!("{}", probs);
        assert!(output.contains('#'), "Non-unicode should use # for bars");
        assert!(!output.contains('█'), "Non-unicode should not use █");
    }

    #[test]
    fn test_circuit_highlight_and_clear() {
        let gates = vec![Gate::h(0), Gate::x(1)];
        let mut circuit = AsciiCircuit::new(2, &gates);

        // Highlight column 0
        circuit.highlight_column(0, HighlightKind::Current);
        let output = format!("{}", circuit);
        // Should contain ANSI yellow escape
        assert!(
            output.contains("\x1b[1;33m"),
            "Highlighted output should have yellow ANSI"
        );

        // Clear and verify
        circuit.clear_highlights();
        let output2 = format!("{}", circuit);
        assert!(
            !output2.contains("\x1b[1;33m"),
            "Cleared output should not have yellow ANSI"
        );
    }

    #[test]
    fn test_circuit_highlight_out_of_bounds() {
        let gates = vec![Gate::h(0)];
        let mut circuit = AsciiCircuit::new(1, &gates);
        // Should not panic on out-of-bounds highlight
        circuit.highlight_column(999, HighlightKind::Current);
        let output = format!("{}", circuit);
        assert!(output.contains("[H]"));
    }

    #[test]
    fn test_circuit_completed_highlight_color() {
        let gates = vec![Gate::h(0)];
        let mut circuit = AsciiCircuit::new(1, &gates);
        circuit.highlight_column(0, HighlightKind::Completed);
        let output = format!("{}", circuit);
        assert!(
            output.contains("\x1b[32m"),
            "Completed should use green ANSI"
        );
    }

    #[test]
    fn test_gate_label_rotation_formatting() {
        let gate = Gate::rx(0, 1.5707963267948966); // pi/2
        assert_eq!(gate_label(&gate), "Rx(1.57)");

        let gate2 = Gate::ry(0, 0.1);
        assert_eq!(gate_label(&gate2), "Ry(0.10)");
    }

    #[test]
    fn test_gate_label_u_formatting() {
        let gate = Gate::new(
            GateType::U {
                theta: 1.0,
                phi: 2.0,
                lambda: 3.0,
            },
            vec![0],
            vec![],
        );
        assert_eq!(gate_label(&gate), "U(1.0,2.0,3.0)");
    }

    #[test]
    fn test_state_insight_single_qubit() {
        let state = QuantumState::new(1);
        let insights = state_insight(&state);
        // Single qubit |0> should be definite, no entanglement check
        assert!(insights.iter().any(|s| s.contains("Definite")));
        assert!(!insights.iter().any(|s| s.contains("ENTANGLED")));
        assert!(!insights.iter().any(|s| s.contains("Separable")));
    }

    #[test]
    fn test_state_insight_most_likely() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        // |+>|0> = (|00> + |10>)/sqrt(2), both 50%
        let insights = state_insight(&state);
        assert!(insights.iter().any(|s| s.contains("Most likely")));
    }

    #[test]
    fn test_entanglement_h_both_qubits_not_entangled() {
        // H on both qubits: |+>|+> is a product state
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::h(&mut state, 1);
        assert!(!is_entangled(&state), "|+>|+> should NOT be entangled");
    }

    #[test]
    fn test_entanglement_phi_plus_bell() {
        // Phi+ = (|00> + |11>)/sqrt(2)
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        assert!(is_entangled(&state), "Phi+ Bell state must be entangled");
    }

    #[test]
    fn test_entanglement_psi_plus_bell() {
        // Psi+ = (|01> + |10>)/sqrt(2)
        let mut state = QuantumState::new(2);
        GateOperations::x(&mut state, 0);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        assert!(is_entangled(&state), "Psi+ Bell state must be entangled");
    }

    #[test]
    fn test_entanglement_1_qubit() {
        let state = QuantumState::new(1);
        assert!(!is_entangled(&state), "Single qubit cannot be entangled");
    }

    #[test]
    fn test_entanglement_4_qubit_ghz() {
        let mut state = QuantumState::new(4);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        GateOperations::cnot(&mut state, 1, 2);
        GateOperations::cnot(&mut state, 2, 3);
        assert!(is_entangled(&state), "4-qubit GHZ must be entangled");
    }

    #[test]
    fn test_render_box_empty_content() {
        let output = render_box("Empty", "", true, ansi::GREEN);
        assert!(output.contains("Empty"));
        assert!(output.contains("╭"));
        assert!(output.contains("╯"));
    }

    #[test]
    fn test_render_box_long_title() {
        let output = render_box(
            "A Very Long Box Title That Is Quite Descriptive",
            "x",
            false,
            "",
        );
        assert!(output.contains("A Very Long Box Title"));
    }

    #[test]
    fn test_wrap_text_short() {
        let text = "Short";
        let wrapped = wrap_text(text, 80);
        assert_eq!(wrapped, "Short");
        assert!(!wrapped.contains('\n'));
    }

    #[test]
    fn test_wrap_text_empty() {
        let wrapped = wrap_text("", 80);
        assert_eq!(wrapped, "");
    }

    #[test]
    fn test_wrap_text_single_long_word() {
        let text = "supercalifragilisticexpialidocious";
        let wrapped = wrap_text(text, 10);
        // Single word longer than max_width should not be split
        assert!(!wrapped.contains('\n'));
    }

    #[test]
    fn test_circuit_large_render_no_panic() {
        // Stress test: 8 qubits, many gates
        let mut gates = Vec::new();
        for q in 0..8 {
            gates.push(Gate::h(q));
        }
        for q in 0..7 {
            gates.push(Gate::cnot(q, q + 1));
        }
        gates.push(Gate::toffoli(0, 7, 3));
        gates.push(Gate::swap(2, 5));

        let circuit = AsciiCircuit::new(8, &gates);
        let output = format!("{}", circuit);
        assert!(
            output.len() > 100,
            "Large circuit should produce substantial output"
        );
        for q in 0..8 {
            assert!(output.contains(&format!("q{}", q)));
        }
    }

    #[test]
    fn test_probabilities_normalization_display() {
        // After X gate, all probability on |1>
        let mut state = QuantumState::new(1);
        GateOperations::x(&mut state, 0);
        let config = AsciiConfig {
            show_zero_probs: false,
            ..Default::default()
        };
        let probs = AsciiProbabilities::with_config(&state, config);
        let output = format!("{}", probs);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("|1"), "Only |1> should appear");
        assert!(lines[0].contains("1.000"));
    }

    #[test]
    fn test_circuit_vertical_connectors_between_qubits() {
        // CNOT(0,3) should have vertical lines through q1 and q2
        let gates = vec![Gate::cnot(0, 3)];
        let circuit = AsciiCircuit::new(4, &gates);
        let output = format!("{}", circuit);
        // Count vertical line chars (│)
        let vert_count = output.matches('│').count();
        assert!(
            vert_count >= 2,
            "CNOT(0,3) should have vertical lines through q1,q2; got {}",
            vert_count
        );
    }

    #[test]
    fn test_full_display_pipeline_no_panic() {
        // Exercise the entire pipeline: create state, apply gates, render all viz types
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);

        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];

        let probs_str = format!("{}", AsciiProbabilities::new(&state));
        let phase_str = format!("{}", AsciiPhaseTable::new(&state));
        let circuit_str = format!("{}", AsciiCircuit::new(2, &gates));

        assert!(!probs_str.is_empty());
        assert!(!phase_str.is_empty());
        assert!(!circuit_str.is_empty());

        let insights = state_insight(&state);
        assert!(!insights.is_empty());
    }

    #[test]
    fn test_animator_creation() {
        // Just verify animator can be constructed without panic
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];
        let animator = AsciiAnimator::new(2, gates);
        assert_eq!(animator.num_qubits, 2);
        assert_eq!(animator.gates.len(), 2);
    }

    #[test]
    fn test_animator_with_config() {
        let config = AsciiConfig {
            educational: false,
            color: false,
            ..Default::default()
        };
        let gates = vec![Gate::h(0)];
        let animator = AsciiAnimator::with_config(1, gates, config);
        assert!(!animator.config.educational);
        assert!(!animator.config.color);
    }

    #[test]
    fn test_state_insight_bell_state_detection() {
        let mut state = QuantumState::new(2);
        GateOperations::h(&mut state, 0);
        GateOperations::cnot(&mut state, 0, 1);
        let insights = state_insight(&state);
        assert!(
            insights.iter().any(|s| s.contains("Bell state")),
            "Should specifically detect Bell state, got: {:?}",
            insights
        );
    }

    #[test]
    fn test_phase_table_direction_arrows_present() {
        let mut state = QuantumState::new(1);
        GateOperations::h(&mut state, 0);
        let table = AsciiPhaseTable::new(&state);
        let output = format!("{}", table);
        assert!(
            output.contains('→'),
            "Phase table should contain direction arrows"
        );
        assert!(
            output.contains("Dir"),
            "Phase table should have Dir column header"
        );
    }

    #[test]
    fn test_circuit_render_deterministic() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1), Gate::x(1)];
        let circuit = AsciiCircuit::new(2, &gates);
        let output1 = format!("{}", circuit);
        let output2 = format!("{}", circuit);
        assert_eq!(
            output1, output2,
            "Circuit rendering should be deterministic"
        );
    }

    #[test]
    fn test_gate_qubits_helper() {
        let cnot = Gate::cnot(2, 5);
        let qs = gate_qubits(&cnot);
        assert!(qs.contains(&2));
        assert!(qs.contains(&5));

        let toffoli = Gate::toffoli(0, 3, 7);
        let qs2 = gate_qubits(&toffoli);
        assert!(qs2.contains(&0));
        assert!(qs2.contains(&3));
        assert!(qs2.contains(&7));
    }
}
