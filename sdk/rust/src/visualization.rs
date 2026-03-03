//! Visualization Tools for Quantum Circuits and States
//!
//! This module provides visualization capabilities for quantum circuits,
//! state vectors, and measurement results using plotters.

#[cfg(feature = "visualization")]
use {plotters::prelude::*, plotters::style::RGBColor, std::path::Path};

use crate::gates::{Gate, GateType};
use crate::QuantumState;

/// Color as (r, g, b) tuple for use in both visualization and non-visualization builds
type Rgb = (u8, u8, u8);

/// Visual configuration for circuit diagrams
#[derive(Clone, Debug)]
pub struct CircuitVisualStyle {
    pub width: u32,
    pub height: u32,
    pub background_color: Rgb,
    pub line_color: Rgb,
    pub text_color: Rgb,
    pub gate_colors: GateColorScheme,
}

/// Color scheme for different gate types
#[derive(Clone, Debug)]
pub enum GateColorScheme {
    Uniform(Rgb),
    ByGateType(Vec<(String, Rgb)>),
}

impl Default for CircuitVisualStyle {
    fn default() -> Self {
        CircuitVisualStyle {
            width: 800,
            height: 400,
            background_color: (255, 255, 255),
            line_color: (0, 0, 0),
            text_color: (0, 0, 0),
            gate_colors: GateColorScheme::ByGateType(vec![
                ("H".to_string(), (100, 150, 255)),
                ("X".to_string(), (255, 100, 100)),
                ("Y".to_string(), (100, 255, 100)),
                ("Z".to_string(), (255, 255, 100)),
                ("S".to_string(), (255, 150, 200)),
                ("T".to_string(), (200, 150, 255)),
                ("CNOT".to_string(), (50, 50, 50)),
                ("CZ".to_string(), (100, 100, 200)),
                ("SWAP".to_string(), (200, 100, 50)),
                ("Rx".to_string(), (150, 200, 255)),
                ("Ry".to_string(), (200, 255, 150)),
                ("Rz".to_string(), (255, 200, 150)),
            ]),
        }
    }
}

#[cfg(feature = "visualization")]
fn rgb(c: Rgb) -> RGBColor {
    RGBColor(c.0, c.1, c.2)
}

/// Circuit diagram generator
pub struct CircuitDiagram {
    pub num_qubits: usize,
    pub gates: Vec<Gate>,
    style: CircuitVisualStyle,
}

impl CircuitDiagram {
    /// Create a new circuit diagram
    pub fn new(num_qubits: usize, gates: Vec<Gate>) -> Self {
        CircuitDiagram {
            num_qubits,
            gates,
            style: CircuitVisualStyle::default(),
        }
    }

    /// Set the visual style
    pub fn with_style(mut self, style: CircuitVisualStyle) -> Self {
        self.style = style;
        self
    }

    /// Save the circuit diagram to a file
    #[cfg(feature = "visualization")]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path.as_ref(), (self.style.width, self.style.height))
            .into_drawing_area();

        root.fill(&rgb(self.style.background_color))?;

        let num_cols = self.gates.len() + 2;
        let row_height = self.style.height as i32 / (self.num_qubits as i32 + 1);
        let col_width = self.style.width as i32 / num_cols as i32;

        let line_color = rgb(self.style.line_color);
        let text_color = rgb(self.style.text_color);

        // Draw qubit lines and labels
        for qubit in 0..self.num_qubits {
            let y = row_height * (qubit as i32 + 1);
            let x_start = col_width;
            let x_end = self.style.width as i32 - col_width / 2;

            // Qubit wire
            root.draw(&PathElement::new(
                vec![(x_start, y), (x_end, y)],
                line_color.stroke_width(1),
            ))?;

            // Label
            root.draw(&Text::new(
                format!("q{}", qubit),
                (5, y - 8),
                ("sans-serif", 14).into_font().color(&text_color),
            ))?;
        }

        // Draw gates
        for (col, gate) in self.gates.iter().enumerate() {
            let x = col_width * (col as i32 + 1) + col_width / 2;
            let gate_name = self.gate_name(gate);
            let color = self.gate_color(gate);
            let gate_rgb = rgb(color);

            match &gate.gate_type {
                GateType::CNOT => {
                    let control = gate.controls.get(0).copied().unwrap_or(0);
                    let target = gate.targets.get(0).copied().unwrap_or(0);
                    let cy = row_height * (control as i32 + 1);
                    let ty = row_height * (target as i32 + 1);

                    // Connecting line
                    root.draw(&PathElement::new(
                        vec![(x, cy), (x, ty)],
                        gate_rgb.stroke_width(2),
                    ))?;

                    // Control dot
                    root.draw(&Circle::new((x, cy), 4, gate_rgb.filled()))?;

                    // Target circle (⊕)
                    root.draw(&Circle::new((x, ty), 10, gate_rgb.stroke_width(2)))?;
                    root.draw(&PathElement::new(
                        vec![(x - 10, ty), (x + 10, ty)],
                        gate_rgb.stroke_width(2),
                    ))?;
                    root.draw(&PathElement::new(
                        vec![(x, ty - 10), (x, ty + 10)],
                        gate_rgb.stroke_width(2),
                    ))?;
                }
                GateType::CZ => {
                    let control = gate.controls.get(0).copied().unwrap_or(0);
                    let target = gate.targets.get(0).copied().unwrap_or(0);
                    let cy = row_height * (control as i32 + 1);
                    let ty = row_height * (target as i32 + 1);

                    // Connecting line
                    root.draw(&PathElement::new(
                        vec![(x, cy), (x, ty)],
                        gate_rgb.stroke_width(2),
                    ))?;

                    // Both dots
                    root.draw(&Circle::new((x, cy), 4, gate_rgb.filled()))?;
                    root.draw(&Circle::new((x, ty), 4, gate_rgb.filled()))?;
                }
                GateType::SWAP => {
                    let a = gate.targets.get(0).copied().unwrap_or(0);
                    let b = gate.targets.get(1).copied().unwrap_or(0);
                    let ya = row_height * (a as i32 + 1);
                    let yb = row_height * (b as i32 + 1);

                    // Connecting line
                    root.draw(&PathElement::new(
                        vec![(x, ya), (x, yb)],
                        gate_rgb.stroke_width(2),
                    ))?;

                    // X marks at each qubit
                    for &qy in &[ya, yb] {
                        root.draw(&PathElement::new(
                            vec![(x - 6, qy - 6), (x + 6, qy + 6)],
                            gate_rgb.stroke_width(2),
                        ))?;
                        root.draw(&PathElement::new(
                            vec![(x + 6, qy - 6), (x - 6, qy + 6)],
                            gate_rgb.stroke_width(2),
                        ))?;
                    }
                }
                _ => {
                    // Single-qubit gate box
                    let target = gate.targets.get(0).copied().unwrap_or(0);
                    let y = row_height * (target as i32 + 1);
                    let half = 15;

                    root.draw(&Rectangle::new(
                        [(x - half, y - half), (x + half, y + half)],
                        gate_rgb.filled(),
                    ))?;

                    root.draw(&Text::new(
                        gate_name.clone(),
                        (x - half + 3, y - 6),
                        ("sans-serif", 12).into_font().color(&text_color),
                    ))?;
                }
            }
        }

        root.present()?;
        Ok(())
    }

    fn gate_name(&self, gate: &Gate) -> String {
        match &gate.gate_type {
            GateType::H => "H".to_string(),
            GateType::X => "X".to_string(),
            GateType::Y => "Y".to_string(),
            GateType::Z => "Z".to_string(),
            GateType::S => "S".to_string(),
            GateType::T => "T".to_string(),
            GateType::Rx(_) => "Rx".to_string(),
            GateType::Ry(_) => "Ry".to_string(),
            GateType::Rz(_) => "Rz".to_string(),
            GateType::U { .. } => "U".to_string(),
            GateType::CNOT => "CX".to_string(),
            GateType::CZ => "CZ".to_string(),
            GateType::SWAP => "SW".to_string(),
            GateType::Toffoli => "CCX".to_string(),
            GateType::CRx(_) => "CRx".to_string(),
            GateType::CRy(_) => "CRy".to_string(),
            GateType::CRz(_) => "CRz".to_string(),
            GateType::CR(_) => "CR".to_string(),
            GateType::Custom(_) => "?".to_string(),
        }
    }

    fn gate_color(&self, gate: &Gate) -> Rgb {
        match &self.style.gate_colors {
            GateColorScheme::Uniform(color) => *color,
            GateColorScheme::ByGateType(ref colors) => {
                let name = self.gate_name(gate);
                for (gate_name, color) in colors {
                    if name.contains(gate_name) {
                        return *color;
                    }
                }
                (200, 200, 200)
            }
        }
    }
}

/// State vector visualization
#[cfg(feature = "visualization")]
pub struct StateVisualization {
    state: QuantumState,
}

#[cfg(feature = "visualization")]
impl StateVisualization {
    pub fn new(state: QuantumState) -> Self {
        StateVisualization { state }
    }

    /// Create a bar chart of measurement probabilities
    pub fn save_probabilities<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let num_qubits = self.state.num_qubits;
        let num_states = 1 << num_qubits;

        let probabilities: Vec<f64> = self
            .state
            .amplitudes_ref()
            .iter()
            .map(|amp| amp.re * amp.re + amp.im * amp.im)
            .collect();

        let max_prob = probabilities
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(0.01);

        let mut chart = ChartBuilder::on(&root)
            .margin(40)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .caption("Measurement Probabilities", ("sans-serif", 20))
            .build_cartesian_2d(0i32..num_states as i32, 0.0..max_prob * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Measurement Outcome")
            .y_desc("Probability")
            .x_label_formatter(&|x| format!("|{:0>width$b}>", x, width = num_qubits))
            .draw()?;

        let bar_color = RGBColor(100, 150, 255);
        chart.draw_series(probabilities.iter().enumerate().map(|(i, &prob)| {
            let x = i as i32;
            Rectangle::new([(x, 0.0), (x + 1, prob)], bar_color.filled())
        }))?;

        root.present()?;
        Ok(())
    }

    /// Create a phase visualization
    pub fn save_phases<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new(path.as_ref(), (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let num_qubits = self.state.num_qubits;
        let num_states = 1 << num_qubits;

        let mut chart = ChartBuilder::on(&root)
            .margin(40)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .caption("Phase Distribution", ("sans-serif", 20))
            .build_cartesian_2d(
                0i32..num_states as i32,
                -std::f64::consts::PI..std::f64::consts::PI,
            )?;

        chart
            .configure_mesh()
            .x_desc("State |i>")
            .y_desc("Phase (radians)")
            .y_label_formatter(&|y| format!("{:.2}pi", y / std::f64::consts::PI))
            .draw()?;

        let dot_color = RGBColor(255, 100, 100);
        chart.draw_series(
            self.state
                .amplitudes_ref()
                .iter()
                .enumerate()
                .map(|(i, amp)| {
                    let phase = amp.im.atan2(amp.re);
                    Circle::new((i as i32, phase), 4, dot_color.filled())
                }),
        )?;

        root.present()?;
        Ok(())
    }

    /// Create a Bloch sphere visualization for single qubit
    pub fn save_bloch_sphere<P: AsRef<Path>>(
        &self,
        path: P,
        _qubit: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.state.num_qubits != 1 {
            return Err("Bloch sphere visualization only supports single-qubit states".into());
        }

        let root = BitMapBackend::new(path.as_ref(), (600, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let (bx, by, _bz) = self.bloch_vector();

        let center_x = 300i32;
        let center_y = 300i32;
        let radius = 200i32;

        // Draw sphere outline as a circle of dots
        let light_gray = RGBColor(200, 200, 200);
        for angle in (0..360).step_by(3) {
            let rad = (angle as f64).to_radians();
            let px = center_x + (radius as f64 * rad.cos()) as i32;
            let py = center_y + (radius as f64 * rad.sin()) as i32;
            root.draw(&Circle::new((px, py), 1, light_gray.filled()))?;
        }

        // Draw axes
        root.draw(&PathElement::new(
            vec![(center_x - radius, center_y), (center_x + radius, center_y)],
            light_gray.stroke_width(1),
        ))?;
        root.draw(&PathElement::new(
            vec![(center_x, center_y - radius), (center_x, center_y + radius)],
            light_gray.stroke_width(1),
        ))?;

        // Draw state vector (projected to 2D)
        let state_x = center_x + (radius as f64 * bx * 0.8) as i32;
        let state_y = center_y - (radius as f64 * by * 0.8) as i32;

        let red = RGBColor(255, 100, 100);
        root.draw(&PathElement::new(
            vec![(center_x, center_y), (state_x, state_y)],
            red.stroke_width(2),
        ))?;

        let dark_red = RGBColor(255, 50, 50);
        root.draw(&Circle::new((state_x, state_y), 6, dark_red.filled()))?;

        // Labels
        let black = RGBColor(0, 0, 0);
        let label_style = ("sans-serif", 16).into_font().color(&black);
        root.draw(&Text::new(
            "|+>",
            (center_x + radius + 5, center_y - 8),
            label_style.clone(),
        ))?;
        root.draw(&Text::new(
            "|0>",
            (center_x - 10, center_y - radius - 18),
            label_style.clone(),
        ))?;
        root.draw(&Text::new(
            "|1>",
            (center_x - 10, center_y + radius + 5),
            label_style,
        ))?;

        root.present()?;
        Ok(())
    }

    fn bloch_vector(&self) -> (f64, f64, f64) {
        let amps = self.state.amplitudes_ref();
        let alpha = amps[0];
        let beta = amps[1];

        let x = 2.0 * (alpha.re * beta.re + alpha.im * beta.im);
        let y = 2.0 * (alpha.im * beta.re - alpha.re * beta.im);
        let z =
            (alpha.re * alpha.re + alpha.im * alpha.im) - (beta.re * beta.re + beta.im * beta.im);

        (x, y, z)
    }
}

/// Generate a circuit diagram from gates
pub fn plot_circuit(
    num_qubits: usize,
    gates: Vec<Gate>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let _diagram = CircuitDiagram::new(num_qubits, gates);
    #[cfg(feature = "visualization")]
    {
        _diagram.save(path)?;
    }
    Ok(())
}

/// Generate state visualization
#[cfg(feature = "visualization")]
pub fn plot_state(state: &QuantumState, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let viz = StateVisualization::new(state.clone());
    viz.save_probabilities(path)
}

/// Generate measurement histogram
pub fn plot_measurements(
    measurements: &[usize],
    num_qubits: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "visualization")]
    {
        let num_states = 1 << num_qubits;
        let mut counts = vec![0usize; num_states];

        for &m in measurements {
            if m < num_states {
                counts[m] += 1;
            }
        }

        let total = measurements.len() as f64;
        let probabilities: Vec<f64> = counts.iter().map(|c| *c as f64 / total).collect();

        let max_prob = probabilities
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
            .max(0.01);

        let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(40)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .caption("Measurement Histogram", ("sans-serif", 20))
            .build_cartesian_2d(0i32..num_states as i32, 0.0..max_prob * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Measurement Outcome")
            .y_desc("Frequency")
            .x_label_formatter(&|x| format!("|{:0>width$b}>", x, width = num_qubits))
            .draw()?;

        let bar_color = RGBColor(100, 150, 255);
        chart.draw_series(probabilities.iter().enumerate().map(|(i, &prob)| {
            let x = i as i32;
            Rectangle::new([(x, 0.0), (x + 1, prob)], bar_color.filled())
        }))?;

        root.present()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    #[test]
    fn test_circuit_diagram_creation() {
        let gates = vec![Gate::h(0), Gate::h(1), Gate::cnot(0, 1)];

        let diagram = CircuitDiagram::new(2, gates);
        assert_eq!(diagram.num_qubits, 2);
        assert_eq!(diagram.gates.len(), 3);
    }

    #[test]
    fn test_default_style() {
        let style = CircuitVisualStyle::default();
        assert_eq!(style.width, 800);
        assert_eq!(style.height, 400);
        assert_eq!(style.background_color, (255, 255, 255));
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_circuit_diagram_save() {
        let gates = vec![Gate::h(0), Gate::cnot(0, 1)];

        let diagram = CircuitDiagram::new(2, gates);
        let result = diagram.save("/tmp/test_circuit.png");
        assert!(result.is_ok());
    }

    #[cfg(feature = "visualization")]
    #[test]
    fn test_probability_chart() {
        let state = QuantumState::new(2);
        let viz = StateVisualization::new(state);
        let result = viz.save_probabilities("/tmp/test_probs.png");
        assert!(result.is_ok());
    }
}
