//! nQPU-Metal Terminal User Interface (TUI)
//!
//! Advanced 3D-style ASCII visualization and interactive terminal interface.

use std::io::{self, Write};
use std::time::Duration;

// ---------------------------------------------------------------------------
// ANSI ESCAPE CODES
// ---------------------------------------------------------------------------

mod ansi {
    pub const RESET: &str = "\x1b[0m";
    pub const CLEAR: &str = "\x1b[2J";
    pub const HOME: &str = "\x1b[H";
    pub const HIDE_CURSOR: &str = "\x1b[?25l";
    pub const SHOW_CURSOR: &str = "\x1b[?25h";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const CYAN: &str = "\x1b[36m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const BLUE: &str = "\x1b[34m";
    pub const RED: &str = "\x1b[31m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m";
    pub const BG_BLUE: &str = "\x1b[44m";
}

// ---------------------------------------------------------------------------
// 3D BLOCH SPHERE RENDERER
// ---------------------------------------------------------------------------

/// 3D Bloch sphere with ASCII art rendering
pub struct BlochSphere3D {
    width: usize,
    height: usize,
    rot_x: f64,
    rot_y: f64,
    theta: f64,
    phi: f64,
}

impl BlochSphere3D {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rot_x: 0.3,
            rot_y: 0.0,
            theta: 0.0,
            phi: 0.0,
        }
    }

    pub fn set_state(&mut self, theta: f64, phi: f64) {
        self.theta = theta;
        self.phi = phi;
    }

    pub fn rotate(&mut self, dx: f64, dy: f64) {
        self.rot_y += dx;
        self.rot_x = (self.rot_x + dy).clamp(-1.5, 1.5);
    }

    fn project(&self, x: f64, y: f64, z: f64) -> (isize, isize) {
        let cos_x = self.rot_x.cos();
        let sin_x = self.rot_x.sin();
        let y1 = y * cos_x - z * sin_x;
        let z1 = y * sin_x + z * cos_x;

        let cos_y = self.rot_y.cos();
        let sin_y = self.rot_y.sin();
        let x2 = x * cos_y + z1 * sin_y;

        let scale = self.width as f64 / 4.0;
        let perspective = 1.0 / (1.0 - (-x * sin_y + y * sin_x * cos_y + z * cos_x * cos_y) * 0.3);

        let screen_x = (self.width as f64 / 2.0 + x2 * scale * perspective) as isize;
        let screen_y = (self.height as f64 / 2.0 - y1 * scale * perspective * 0.5) as isize;

        (screen_x, screen_y)
    }

    pub fn render(&self) -> Vec<String> {
        let mut buffer = vec![vec![' '; self.width]; self.height];

        // Draw sphere wireframe with characters
        let chars = ['.', ':', ';', '!', 'i', 'l', 'I', '|', 'C', 'O', '@', '#'];

        for lat in 0..12 {
            let theta = (lat as f64 / 12.0) * std::f64::consts::PI;
            for lon in 0..24 {
                let phi = (lon as f64 / 24.0) * 2.0 * std::f64::consts::PI;

                let x = theta.sin() * phi.cos();
                let y = theta.cos();
                let z = theta.sin() * phi.sin();

                let (sx, sy) = self.project(x, y, z);

                if sx >= 0 && sx < self.width as isize && sy >= 0 && sy < self.height as isize {
                    let shade = ((z + 1.0) * 0.5 * (chars.len() - 1) as f64) as usize;
                    let shade = shade.min(chars.len() - 1);
                    buffer[sy as usize][sx as usize] = chars[shade];
                }
            }
        }

        // Draw state vector
        let vx = self.theta.sin() * self.phi.cos();
        let vy = self.theta.cos();
        let vz = self.theta.sin() * self.phi.sin();

        for i in 0..20 {
            let t = i as f64 / 20.0;
            let (sx, sy) = self.project(vx * t, vy * t, vz * t);
            if sx >= 0 && sx < self.width as isize && sy >= 0 && sy < self.height as isize {
                buffer[sy as usize][sx as usize] = if i > 16 { '*' } else { '+' };
            }
        }

        buffer.iter().map(|row| row.iter().collect()).collect()
    }
}

// ---------------------------------------------------------------------------
// PERFORMANCE DASHBOARD
// ---------------------------------------------------------------------------

pub struct PerfDashboard {
    gates_per_sec: f64,
    memory_mb: f64,
    qubits: usize,
    depth: usize,
    backend: String,
}

impl PerfDashboard {
    pub fn new() -> Self {
        Self {
            gates_per_sec: 0.0,
            memory_mb: 0.0,
            qubits: 0,
            depth: 0,
            backend: "CPU".to_string(),
        }
    }

    pub fn update(
        &mut self,
        gates_per_sec: f64,
        memory_mb: f64,
        qubits: usize,
        depth: usize,
        backend: &str,
    ) {
        self.gates_per_sec = gates_per_sec;
        self.memory_mb = memory_mb;
        self.qubits = qubits;
        self.depth = depth;
        self.backend = backend.to_string();
    }

    pub fn render(&self) -> Vec<String> {
        let mut lines = Vec::new();
        let b = ansi::BRIGHT_GREEN;
        let r = ansi::RESET;

        lines.push(format!(
            "{}╔══════════════════════════════════════╗{}",
            b, r
        ));
        lines.push(format!(
            "{}║     nQPU-Metal Performance           ║{}",
            b, r
        ));
        lines.push(format!(
            "{}╠══════════════════════════════════════╣{}",
            b, r
        ));

        let mhz = self.gates_per_sec / 1_000_000.0;
        let color = if mhz > 50.0 {
            ansi::BRIGHT_GREEN
        } else if mhz > 20.0 {
            ansi::BRIGHT_YELLOW
        } else {
            ansi::BRIGHT_RED
        };

        lines.push(format!(
            "{}║{} Throughput: {}{:>10.2} MHz{}     {}║{}",
            b, r, color, mhz, r, b, r
        ));
        lines.push(format!(
            "{}║{} Qubits:     {}{:>10}{}          {}║{}",
            b,
            r,
            ansi::BRIGHT_CYAN,
            self.qubits,
            r,
            b,
            r
        ));
        lines.push(format!(
            "{}║{} Depth:      {}{:>10}{}          {}║{}",
            b,
            r,
            ansi::BRIGHT_CYAN,
            self.depth,
            r,
            b,
            r
        ));
        lines.push(format!(
            "{}║{} Memory:     {}{:>10.1} MB{}       {}║{}",
            b,
            r,
            ansi::BRIGHT_MAGENTA,
            self.memory_mb,
            r,
            b,
            r
        ));
        lines.push(format!(
            "{}║{} Backend:    {}{:>10}{}          {}║{}",
            b,
            r,
            ansi::BRIGHT_BLUE,
            self.backend,
            r,
            b,
            r
        ));
        lines.push(format!(
            "{}╚══════════════════════════════════════╝{}",
            b, r
        ));

        lines
    }
}

impl Default for PerfDashboard {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HISTOGRAM
// ---------------------------------------------------------------------------

pub struct Histogram3D {
    width: usize,
    height: usize,
}

impl Histogram3D {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    pub fn render(&self, probabilities: &[f64]) -> Vec<String> {
        let mut lines = Vec::new();
        if probabilities.is_empty() {
            return lines;
        }

        let max_prob = probabilities.iter().cloned().fold(0.0, f64::max).max(0.001);

        for (i, &prob) in probabilities.iter().enumerate() {
            let normalized = prob / max_prob;
            let filled = (normalized * self.height as f64) as usize;
            let filled = filled.min(self.height);

            let bar: String = (0..filled).map(|_| "█").collect();
            let empty: String = (0..(self.height - filled)).map(|_| "░").collect();

            let label = format!("|{}⟩", i);
            let pct = format!("{:>6.2}%", prob * 100.0);

            lines.push(format!(
                "{:>6} {}{} {}{}{}{}",
                label,
                empty,
                bar,
                ansi::GREEN,
                pct,
                ansi::RESET,
                if prob == max_prob { " ★" } else { "" }
            ));
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// CIRCUIT DIAGRAM
// ---------------------------------------------------------------------------

pub struct CircuitDiagram3D {
    _width: usize,
}

impl CircuitDiagram3D {
    pub fn new(width: usize) -> Self {
        Self { _width: width }
    }

    pub fn render(&self, gates: &[(&str, usize, Option<usize>)], n_qubits: usize) -> Vec<String> {
        let mut lines: Vec<String> = (0..n_qubits).map(|q| format!("q{}: ─", q)).collect();

        for (gate, target, control) in gates {
            for q in 0..n_qubits {
                if q == *target {
                    match control {
                        Some(c) => {
                            lines[q].push_str(&format!("{}[{}]", ansi::YELLOW, gate));
                            lines[*c].push_str(&format!("{}●{}", ansi::CYAN, ansi::RESET));
                        }
                        None => {
                            lines[q].push_str(&format!("{}[{}]{}", ansi::GREEN, gate, ansi::RESET));
                        }
                    }
                } else if control.map(|c| c == q).unwrap_or(false) {
                    lines[q].push_str(&format!("{}●{}", ansi::CYAN, ansi::RESET));
                } else {
                    lines[q].push_str("──");
                }
            }
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// MAIN TUI
// ---------------------------------------------------------------------------

pub struct QuantumTUI {
    bloch: BlochSphere3D,
    histogram: Histogram3D,
    dashboard: PerfDashboard,
    circuit: CircuitDiagram3D,
    running: bool,
    frame: u64,
}

impl QuantumTUI {
    pub fn new() -> Self {
        Self {
            bloch: BlochSphere3D::new(40, 20),
            histogram: Histogram3D::new(60, 10),
            dashboard: PerfDashboard::new(),
            circuit: CircuitDiagram3D::new(80),
            running: true,
            frame: 0,
        }
    }

    pub fn clear(&self) {
        print!("{}{}", ansi::CLEAR, ansi::HOME);
    }

    pub fn hide_cursor(&self) {
        print!("{}", ansi::HIDE_CURSOR);
    }

    pub fn show_cursor(&self) {
        print!("{}", ansi::SHOW_CURSOR);
    }

    pub fn render(&mut self) {
        self.clear();

        // Title
        println!(
            "{}{}╔═══════════════════════════════════════════════════════════════════╗{}",
            ansi::BOLD,
            ansi::BG_BLUE,
            ansi::RESET
        );
        println!(
            "{}{}║             {}nQPU-Metal Quantum Simulator TUI v2.0{}               {}║{}",
            ansi::BOLD,
            ansi::BG_BLUE,
            ansi::BRIGHT_WHITE,
            ansi::RESET,
            ansi::BG_BLUE,
            ansi::RESET
        );
        println!(
            "{}{}╚═══════════════════════════════════════════════════════════════════╝{}",
            ansi::BOLD,
            ansi::BG_BLUE,
            ansi::RESET
        );
        println!();

        // Dashboard
        for line in self.dashboard.render() {
            println!("  {}", line);
        }
        println!();

        // Bloch Sphere
        println!(
            "{}{} Bloch Sphere (3D){}",
            ansi::BOLD,
            ansi::CYAN,
            ansi::RESET
        );
        for line in self.bloch.render() {
            println!("  {}", line);
        }
        println!();

        // Histogram
        println!(
            "{}{} State Probabilities{}",
            ansi::BOLD,
            ansi::CYAN,
            ansi::RESET
        );
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        for line in self.histogram.render(&probs) {
            println!("  {}", line);
        }
        println!();

        // Circuit
        println!(
            "{}{} Circuit Diagram{}",
            ansi::BOLD,
            ansi::CYAN,
            ansi::RESET
        );
        let gates: Vec<(&str, usize, Option<usize>)> =
            vec![("H", 0, None), ("CX", 1, Some(0)), ("Z", 0, None)];
        for line in self.circuit.render(&gates, 3) {
            println!("  {}", line);
        }
        println!();

        // Controls
        println!(
            "{}[q]{}uit  {}[r]{}otate  {}[space]{}step",
            ansi::GREEN,
            ansi::RESET,
            ansi::GREEN,
            ansi::RESET,
            ansi::GREEN,
            ansi::RESET
        );
        println!("{}Frame: {}{}", ansi::DIM, self.frame, ansi::RESET);

        io::stdout().flush().ok();
    }

    pub fn tick(&mut self) {
        self.frame += 1;
        self.bloch.rotate(0.02, 0.01);

        let mhz = 37.0 + (self.frame as f64 * 0.1).sin() * 5.0;
        self.dashboard.update(
            mhz * 1_000_000.0,
            128.0,
            10,
            50,
            if self.frame % 100 < 50 {
                "CPU (NEON)"
            } else {
                "GPU (Metal)"
            },
        );

        let theta = (self.frame as f64 * 0.02).sin() * std::f64::consts::PI;
        let phi = self.frame as f64 * 0.05;
        self.bloch.set_state(theta, phi);
    }

    pub fn run(&mut self) {
        self.hide_cursor();

        while self.running {
            self.render();
            self.tick();

            std::thread::sleep(Duration::from_millis(50));

            if self.frame > 200 {
                self.running = false;
            }
        }

        self.show_cursor();
        println!("{}TUI demo complete!{}", ansi::GREEN, ansi::RESET);
    }
}

impl Default for QuantumTUI {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DEMO FUNCTION
// ---------------------------------------------------------------------------

pub fn run_tui_demo() {
    println!(
        "{}{}Starting nQPU-Metal TUI...{}",
        ansi::CLEAR,
        ansi::HOME,
        ansi::RESET
    );
    println!();
    println!("Features:");
    println!(
        "  {}•{} 3D Bloch sphere visualization",
        ansi::GREEN,
        ansi::RESET
    );
    println!(
        "  {}•{} Real-time performance dashboard",
        ansi::GREEN,
        ansi::RESET
    );
    println!("  {}•{} Probability histograms", ansi::GREEN, ansi::RESET);
    println!("  {}•{} Circuit diagrams", ansi::GREEN, ansi::RESET);
    println!();
    println!("Running animation for 200 frames...");
    println!();

    let mut tui = QuantumTUI::new();
    tui.run();
}

// ---------------------------------------------------------------------------
// TESTS
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloch_sphere() {
        let bloch = BlochSphere3D::new(40, 20);
        assert_eq!(bloch.width, 40);
        let lines = bloch.render();
        assert_eq!(lines.len(), 20);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram3D::new(60, 10);
        let probs = vec![0.5, 0.3, 0.2];
        let lines = hist.render(&probs);
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_dashboard() {
        let mut dash = PerfDashboard::new();
        dash.update(50_000_000.0, 256.0, 20, 100, "CPU");
        let lines = dash.render();
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_tui_creation() {
        let tui = QuantumTUI::new();
        assert!(tui.running);
    }
}
