# Terminal User Interface (TUI)

nQPU-Metal includes an interactive terminal UI that renders a 3D Bloch sphere, probability histograms, circuit diagrams, and a real-time performance dashboard -- all in ASCII art with ANSI color.

## Table of Contents

- [Launching the TUI](#launching-the-tui)
- [What You See](#what-you-see)
- [Controls](#controls)
- [Terminal Requirements](#terminal-requirements)
- [Programmatic Usage](#programmatic-usage)
- [Example Output](#example-output)

---

## Launching the TUI

```bash
cargo run --release --bin nqpu_tui
```

The TUI binary is at `sdk/rust/src/bin/nqpu_tui.rs`. It does not require any feature flags beyond the defaults.

On launch, it prints a welcome banner and then runs a 200-frame animated demo that cycles through quantum states.

---

## What You See

The TUI renders four panels, stacked vertically:

### 1. Performance Dashboard

A bordered box showing real-time simulation metrics:

| Metric | Description |
|--------|-------------|
| **Throughput** | Gate application speed in MHz. Color-coded: green (>50 MHz), yellow (>20 MHz), red (<20 MHz). |
| **Qubits** | Number of active qubits in the simulation |
| **Depth** | Current circuit depth |
| **Memory** | Memory usage in MB |
| **Backend** | Active execution backend (e.g., `CPU (NEON)` or `GPU (Metal)`) |

### 2. Bloch Sphere (3D)

A 40x20 character wireframe sphere rendered with perspective projection. The sphere uses graduated ASCII shading characters (`.`, `:`, `;`, `!`, `i`, `l`, `I`, `|`, `C`, `O`, `@`, `#`) for depth cues. The current qubit state vector is drawn as a line from the origin to the surface, ending in a `*` marker, with `+` characters along the shaft.

The sphere rotates continuously (0.02 rad/frame horizontal, 0.01 rad/frame vertical) and the state vector traces a path determined by the current theta and phi angles.

### 3. State Probabilities

A horizontal bar chart showing measurement probabilities for each computational basis state. Each bar uses filled (`█`) and empty (`░`) block characters, with the probability percentage displayed to the right. The most probable state is marked with a star (`★`).

### 4. Circuit Diagram

An inline circuit diagram showing qubit wires (`q0: ─`, `q1: ─`, ...) with gate labels inserted at the appropriate positions:

- Single-qubit gates appear as `[H]`, `[Z]`, etc. in green.
- Controlled gates show a control dot (`●`) in cyan on the control qubit and `[CX]` in yellow on the target qubit.

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit the TUI |
| `r` | Rotate the Bloch sphere |
| `Space` | Step forward one simulation frame |

The control hints are displayed at the bottom of the screen.

> **Note:** The current demo build runs a fixed 200-frame animation and exits automatically. Interactive keyboard input requires a raw-mode terminal library (not yet wired in). The control labels are displayed for future interactive mode.

---

## Terminal Requirements

The TUI uses ANSI escape codes for rendering. For the best experience:

| Requirement | Details |
|-------------|---------|
| **Color support** | 16-color ANSI minimum. 256-color terminals produce better results. The TUI uses standard and bright ANSI colors (codes 30-37 and 90-97). |
| **Terminal size** | At least 80 columns wide and 50 rows tall to display all panels without wrapping. |
| **Unicode support** | Required for box-drawing characters (`╔`, `║`, `╚`), block elements (`█`, `░`), and quantum notation (`⟩`). Most modern terminals support these. |
| **Cursor hiding** | The TUI hides the cursor during rendering and restores it on exit. If the process is killed unexpectedly, run `tput cnorm` or `printf '\e[?25h'` to restore the cursor. |

**Tested terminals:** Terminal.app (macOS), iTerm2, Alacritty, WezTerm, Kitty. Windows Terminal also works. Avoid very old terminal emulators that lack ANSI support.

---

## Programmatic Usage

You can embed TUI components in your own code. The components are defined in `sdk/rust/src/infra/tui.rs` and exported through the `tui` module.

### Bloch Sphere

```rust
use nqpu_metal::tui::BlochSphere3D;

let mut bloch = BlochSphere3D::new(40, 20); // width, height in characters
bloch.set_state(std::f64::consts::FRAC_PI_4, 0.0); // theta, phi
bloch.rotate(0.1, 0.05); // rotate the view

let lines: Vec<String> = bloch.render();
for line in &lines {
    println!("{}", line);
}
```

### Probability Histogram

```rust
use nqpu_metal::tui::Histogram3D;

let hist = Histogram3D::new(60, 10); // width, height
let probabilities = vec![0.5, 0.3, 0.15, 0.05];
let lines = hist.render(&probabilities);
for line in &lines {
    println!("{}", line);
}
```

### Performance Dashboard

```rust
use nqpu_metal::tui::PerfDashboard;

let mut dash = PerfDashboard::new();
dash.update(
    50_000_000.0, // gates per second
    256.0,        // memory in MB
    20,           // qubits
    100,          // circuit depth
    "GPU (Metal)" // backend name
);
for line in dash.render() {
    println!("{}", line);
}
```

### Circuit Diagram

```rust
use nqpu_metal::tui::CircuitDiagram3D;

let circuit = CircuitDiagram3D::new(80); // width
let gates: Vec<(&str, usize, Option<usize>)> = vec![
    ("H", 0, None),          // Hadamard on qubit 0
    ("CX", 1, Some(0)),      // CNOT: control=0, target=1
    ("Z", 0, None),          // Pauli-Z on qubit 0
];
let lines = circuit.render(&gates, 3); // 3 qubits
for line in &lines {
    println!("{}", line);
}
```

### Full TUI

```rust
use nqpu_metal::tui::QuantumTUI;

let mut tui = QuantumTUI::new();
tui.run(); // Runs 200-frame animated demo
```

Or call the convenience function:

```rust
use nqpu_metal::tui::run_tui_demo;
run_tui_demo();
```

---

## Example Output

Below is a simplified text representation of what the TUI renders. Actual output includes ANSI colors.

```
╔═══════════════════════════════════════════════════════════════════╗
║             nQPU-Metal Quantum Simulator TUI v2.0               ║
╚═══════════════════════════════════════════════════════════════════╝

  ╔══════════════════════════════════════╗
  ║     nQPU-Metal Performance           ║
  ╠══════════════════════════════════════╣
  ║ Throughput:      37.00 MHz           ║
  ║ Qubits:              10             ║
  ║ Depth:               50             ║
  ║ Memory:          128.0 MB           ║
  ║ Backend:     CPU (NEON)             ║
  ╚══════════════════════════════════════╝

 Bloch Sphere (3D)
          ..::;;!!
        .:;!ilI|CO@
       .;!lI|CO@##
       :;iI|CO@#
        .;!lI|CO@
         ..::;+!
            *

 State Probabilities
    |0>  ██████████  50.00% ★
    |1>  ██████      30.00%
    |2>  ███         15.00%
    |3>  █            5.00%

 Circuit Diagram
  q0: ─[H]──●──[Z]──
  q1: ────[CX]──────
  q2: ───────────────

[q]uit  [r]otate  [space]step
Frame: 1
```

The Bloch sphere rotates continuously, the dashboard metrics update each frame, and the backend label alternates between `CPU (NEON)` and `GPU (Metal)` in the demo to illustrate backend switching.
