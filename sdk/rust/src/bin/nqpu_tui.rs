//! nQPU-Metal Terminal User Interface
//!
//! Run with: cargo run --release --bin nqpu_tui

use nqpu_metal::tui::run_tui_demo;

fn main() {
    println!();
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        nQPU-Metal Quantum Simulator TUI v2.0               ║");
    println!("║                                                             ║");
    println!("║  Features:                                                  ║");
    println!("║    • 3D Bloch sphere rendering                              ║");
    println!("║    • Real-time performance dashboard                        ║");
    println!("║    • Probability histograms                                 ║");
    println!("║    • Circuit diagrams                                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();

    run_tui_demo();
}
