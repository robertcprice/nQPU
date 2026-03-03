use std::env;
use std::time::Instant;

use nqpu_metal::adaptive_mps::{AdaptiveConfig, AdaptiveMPS};
use nqpu_metal::tensor_network::MPSSimulator;
#[cfg(all(feature = "metal", target_os = "macos"))]
use nqpu_metal::metal_mps::MetalMPSimulator;

fn apply_layers_mps(sim: &mut MPSSimulator, qubits: usize, layers: usize) {
    // Initial superposition
    for q in 0..qubits {
        sim.h(q);
    }
    for layer in 0..layers {
        let theta = 0.13 * (layer as f64 + 1.0);
        for q in 0..qubits {
            if q % 2 == 0 {
                sim.rx(q, theta);
            } else {
                sim.ry(q, theta * 0.9);
            }
        }
        if layer % 3 == 0 {
            for q in 0..qubits {
                sim.rz(q, theta * 0.5);
            }
        }
        // Brickwork entangling layers
        if qubits >= 2 {
            for q in (0..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1);
            }
            for q in (1..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1);
            }
        }
    }
}

fn apply_layers_adaptive(sim: &mut AdaptiveMPS, qubits: usize, layers: usize) {
    for q in 0..qubits {
        sim.h(q);
    }
    for layer in 0..layers {
        let theta = 0.13 * (layer as f64 + 1.0);
        for q in 0..qubits {
            if q % 2 == 0 {
                sim.rx(q, theta);
            } else {
                sim.ry(q, theta * 0.9);
            }
        }
        if layer % 3 == 0 {
            for q in 0..qubits {
                sim.rz(q, theta * 0.5);
            }
        }
        if qubits >= 2 {
            for q in (0..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1);
            }
            for q in (1..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1);
            }
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn apply_layers_metal(sim: &mut MetalMPSimulator, qubits: usize, layers: usize) -> Result<(), String> {
    for q in 0..qubits {
        sim.h(q)?;
    }
    for layer in 0..layers {
        let theta = 0.13 * (layer as f64 + 1.0);
        for q in 0..qubits {
            if q % 2 == 0 {
                sim.rx(q, theta)?;
            } else {
                sim.ry(q, theta * 0.9)?;
            }
        }
        if layer % 3 == 0 {
            for q in 0..qubits {
                sim.rz(q, theta * 0.5)?;
            }
        }
        if qubits >= 2 {
            for q in (0..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1)?;
            }
            for q in (1..(qubits - 1)).step_by(2) {
                sim.cnot(q, q + 1)?;
            }
        }
    }
    Ok(())
}

fn parse_arg(args: &[String], name: &str, default: usize) -> usize {
    if let Some(pos) = args.iter().position(|a| a == name) {
        if let Some(v) = args.get(pos + 1) {
            return v.parse::<usize>().unwrap_or(default);
        }
    }
    default
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let qubits = parse_arg(&args, "--qubits", 64);
    let bond = parse_arg(&args, "--bond", 64);
    let layers = parse_arg(&args, "--layers", 8);
    let adaptive = has_flag(&args, "--adaptive");
    let metal = has_flag(&args, "--metal");
    let track_ent = has_flag(&args, "--track-ent");

    let start = Instant::now();
    if metal {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let mut sim = match MetalMPSimulator::new(qubits, bond) {
                Ok(sim) => sim,
                Err(e) => {
                    eprintln!("ERROR metal init failed: {}", e);
                    std::process::exit(1);
                }
            };
            if let Err(e) = apply_layers_metal(&mut sim, qubits, layers) {
                eprintln!("ERROR metal apply failed: {}", e);
                std::process::exit(1);
            }
            let max_bond = sim.max_bond_dim().unwrap_or(0);
            let _ = sim.measure();
            let elapsed = start.elapsed().as_millis();
            println!(
                "RESULT op=mps_stress adaptive=0 metal=1 qubits={} bond={} layers={} max_bond={} time_ms={} detail=ok",
                qubits,
                bond,
                layers,
                max_bond,
                elapsed
            );
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            eprintln!("ERROR metal backend not available on this platform");
            std::process::exit(1);
        }
    } else if adaptive {
        let initial = bond.min(4).max(2);
        let cfg = AdaptiveConfig::default()
            .with_initial_bond_dim(initial)
            .with_max_bond_dim(bond)
            .with_threshold(0.7)
            .with_verbose(false)
            .with_growth_factor(2);
        let mut sim = AdaptiveMPS::new(qubits, cfg);
        apply_layers_adaptive(&mut sim, qubits, layers);
        let current = sim.current_bond_dim();
        let expansions = sim.expansion_count();
        let max_bond = sim.bond_dimensions().into_iter().max().unwrap_or(1);
        let _ = sim.measure();
        let elapsed = start.elapsed().as_millis();
        println!(
            "RESULT op=mps_stress adaptive=1 qubits={} bond={} layers={} max_bond={} current_bond={} expansions={} time_ms={} detail=ok",
            qubits,
            bond,
            layers,
            max_bond,
            current,
            expansions,
            elapsed
        );
    } else {
        let mut sim = MPSSimulator::new(qubits, Some(bond));
        if track_ent {
            sim.enable_entanglement_tracking(true);
        }
        apply_layers_mps(&mut sim, qubits, layers);
        let max_bond = sim.max_bond_dim();
        let max_ent = if track_ent && qubits > 1 {
            let mut m = 0.0f64;
            for b in 0..(qubits - 1) {
                if let Some(s) = sim.bond_entanglement_entropy(b) {
                    if s > m {
                        m = s;
                    }
                }
            }
            Some(m)
        } else {
            None
        };
        let _ = sim.measure();
        let elapsed = start.elapsed().as_millis();
        println!(
            "RESULT op=mps_stress adaptive=0 metal=0 qubits={} bond={} layers={} max_bond={} max_ent={} time_ms={} detail=ok",
            qubits,
            bond,
            layers,
            max_bond,
            max_ent.map(|v| format!("{:.6}", v)).unwrap_or_else(|| "na".to_string()),
            elapsed
        );
    }
}
