use std::env;
use std::time::Instant;

use nqpu_metal::QuantumSimulator;
use nqpu_metal::density_matrix::DensityMatrixSimulator;
use nqpu_metal::adaptive_mps::{AdaptiveConfig, AdaptiveMPS};
use nqpu_metal::tensor_network::MPSSimulator;
use nqpu_metal::peps::PEPSimulator;
use nqpu_metal::qft_2d::QFT2D;
use nqpu_metal::simulation_3d::Simulator3D;
use nqpu_metal::advanced_noise::{NoiseModel, NoisySimulator};
use nqpu_metal::gates::Gate;
use nqpu_metal::state_tomography::{
    StateTomography, ProcessTomography, TomographySettings, TomographyMeasurement, MeasurementBasis,
};
use nqpu_metal::vqe::{VQESolver, hamiltonians};
use nqpu_metal::qao::{QAOSolver, CostFunction, Mixer};
use nqpu_metal::qpe::QPESolver;
use nqpu_metal::annealing::{AnnealingConfig, QuantumAnnealing};
#[cfg(all(feature = "metal", target_os = "macos"))]
use nqpu_metal::metal_mps::MetalMPSimulator;

use ndarray::Array2;
use num_complex::Complex64;

fn get_arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|idx| args.get(idx + 1))
        .cloned()
}

fn parse_usize(args: &[String], key: &str, default: usize) -> usize {
    get_arg_value(args, key)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u64(args: &[String], key: &str, default: u64) -> u64 {
    get_arg_value(args, key)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_f64(args: &[String], key: &str, default: f64) -> f64 {
    get_arg_value(args, key)
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

fn print_result(
    op: &str,
    qubits: usize,
    size: usize,
    bond: usize,
    steps: usize,
    time_ms: u128,
    detail: &str,
) {
    println!(
        "RESULT op={} qubits={} size={} bond={} steps={} time_ms={} detail={}",
        op, qubits, size, bond, steps, time_ms, detail
    );
}

fn op_state_vector(qubits: usize, steps: usize) -> Result<String, String> {
    let mut sim = QuantumSimulator::new(qubits);
    for i in 0..qubits {
        sim.h(i);
    }
    for i in 0..qubits.saturating_sub(1) {
        sim.cnot(i, i + 1);
    }
    for i in 0..steps.min(qubits) {
        sim.z(i);
    }
    let _ = sim.measure();
    Ok("ok".to_string())
}

fn op_density_matrix(qubits: usize) -> Result<String, String> {
    let mut sim = DensityMatrixSimulator::new(qubits);
    for i in 0..qubits {
        sim.h(i);
    }
    for i in 0..qubits.saturating_sub(1) {
        sim.cnot(i, i + 1);
    }
    let _ = sim.measure_all();
    Ok("ok".to_string())
}

fn op_mps(qubits: usize, bond: usize) -> Result<String, String> {
    let mut sim = MPSSimulator::new(qubits, Some(bond));
    for i in 0..qubits {
        sim.h(i);
    }
    for i in 0..qubits.saturating_sub(1) {
        sim.cnot(i, i + 1);
    }
    let max_bond = sim.max_bond_dim();
    let _ = sim.measure();
    Ok(format!("max_bond={}", max_bond))
}

fn op_adaptive_mps(qubits: usize, bond: usize) -> Result<String, String> {
    let initial = bond.min(4).max(2);
    let cfg = AdaptiveConfig::default()
        .with_initial_bond_dim(initial)
        .with_max_bond_dim(bond)
        .with_threshold(0.7)
        .with_verbose(false)
        .with_growth_factor(2);
    let mut sim = AdaptiveMPS::new(qubits, cfg);
    for i in 0..qubits {
        sim.h(i);
    }
    for i in 0..qubits.saturating_sub(1) {
        sim.cnot(i, i + 1);
    }
    let current = sim.current_bond_dim();
    let expansions = sim.expansion_count();
    let _ = sim.measure();
    Ok(format!("current_bond={} expansions={}", current, expansions))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn op_metal_mps(qubits: usize, bond: usize) -> Result<String, String> {
    let mut sim = MetalMPSimulator::new(qubits, bond)
        .map_err(|e| e.to_string())?;
    for i in 0..qubits {
        sim.h(i)?;
    }
    for i in 0..qubits.saturating_sub(1) {
        sim.cnot(i, i + 1)?;
    }
    let max_bond = sim.max_bond_dim().unwrap_or(0);
    let _ = sim.measure();
    Ok(format!("max_bond={}", max_bond))
}

fn op_peps(size: usize) -> Result<String, String> {
    let mut sim = PEPSimulator::with_defaults(size, size);
    for y in 0..size {
        for x in 0..size {
            sim.h(x, y);
        }
    }
    if size > 1 {
        sim.cnot(0, 0, 1, 0);
    }
    let _ = sim.measure();
    let stats = sim.statistics();
    let max_bond = stats.max_bond_dim.iter().cloned().max().unwrap_or(0);
    Ok(format!("max_bond={}", max_bond))
}

fn op_qft_2d(size: usize) -> Result<String, String> {
    let width = size;
    let height = size;
    let mut state = vec![Complex64::new(0.0, 0.0); width * height];
    if !state.is_empty() {
        state[0] = Complex64::new(1.0, 0.0);
    }
    let qft = QFT2D::new(width, height);
    qft.apply(&mut state);
    qft.apply_inverse(&mut state);
    Ok("ok".to_string())
}

fn op_sim_3d(size: usize) -> Result<String, String> {
    let mut sim = Simulator3D::new(size, size, size);
    if size > 0 {
        sim.h(0, 0, 0);
    }
    if size > 1 {
        sim.cnot(0, 0, 0, 1, 0, 0);
    }
    let _ = sim.measure();
    Ok("ok".to_string())
}

fn op_noise(qubits: usize) -> Result<String, String> {
    let model = NoiseModel::default();
    let mut sim = NoisySimulator::new(qubits, model);
    let mut rng = rand::thread_rng();
    sim.apply_gate(&Gate::h(0), &mut rng);
    if qubits > 1 {
        sim.apply_gate(&Gate::cnot(0, 1), &mut rng);
    }
    Ok("ok".to_string())
}

fn op_state_tomography(qubits: usize) -> Result<String, String> {
    let tomo = StateTomography::new(qubits);
    let settings = TomographySettings::simplified_tomography(qubits, 10);
    let measurements = vec![TomographyMeasurement {
        basis_setting: vec![MeasurementBasis::Z; qubits],
        outcome: 0,
        counts: 10,
    }];
    let _ = tomo.reconstruct_state(&measurements, &settings)?;
    Ok("ok".to_string())
}

fn op_process_tomography(qubits: usize) -> Result<String, String> {
    let proc = ProcessTomography::new(qubits);
    let input = vec![nqpu_metal::QuantumState::new(qubits)];
    let output = vec![nqpu_metal::QuantumState::new(qubits)];
    let _ = proc.reconstruct_process(&input, &output)?;
    Ok("ok".to_string())
}

fn op_vqe(qubits: usize, steps: usize) -> Result<String, String> {
    let h = hamiltonians::transverse_field_ising(qubits, 1.0, 1.0);
    let mut solver = VQESolver::new(qubits, 1, h, 0.05);
    solver.max_iterations = steps.max(1).min(50);
    let res = solver.find_ground_state();
    Ok(format!("iters={} energy={}", res.iterations, res.ground_state_energy))
}

fn op_qao(qubits: usize, steps: usize) -> Result<String, String> {
    let mut mat = Array2::<f64>::zeros((qubits, qubits));
    for i in 0..qubits {
        mat[[i, i]] = 1.0;
    }
    let cost = CostFunction::QUBO { matrix: mat, num_variables: qubits };
    let mut solver = QAOSolver::new(cost, 1, Mixer::Classical { angle: 0.5 });
    solver.max_iterations = steps.max(1).min(50);
    let res = solver.optimize();
    Ok(format!("iters={} best_cost={}", res.iterations, res.best_cost))
}

fn op_qpe(precision_qubits: usize) -> Result<String, String> {
    // Estimate eigenphase of a phase gate with phi = pi/4
    let eigenphase = std::f64::consts::FRAC_PI_4;
    let solver = QPESolver::new(eigenphase, precision_qubits);
    let res = solver.estimate_phase();
    Ok(format!("phase={} confidence={}", res.phase_estimate, res.confidence))
}

fn op_annealing(vars: usize, steps: usize) -> Result<String, String> {
    let mut cost = Array2::<f64>::zeros((vars, vars));
    for i in 0..vars {
        cost[[i, i]] = 1.0;
    }
    let mut cfg = AnnealingConfig::default();
    cfg.cost_matrix = cost;
    cfg.num_variables = vars;
    cfg.max_iterations = steps.max(10).min(200);
    let mut annealer = QuantumAnnealing::new(cfg);
    let res = annealer.anneal();
    Ok(format!("iters={} best_cost={}", res.iterations, res.best_cost))
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn op_metal_backend(qubits: usize) -> Result<String, String> {
    let sim = nqpu_metal::metal_backend::MetalSimulator::new(qubits)
        .map_err(|e| e.to_string())?;
    let mut gates = vec![Gate::h(0)];
    if qubits > 1 {
        gates.push(Gate::cnot(0, 1));
    }
    sim.run_circuit(&gates);
    let _ = sim.probabilities();
    Ok(format!("device={}", sim.device_name()))
}

#[cfg(feature = "cuda")]
fn op_cuda_backend(qubits: usize) -> Result<String, String> {
    let sim = nqpu_metal::cuda_backend::CudaQuantumSimulator::new(qubits)
        .map_err(|e| format!("cuda init failed: {:?}", e))?;
    Ok(format!("num_qubits={}", sim.num_qubits()))
}

#[cfg(feature = "rocm")]
fn op_rocm_backend(qubits: usize) -> Result<String, String> {
    let sim = nqpu_metal::rocm_backend::RocmQuantumSimulator::new(qubits)
        .map_err(|e| format!("rocm init failed: {:?}", e))?;
    Ok(format!("num_qubits={}", sim.num_qubits()))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let op = get_arg_value(&args, "--op").unwrap_or_else(|| "".to_string());
    if op.is_empty() {
        eprintln!("missing --op");
        std::process::exit(2);
    }
    let qubits = parse_usize(&args, "--qubits", 4);
    let size = parse_usize(&args, "--size", 4);
    let bond = parse_usize(&args, "--bond", 16);
    let steps = parse_usize(&args, "--steps", 10);
    let _seed = parse_u64(&args, "--seed", 0);
    let _dummy = parse_f64(&args, "--dummy", 0.0);

    let start = Instant::now();
    let result = match op.as_str() {
        "state_vector" => op_state_vector(qubits, steps),
        "density_matrix" => op_density_matrix(qubits),
        "mps" => op_mps(qubits, bond),
        "adaptive_mps" => op_adaptive_mps(qubits, bond),
        "metal_mps" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            { op_metal_mps(qubits, bond) }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            { Err("metal mps not available".to_string()) }
        }
        "peps" => op_peps(size),
        "qft_2d" => op_qft_2d(size),
        "sim_3d" => op_sim_3d(size),
        "noise" => op_noise(qubits),
        "tomography_state" => op_state_tomography(qubits),
        "tomography_process" => op_process_tomography(qubits),
        "vqe" => op_vqe(qubits, steps),
        "qao" => op_qao(qubits, steps),
        "qpe" => op_qpe(qubits),
        "annealing" => op_annealing(qubits, steps),
        "metal_backend" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            { op_metal_backend(qubits) }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            { Err("metal backend not available".to_string()) }
        }
        "cuda_backend" => {
            #[cfg(feature = "cuda")]
            { op_cuda_backend(qubits) }
            #[cfg(not(feature = "cuda"))]
            { Err("cuda backend not available".to_string()) }
        }
        "rocm_backend" => {
            #[cfg(feature = "rocm")]
            { op_rocm_backend(qubits) }
            #[cfg(not(feature = "rocm"))]
            { Err("rocm backend not available".to_string()) }
        }
        _ => Err(format!("unsupported op: {}", op)),
    };
    let elapsed = start.elapsed().as_millis();
    match result {
        Ok(detail) => {
            print_result(&op, qubits, size, bond, steps, elapsed, &detail);
        }
        Err(e) => {
            eprintln!("ERROR op={} qubits={} size={} bond={} steps={} err={}", op, qubits, size, bond, steps, e);
            std::process::exit(1);
        }
    }
}
