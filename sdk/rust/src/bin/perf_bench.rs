use nqpu_metal::benchmark_suite::*;
use nqpu_metal::quantum_f32::{QuantumStateF32, GateOpsF32};

fn main() {
    println!("=== NQPU-METAL BENCHMARK SUITE ===\n");

    let results = run_full_suite();
    print_results(&results);

    println!("\n=== RAW GATE THROUGHPUT (Hadamard f64) ===");
    for n in [10, 15, 20, 22, 24] {
        let mut state = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        let iters = 1000;
        for _ in 0..iters {
            nqpu_metal::GateOperations::h(&mut state, 0);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let us_per_gate = elapsed * 1e6 / iters as f64;
        let dim = 1usize << n;
        let gflops = (dim as f64 * 6.0 * iters as f64) / elapsed / 1e9;
        println!("  n={:2} (2^{:2} = {:>10}): {:>8.3} us/gate  {:.1} GFlop/s",
            n, n, dim, us_per_gate, gflops);
    }

    // ============================================================
    // F32 vs F64 COMPARISON
    // ============================================================
    println!("\n=== F32 vs F64: HADAMARD THROUGHPUT ===");
    for n in [10, 15, 20, 22, 24] {
        let iters = 1000;
        let dim = 1usize << n;

        // f64
        let mut state64 = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            nqpu_metal::GateOperations::h(&mut state64, 0);
        }
        let elapsed_f64 = start.elapsed().as_secs_f64();
        let us_f64 = elapsed_f64 * 1e6 / iters as f64;

        // f32
        let mut state32 = QuantumStateF32::new(n);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            GateOpsF32::h(&mut state32, 0);
        }
        let elapsed_f32 = start.elapsed().as_secs_f64();
        let us_f32 = elapsed_f32 * 1e6 / iters as f64;

        let speedup = elapsed_f64 / elapsed_f32;
        let mem_f64 = (dim * 16) as f64 / (1024.0 * 1024.0);
        let mem_f32 = (dim * 8) as f64 / (1024.0 * 1024.0);

        println!("  n={:2}: f64={:>8.1}us  f32={:>8.1}us  speedup={:.2}x  mem: {:.0}MB→{:.0}MB",
            n, us_f64, us_f32, speedup, mem_f64, mem_f32);
    }

    println!("\n=== F32 vs F64: CNOT THROUGHPUT ===");
    for n in [10, 15, 20, 22, 24] {
        let iters = 1000;

        // f64
        let mut state64 = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            nqpu_metal::GateOperations::cnot(&mut state64, 0, 1);
        }
        let us_f64 = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        // f32
        let mut state32 = QuantumStateF32::new(n);
        let start = std::time::Instant::now();
        for _ in 0..iters {
            GateOpsF32::cnot(&mut state32, 0, 1);
        }
        let us_f32 = start.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let speedup = us_f64 / us_f32;
        println!("  n={:2}: f64={:>8.1}us  f32={:>8.1}us  speedup={:.2}x", n, us_f64, us_f32, speedup);
    }

    println!("\n=== F32 vs F64: BELL CIRCUIT ===");
    for n in [10, 15, 20, 22] {
        // f64
        let mut state64 = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        nqpu_metal::GateOperations::h(&mut state64, 0);
        for q in 0..n-1 {
            nqpu_metal::GateOperations::cnot(&mut state64, q, q+1);
        }
        let ms_f64 = start.elapsed().as_secs_f64() * 1000.0;

        // f32
        let mut state32 = QuantumStateF32::new(n);
        let start = std::time::Instant::now();
        GateOpsF32::h(&mut state32, 0);
        for q in 0..n-1 {
            GateOpsF32::cnot(&mut state32, q, q+1);
        }
        let ms_f32 = start.elapsed().as_secs_f64() * 1000.0;

        // Precision check
        let fidelity = state32.fidelity_vs_f64(&state64);
        let speedup = ms_f64 / ms_f32;

        println!("  n={:2}: f64={:>8.2}ms  f32={:>8.2}ms  speedup={:.2}x  fidelity={:.10}",
            n, ms_f64, ms_f32, speedup, fidelity);
    }

    println!("\n=== F32 PRECISION ANALYSIS ===");
    for depth in [10, 50, 100, 200] {
        let n = 10;

        // f64 reference
        let mut state64 = nqpu_metal::QuantumState::new(n);
        for _ in 0..depth {
            for q in 0..n {
                nqpu_metal::GateOperations::h(&mut state64, q);
            }
            for q in 0..n-1 {
                nqpu_metal::GateOperations::cnot(&mut state64, q, q+1);
            }
        }

        // f32
        let mut state32 = QuantumStateF32::new(n);
        for _ in 0..depth {
            for q in 0..n {
                GateOpsF32::h(&mut state32, q);
            }
            for q in 0..n-1 {
                GateOpsF32::cnot(&mut state32, q, q+1);
            }
        }

        let fidelity = state32.fidelity_vs_f64(&state64);
        println!("  depth={:>3} ({}q, {} gates): fidelity={:.10}",
            depth, n, depth * (n + n - 1), fidelity);
    }

    // ============================================================
    // 2-QUBIT FUSION BENCHMARKS
    // ============================================================
    println!("\n=== 2-QUBIT FUSION: QFT (with absorption) ===");
    use nqpu_metal::gate_fusion::{fuse_gates, execute_fused_circuit, FusedOrOriginal};
    for n in [8, 10, 12, 14, 16, 18, 20] {
        let gates = qft_circuit(n);
        let fusion = fuse_gates(&gates);

        // Count gate types
        let fused_1q = fusion.gates.iter().filter(|g| matches!(g, FusedOrOriginal::Fused(_))).count();
        let fused_2q = fusion.gates.iter().filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_))).count();
        let originals = fusion.gates.iter().filter(|g| matches!(g, FusedOrOriginal::Original(_))).count();

        // Time sequential
        let mut state_seq = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for gate in &gates {
            nqpu_metal::ascii_viz::apply_gate_to_state(&mut state_seq, gate);
        }
        let ms_seq = start.elapsed().as_secs_f64() * 1000.0;

        // Time fused (with 2-qubit absorption)
        let mut state_fused = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        execute_fused_circuit(&mut state_fused, &fusion);
        let ms_fused = start.elapsed().as_secs_f64() * 1000.0;

        let fidelity = state_fused.fidelity(&state_seq);
        let speedup = ms_seq / ms_fused;

        println!("  QFT-{:2}: {:>4} gates → {:>3} fused1q + {:>3} fused2q + {:>3} orig | seq={:>8.2}ms fused={:>8.2}ms speedup={:.2}x fid={:.10}",
            n, gates.len(), fused_1q, fused_2q, originals, ms_seq, ms_fused, speedup, fidelity);
    }

    println!("\n=== 2-QUBIT FUSION: RANDOM CIRCUITS ===");
    for n in [10, 15, 20] {
        let gates = random_circuit(n, 20, 42);
        let fusion = fuse_gates(&gates);

        let fused_2q = fusion.gates.iter().filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_))).count();

        let mut state_seq = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for gate in &gates {
            nqpu_metal::ascii_viz::apply_gate_to_state(&mut state_seq, gate);
        }
        let ms_seq = start.elapsed().as_secs_f64() * 1000.0;

        let mut state_fused = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        execute_fused_circuit(&mut state_fused, &fusion);
        let ms_fused = start.elapsed().as_secs_f64() * 1000.0;

        let fidelity = state_fused.fidelity(&state_seq);
        let speedup = ms_seq / ms_fused;

        println!("  Random-{:2}: {} gates, {} fused2q, elim={} | seq={:.2}ms fused={:.2}ms speedup={:.2}x fid={:.10}",
            n, gates.len(), fused_2q, fusion.gates_eliminated, ms_seq, ms_fused, speedup, fidelity);
    }

    println!("\n=== 2-QUBIT FUSION: GROVER ===");
    for n in [10, 14] {
        let target = (1 << n) / 2;
        let gates = grover_circuit(n, target);
        let fusion = fuse_gates(&gates);

        let fused_2q = fusion.gates.iter().filter(|g| matches!(g, FusedOrOriginal::FusedTwo(_))).count();

        let mut state_seq = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for gate in &gates {
            nqpu_metal::ascii_viz::apply_gate_to_state(&mut state_seq, gate);
        }
        let ms_seq = start.elapsed().as_secs_f64() * 1000.0;

        let mut state_fused = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        execute_fused_circuit(&mut state_fused, &fusion);
        let ms_fused = start.elapsed().as_secs_f64() * 1000.0;

        let fidelity = state_fused.fidelity(&state_seq);
        let speedup = ms_seq / ms_fused;

        println!("  Grover-{:2}: {} gates, {} fused2q, elim={} | seq={:.2}ms fused={:.2}ms speedup={:.2}x fid={:.10}",
            n, gates.len(), fused_2q, fusion.gates_eliminated, ms_seq, ms_fused, speedup, fidelity);
    }

    // ============================================================
    // COMBINED: f32 + 2-qubit fusion
    // ============================================================
    println!("\n=== COMBINED: F32 + FUSION (Grover-10) ===");
    {
        let n = 10;
        let target = (1 << n) / 2;
        let gates = grover_circuit(n, target);

        // f64 sequential (baseline)
        let mut state_baseline = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        for gate in &gates {
            nqpu_metal::ascii_viz::apply_gate_to_state(&mut state_baseline, gate);
        }
        let ms_baseline = start.elapsed().as_secs_f64() * 1000.0;

        // f64 fused
        let fusion = fuse_gates(&gates);
        let mut state_fused = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        execute_fused_circuit(&mut state_fused, &fusion);
        let ms_fused = start.elapsed().as_secs_f64() * 1000.0;

        // f32 sequential (approximate - just gate-by-gate)
        let mut state32 = QuantumStateF32::new(n);
        let start = std::time::Instant::now();
        // Simplified: H on all, then Grover iterations with f32 gates
        for q in 0..n { GateOpsF32::h(&mut state32, q); }
        let num_iters = ((std::f64::consts::PI / 4.0) * ((1 << n) as f64).sqrt()) as usize;
        for _ in 0..num_iters.max(1) {
            for q in 0..n {
                if target & (1 << q) == 0 { GateOpsF32::x(&mut state32, q); }
            }
            for q in 0..n-1 { GateOpsF32::z(&mut state32, q); } // simplified oracle
            for q in 0..n {
                if target & (1 << q) == 0 { GateOpsF32::x(&mut state32, q); }
            }
            for q in 0..n { GateOpsF32::h(&mut state32, q); }
            for q in 0..n { GateOpsF32::x(&mut state32, q); }
            for q in 0..n-1 { GateOpsF32::z(&mut state32, q); }
            for q in 0..n { GateOpsF32::x(&mut state32, q); }
            for q in 0..n { GateOpsF32::h(&mut state32, q); }
        }
        let ms_f32 = start.elapsed().as_secs_f64() * 1000.0;

        println!("  f64 seq:     {:>8.2} ms (baseline)", ms_baseline);
        println!("  f64 fused:   {:>8.2} ms ({:.2}x)", ms_fused, ms_baseline / ms_fused);
        println!("  f32 seq:     {:>8.2} ms ({:.2}x)", ms_f32, ms_baseline / ms_f32);
        println!("  gates: {} → {} after fusion (elim={})", gates.len(), fusion.gates.len(), fusion.gates_eliminated);
    }

    // Legacy sections
    println!("\n=== FUSION BENEFIT (H+T chains, 20 gates/qubit) ===");
    use nqpu_metal::gates::Gate;
    for n in [10, 15, 20] {
        let mut gates = Vec::new();
        for q in 0..n {
            for _ in 0..10 {
                gates.push(Gate::h(q));
                gates.push(Gate::t(q));
            }
        }
        let result = run_benchmark(&format!("Deep-1Q-{}", n), &gates, n);
        println!("  {}", result);
    }

    println!("\n=== SWAP THROUGHPUT (parallel) ===");
    for n in [10, 15, 20, 22] {
        let mut state = nqpu_metal::QuantumState::new(n);
        let start = std::time::Instant::now();
        let iters = 1000;
        for _ in 0..iters {
            nqpu_metal::GateOperations::swap(&mut state, 0, n-1);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let us_per_gate = elapsed * 1e6 / iters as f64;
        println!("  n={:2}: {:>8.3} us/gate", n, us_per_gate);
    }
}
