use criterion::{criterion_group, criterion_main, Criterion};
use nqpu_metal::{GateOperations, QuantumState};

fn bench_quantum_circuit(c: &mut Criterion) {
    c.bench_function("hadamard_10qubits", |b| {
        b.iter(|| {
            let mut state = QuantumState::new(10);
            for q in 0..10 {
                GateOperations::h(&mut state, q);
            }
            state.probabilities()
        })
    });
}

criterion_group!(benches, bench_quantum_circuit);
criterion_main!(benches);
