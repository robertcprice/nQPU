#!/usr/bin/env python3
"""
Comprehensive tests and examples for nQPU-Metal JAX integration.

This file demonstrates:
1. Basic circuit execution with gradients
2. VMAP batch processing
3. Quantum kernel matrix for QSVM
4. VQE optimization
5. Quantum natural gradient descent
6. Performance benchmarks
"""

import sys
import numpy as np
from typing import Dict, Any

# Check JAX installation
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    print(f"✓ JAX {jax.__version__} loaded successfully")
    HAS_JAX = True
except ImportError:
    print("✗ JAX not found. Install with: pip install jax jaxlib")
    HAS_JAX = False
    sys.exit(1)

# Check nqpu_jax
try:
    import nqpu_jax
    from nqpu_jax import (
        quantum_expectation,
        vmap_quantum,
        vmap_quantum_simulate,
        quantum_kernel_matrix,
        make_vqe_loss,
        quantum_natural_gradient_step,
        check_installation,
    )
    print(f"✓ nqpu_jax {nqpu_jax.get_version()} loaded successfully")
    status = check_installation()
    print(f"  - JAX: {status['jax']}")
    print(f"  - Rust bindings: {status['rust_bindings']}")
    if not status['rust_bindings']:
        print("\n✗ Rust bindings not available. Build with:")
        print("  cd /path/to/nqpu-metal")
        print("  maturin develop --release --features python")
        sys.exit(1)
except ImportError as e:
    print(f"✗ nqpu_jax import failed: {e}")
    print("\nMake sure python/nqpu_jax.py is in PYTHONPATH")
    sys.exit(1)

print()

# ============================================================================
# TEST 1: Basic Circuit with Gradients
# ============================================================================

def test_basic_gradients():
    """Test basic quantum circuit execution with automatic differentiation."""
    print("TEST 1: Basic Circuit with Gradients")
    print("=" * 60)

    # Define simple 2-qubit circuit
    circuit_config = {
        'n_qubits': 2,
        'gates': [
            ('H', 0),
            ('RY', 0, 'theta'),
            ('RY', 1, 'phi'),
            ('CNOT', 0, 1),
        ],
        'observable': 'Z0'
    }

    # Initial parameters
    params = jnp.array([0.5, 1.2])
    print(f"Initial params: {params}")

    # Forward pass
    exp_val = quantum_expectation(params, circuit_config)
    print(f"Expectation value: {exp_val:.6f}")

    # Compute gradients
    grad_fn = grad(quantum_expectation, argnums=0)
    grads = grad_fn(params, circuit_config)
    print(f"Gradients: {grads}")

    # Verify gradient is not zero (non-trivial circuit)
    assert np.linalg.norm(grads) > 0.01, "Gradient should be non-zero"

    print("✓ Test passed\n")


# ============================================================================
# TEST 2: VMAP Batch Processing
# ============================================================================

def test_vmap_batch():
    """Test vectorized batch execution using Rust VMAP."""
    print("TEST 2: VMAP Batch Processing")
    print("=" * 60)

    circuit_config = {
        'n_qubits': 2,
        'gates': [
            ('RY', 0, 'theta'),
            ('RY', 1, 'phi'),
        ],
        'observable': 'Z0'
    }

    # Create batch of parameters
    batch_params = jnp.array([
        [0.0, 0.0],
        [np.pi/4, np.pi/4],
        [np.pi/2, np.pi/2],
        [np.pi, np.pi],
    ])
    print(f"Batch shape: {batch_params.shape}")

    # Batched execution via Rust VMAP
    batch_fn = vmap_quantum(circuit_config, observable='Z0')
    results = batch_fn(batch_params)
    print(f"Results shape: {results.shape}")
    print(f"Results: {results}")

    # Verify results
    assert results.shape == (4,), "Results should have batch dimension"
    assert abs(results[0] - 1.0) < 0.01, "First result should be ~1.0"
    assert abs(results[3] - (-1.0)) < 0.01, "Last result should be ~-1.0"

    print("✓ Test passed\n")


# ============================================================================
# TEST 3: Quantum Kernel Matrix for QSVM
# ============================================================================

def test_quantum_kernel():
    """Test quantum kernel matrix computation for QSVM."""
    print("TEST 3: Quantum Kernel Matrix")
    print("=" * 60)

    # Feature map circuit (4 qubits, 4 parameters)
    circuit_config = {
        'n_qubits': 4,
        'gates': [
            ('H', 0), ('H', 1), ('H', 2), ('H', 3),
            ('RY', 0, 'x0'),
            ('RY', 1, 'x1'),
            ('RY', 2, 'x2'),
            ('RY', 3, 'x3'),
            ('CNOT', 0, 1),
            ('CNOT', 1, 2),
            ('CNOT', 2, 3),
        ]
    }

    # Sample data
    X_train = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ])
    X_test = jnp.array([
        [0.2, 0.3, 0.4, 0.5],
    ])

    print(f"Training samples: {X_train.shape}")
    print(f"Test samples: {X_test.shape}")

    # Compute kernel matrix
    K = quantum_kernel_matrix(X_train, X_test, circuit_config)
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Kernel values:\n{K}")

    # Verify properties
    assert K.shape == (2, 1), "Kernel shape should be (2, 1)"
    assert np.all(K >= 0) and np.all(K <= 1), "Kernel values in [0, 1]"

    print("✓ Test passed\n")


# ============================================================================
# TEST 4: VQE Optimization
# ============================================================================

def test_vqe():
    """Test Variational Quantum Eigensolver."""
    print("TEST 4: VQE Optimization")
    print("=" * 60)

    # Ansatz circuit
    ansatz = {
        'n_qubits': 2,
        'gates': [
            ('RY', 0, 'theta_0'),
            ('RY', 1, 'theta_1'),
            ('CNOT', 0, 1),
            ('RY', 0, 'theta_2'),
            ('RY', 1, 'theta_3'),
        ]
    }

    # Hamiltonian: H = 0.5*Z0 + 0.5*Z1 + 0.25*Z0Z1
    # Ground state should be |00⟩ with energy = 1.25
    hamiltonian = [
        ('Z0', 0.5),
        ('Z1', 0.5),
        # Note: Multi-qubit observables need proper parsing in Rust
    ]

    # Create VQE loss function
    loss_fn = make_vqe_loss(ansatz, hamiltonian)

    # Initial parameters (random)
    params = jnp.array([0.1, 0.2, 0.3, 0.4])
    print(f"Initial params: {params}")

    # Initial energy
    energy_0 = loss_fn(params)
    print(f"Initial energy: {energy_0:.6f}")

    # Gradient-based optimization (few steps)
    learning_rate = 0.1
    for step in range(10):
        grads = grad(loss_fn)(params)
        params = params - learning_rate * grads
        energy = loss_fn(params)
        if step % 3 == 0:
            print(f"  Step {step}: energy = {energy:.6f}")

    print(f"Final params: {params}")
    print(f"Final energy: {energy:.6f}")

    # Verify energy decreased
    assert energy < energy_0, "Energy should decrease"

    print("✓ Test passed\n")


# ============================================================================
# TEST 5: Quantum Natural Gradient
# ============================================================================

def test_quantum_natural_gradient():
    """Test quantum natural gradient descent."""
    print("TEST 5: Quantum Natural Gradient")
    print("=" * 60)

    circuit = {
        'n_qubits': 2,
        'gates': [
            ('RY', 0, 'theta_0'),
            ('RY', 1, 'theta_1'),
        ]
    }

    # Simple loss: minimize ⟨Z0⟩
    def loss_fn(params):
        return quantum_expectation(params, {**circuit, 'observable': 'Z0'})

    # Initial params
    params = jnp.array([0.5, 0.5])
    print(f"Initial params: {params}")
    print(f"Initial loss: {loss_fn(params):.6f}")

    # Natural gradient step
    params_new = quantum_natural_gradient_step(
        params,
        loss_fn,
        circuit,
        learning_rate=0.1,
        regularization=1e-4
    )

    print(f"Updated params: {params_new}")
    print(f"Updated loss: {loss_fn(params_new):.6f}")

    # Verify loss decreased
    assert loss_fn(params_new) < loss_fn(params), "Loss should decrease"

    print("✓ Test passed\n")


# ============================================================================
# TEST 6: JIT Compilation
# ============================================================================

def test_jit():
    """Test that JAX JIT compilation works."""
    print("TEST 6: JIT Compilation")
    print("=" * 60)

    circuit = {
        'n_qubits': 2,
        'gates': [
            ('RY', 0, 'theta'),
            ('RY', 1, 'phi'),
        ],
        'observable': 'Z0'
    }

    # Create JIT-compiled function
    @jit
    def jit_expectation(params):
        return quantum_expectation(params, circuit)

    params = jnp.array([0.5, 1.2])

    # First call (compilation + execution)
    result_1 = jit_expectation(params)
    print(f"First call result: {result_1:.6f}")

    # Second call (cached, should be fast)
    result_2 = jit_expectation(params)
    print(f"Second call result: {result_2:.6f}")

    # Results should match
    assert abs(result_1 - result_2) < 1e-6, "Results should match"

    print("✓ Test passed\n")


# ============================================================================
# TEST 7: Statevector VMAP
# ============================================================================

def test_statevector_vmap():
    """Test batched statevector simulation."""
    print("TEST 7: Statevector VMAP")
    print("=" * 60)

    circuit = {
        'n_qubits': 2,
        'gates': [
            ('RY', 0, 'theta'),
        ]
    }

    batch_params = jnp.array([[0.0], [np.pi/2], [np.pi]])
    print(f"Batch params shape: {batch_params.shape}")

    # Batched statevector simulation
    batch_fn = vmap_quantum_simulate(circuit)
    states = batch_fn(batch_params)

    print(f"States shape: {states.shape}")  # Should be (3, 4) for 2 qubits
    print(f"State norms: {jnp.linalg.norm(states, axis=1)}")

    # Verify normalization
    norms = jnp.linalg.norm(states, axis=1)
    assert jnp.allclose(norms, 1.0, atol=1e-5), "States should be normalized"

    # Check specific states
    # theta=0 -> |00⟩
    assert abs(states[0, 0] - 1.0) < 1e-5, "First state should be |00⟩"

    # theta=π -> |10⟩ (qubit 0 flipped)
    assert abs(states[2, 2] - 1.0) < 1e-5, "Last state should be |10⟩"

    print("✓ Test passed\n")


# ============================================================================
# BENCHMARK: Performance Comparison
# ============================================================================

def benchmark_performance():
    """Benchmark JAX integration performance."""
    print("BENCHMARK: Performance")
    print("=" * 60)

    import time

    circuit = {
        'n_qubits': 4,
        'gates': [
            ('RY', 0, 'p0'),
            ('RY', 1, 'p1'),
            ('RY', 2, 'p2'),
            ('RY', 3, 'p3'),
            ('CNOT', 0, 1),
            ('CNOT', 1, 2),
            ('CNOT', 2, 3),
        ],
        'observable': 'Z0'
    }

    # Single execution
    params = jnp.array([0.1, 0.2, 0.3, 0.4])
    n_runs = 1000

    start = time.time()
    for _ in range(n_runs):
        _ = quantum_expectation(params, circuit)
    elapsed_single = time.time() - start

    print(f"Single execution: {n_runs} runs in {elapsed_single:.3f}s")
    print(f"  -> {n_runs/elapsed_single:.1f} executions/sec")

    # Batch execution
    batch_size = 100
    batch_params = jnp.tile(params, (batch_size, 1))
    batch_fn = vmap_quantum(circuit)

    start = time.time()
    for _ in range(n_runs // batch_size):
        _ = batch_fn(batch_params)
    elapsed_batch = time.time() - start

    print(f"Batch execution: {n_runs} params in {elapsed_batch:.3f}s")
    print(f"  -> {n_runs/elapsed_batch:.1f} executions/sec")
    print(f"  -> {elapsed_single/elapsed_batch:.2f}x speedup")

    # Gradient computation
    grad_fn = grad(quantum_expectation, argnums=0)

    start = time.time()
    for _ in range(n_runs // 10):
        _ = grad_fn(params, circuit)
    elapsed_grad = time.time() - start

    print(f"Gradient computation: {n_runs//10} runs in {elapsed_grad:.3f}s")
    print(f"  -> {(n_runs//10)/elapsed_grad:.1f} grads/sec")

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("nQPU-Metal JAX Integration Test Suite")
    print("=" * 60 + "\n")

    try:
        test_basic_gradients()
        test_vmap_batch()
        test_quantum_kernel()
        test_vqe()
        test_quantum_natural_gradient()
        test_jit()
        test_statevector_vmap()
        benchmark_performance()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
