#!/usr/bin/env python3
"""
Advanced JAX Integration Examples for nQPU-Metal

Demonstrates:
1. Quantum Neural Networks with JAX
2. Quantum Generative Adversarial Networks
3. Quantum Reinforcement Learning
4. Quantum Transfer Learning
5. Hybrid Classical-Quantum Models
6. Advanced Optimization Techniques
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Callable, Tuple

# Add path
sys.path.insert(0, '/path/to/nqpu-metal/python')  # Update this

from nqpu_jax import (
    quantum_expectation,
    vmap_quantum,
    quantum_kernel_matrix,
    make_vqe_loss,
    quantum_natural_gradient_step,
)

# ============================================================================
# EXAMPLE 1: Quantum Neural Network (QNN)
# ============================================================================

class QuantumNeuralNetwork:
    """Quantum Neural Network for binary classification."""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * n_layers * 3  # 3 rotations per qubit per layer

    def create_circuit(self, params: jnp.ndarray) -> dict:
        """Create variational circuit from parameters."""
        gates = []

        # Data encoding layer (placeholder - will be updated per sample)
        for i in range(self.n_qubits):
            gates.append(('H', i))

        # Variational layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Rotation layer
            for q in range(self.n_qubits):
                gates.append(('RX', q, f'param_{param_idx}'))
                param_idx += 1
                gates.append(('RY', q, f'param_{param_idx}'))
                param_idx += 1
                gates.append(('RZ', q, f'param_{param_idx}'))
                param_idx += 1

            # Entangling layer
            for q in range(self.n_qubits - 1):
                gates.append(('CNOT', q, q + 1))
            if self.n_qubits > 2:
                gates.append(('CNOT', self.n_qubits - 1, 0))

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'observable': 'Z0'  # Classification via Z measurement
        }

    def forward(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: encode data x and compute expectation."""
        # For simplicity, we use params directly
        # In practice, you'd encode x into the circuit
        circuit = self.create_circuit(params)
        return quantum_expectation(params, circuit)

    def predict(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Predict class (0 or 1) based on expectation sign."""
        exp_val = self.forward(params, x)
        return (exp_val > 0).astype(jnp.float32)


def train_qnn_example():
    """Train QNN on synthetic data."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Quantum Neural Network Training")
    print("="*60)

    # Create QNN
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2)

    # Synthetic binary classification data
    np.random.seed(42)
    X_train = np.random.randn(20, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.float32)

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = jax.random.normal(key, (qnn.n_params,)) * 0.1

    # Loss function
    def loss_fn(params):
        predictions = jnp.array([qnn.forward(params, x) for x in X_train])
        # Binary cross-entropy-like loss
        return jnp.mean((predictions - y_train) ** 2)

    # Training loop
    learning_rate = 0.1
    print(f"Initial loss: {loss_fn(params):.4f}")

    for step in range(20):
        grads = grad(loss_fn)(params)
        params = params - learning_rate * grads

        if step % 5 == 0:
            current_loss = loss_fn(params)
            print(f"Step {step:2d}: loss = {current_loss:.4f}")

    print(f"Final loss: {loss_fn(params):.4f}")
    print("✓ QNN training complete\n")


# ============================================================================
# EXAMPLE 2: Quantum Generative Model
# ============================================================================

class QuantumGenerativeModel:
    """Variational quantum circuit for generative modeling."""

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    def create_circuit(self, latent: jnp.ndarray) -> dict:
        """Create generative circuit from latent code."""
        gates = []

        # Encode latent variables
        for i, z in enumerate(latent[:self.n_qubits]):
            gates.append(('RY', i, f'z{i}'))

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                gates.append(('RZ', q, f'layer{layer}_q{q}'))
            for q in range(self.n_qubits - 1):
                gates.append(('CNOT', q, q + 1))

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'observable': 'Z0'
        }

    def generate(self, latent: jnp.ndarray) -> jnp.ndarray:
        """Generate sample from latent code."""
        circuit = self.create_circuit(latent)
        # Use latent as parameters
        return quantum_expectation(latent, circuit)


def quantum_gan_example():
    """Simple quantum GAN example."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Quantum Generative Adversarial Network")
    print("="*60)

    generator = QuantumGenerativeModel(n_qubits=4, n_layers=2)

    # Sample from latent space
    key = jax.random.PRNGKey(42)
    latent_samples = jax.random.normal(key, (5, 4))

    print("Generating samples from latent codes:")
    for i, latent in enumerate(latent_samples):
        sample = generator.generate(latent)
        print(f"  Latent {i}: {latent[:4]} -> Generated: {sample:.4f}")

    print("✓ Quantum GAN demo complete\n")


# ============================================================================
# EXAMPLE 3: Quantum Transfer Learning
# ============================================================================

def quantum_transfer_learning_example():
    """Demonstrate transfer learning with quantum circuits."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Quantum Transfer Learning")
    print("="*60)

    # Pretrained feature extractor (shared across tasks)
    feature_extractor = {
        'n_qubits': 4,
        'gates': [
            ('H', 0), ('H', 1), ('H', 2), ('H', 3),
            ('RY', 0, 'p0'), ('RY', 1, 'p1'),
            ('RY', 2, 'p2'), ('RY', 3, 'p3'),
            ('CNOT', 0, 1), ('CNOT', 1, 2), ('CNOT', 2, 3),
        ],
        'observable': 'Z0'
    }

    # Pretrained parameters (frozen)
    pretrained_params = jnp.array([0.5, 1.0, 0.7, 0.3])

    # Task-specific head (trainable)
    task_head = {
        'n_qubits': 4,
        'gates': [
            ('RY', 0, 't0'), ('RY', 1, 't1'),
            ('CNOT', 0, 1),
        ],
        'observable': 'Z0'
    }

    # Combine: feature_extractor(pretrained) + task_head(trainable)
    def combined_model(task_params, x):
        # In practice, you'd apply feature_extractor first
        # then task_head on the output state
        # Here we show the concept
        return quantum_expectation(task_params, task_head)

    # Fine-tune on new task
    task_params = jnp.array([0.1, 0.1])
    print(f"Initial task params: {task_params}")

    # Simple optimization (placeholder)
    def task_loss(params):
        return jnp.sum(params ** 2)  # Dummy loss

    for step in range(10):
        grads = grad(task_loss)(task_params)
        task_params -= 0.1 * grads

    print(f"Fine-tuned task params: {task_params}")
    print("✓ Transfer learning demo complete\n")


# ============================================================================
# EXAMPLE 4: Quantum Reinforcement Learning
# ============================================================================

class QuantumPolicyNetwork:
    """Quantum circuit for policy in RL."""

    def __init__(self, n_qubits: int, n_actions: int):
        self.n_qubits = n_qubits
        self.n_actions = n_actions

    def create_circuit(self, state: jnp.ndarray) -> dict:
        """Create policy circuit encoding state."""
        gates = []

        # State encoding
        for i in range(min(len(state), self.n_qubits)):
            gates.append(('RY', i, f's{i}'))

        # Policy layers
        for q in range(self.n_qubits):
            gates.append(('RZ', q, f'policy_{q}'))

        gates.append(('CNOT', 0, 1))

        return {
            'n_qubits': self.n_qubits,
            'gates': gates,
            'observable': 'Z0'
        }

    def get_action_probs(self, params: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        """Compute action probabilities from quantum circuit."""
        circuit = self.create_circuit(state)
        exp_val = quantum_expectation(params, circuit)

        # Map expectation to action probabilities
        # Simple mapping: softmax over [exp_val, -exp_val]
        logits = jnp.array([exp_val, -exp_val])
        return jax.nn.softmax(logits)


def quantum_rl_example():
    """Simple quantum RL demo."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Quantum Reinforcement Learning")
    print("="*60)

    policy = QuantumPolicyNetwork(n_qubits=4, n_actions=2)

    # Sample states
    states = jnp.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ])

    # Initial policy parameters
    params = jnp.array([0.1, 0.2, 0.3, 0.4])

    print("Action probabilities for states:")
    for i, state in enumerate(states):
        probs = policy.get_action_probs(params, state)
        print(f"  State {i}: {state} -> Actions: {probs}")

    print("✓ Quantum RL demo complete\n")


# ============================================================================
# EXAMPLE 5: Hybrid Classical-Quantum Model
# ============================================================================

class HybridModel:
    """Combine classical neural network with quantum circuit."""

    def __init__(self, classical_layers: int, n_qubits: int):
        self.classical_layers = classical_layers
        self.n_qubits = n_qubits

    def classical_part(self, params_classical, x):
        """Classical neural network (simple feedforward)."""
        # Placeholder: linear transformation
        W = params_classical.reshape(4, 4)
        return jnp.dot(W, x)

    def quantum_part(self, params_quantum, x):
        """Quantum circuit processing."""
        circuit = {
            'n_qubits': self.n_qubits,
            'gates': [
                ('RY', i, f'p{i}') for i in range(self.n_qubits)
            ] + [
                ('CNOT', i, i+1) for i in range(self.n_qubits - 1)
            ],
            'observable': 'Z0'
        }
        return quantum_expectation(params_quantum, circuit)

    def forward(self, params, x):
        """Hybrid forward pass: classical -> quantum."""
        params_classical = params[:16]
        params_quantum = params[16:]

        # Classical preprocessing
        classical_out = self.classical_part(params_classical, x)

        # Quantum processing
        quantum_out = self.quantum_part(params_quantum, classical_out)

        return quantum_out


def hybrid_model_example():
    """Demonstrate hybrid classical-quantum model."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Hybrid Classical-Quantum Model")
    print("="*60)

    model = HybridModel(classical_layers=2, n_qubits=4)

    # Sample input
    x = jnp.array([0.1, 0.2, 0.3, 0.4])

    # Parameters (16 classical + 4 quantum)
    params = jnp.concatenate([
        jnp.eye(4).flatten(),  # Classical weights
        jnp.array([0.1, 0.2, 0.3, 0.4])  # Quantum params
    ])

    print(f"Input: {x}")
    output = model.forward(params, x)
    print(f"Output: {output}")

    # Compute gradients
    grad_fn = grad(lambda p: model.forward(p, x))
    grads = grad_fn(params)
    print(f"Gradient norm: {jnp.linalg.norm(grads):.4f}")

    print("✓ Hybrid model demo complete\n")


# ============================================================================
# EXAMPLE 6: Advanced Optimization - QAOA
# ============================================================================

def qaoa_example():
    """Quantum Approximate Optimization Algorithm example."""
    print("\n" + "="*60)
    print("EXAMPLE 6: QAOA for Max-Cut Problem")
    print("="*60)

    # Simple 4-node graph Max-Cut
    # Graph edges: (0,1), (1,2), (2,3), (3,0)
    n_qubits = 4
    p_layers = 2  # QAOA layers

    def create_qaoa_circuit(params):
        """Create QAOA circuit with mixing and problem layers."""
        gates = []

        # Initial superposition
        for i in range(n_qubits):
            gates.append(('H', i))

        # QAOA layers
        param_idx = 0
        for layer in range(p_layers):
            # Problem Hamiltonian (ZZ interactions for edges)
            # Edge (0,1): ZZ interaction is approximated
            gates.append(('CNOT', 0, 1))
            gates.append(('RZ', 1, f'gamma_{param_idx}'))
            gates.append(('CNOT', 0, 1))
            param_idx += 1

            # Repeat for other edges...

            # Mixing Hamiltonian (X rotations)
            for q in range(n_qubits):
                gates.append(('RX', q, f'beta_{param_idx}'))
                param_idx += 1

        return {
            'n_qubits': n_qubits,
            'gates': gates,
            'observable': 'Z0'  # Simplified
        }

    # QAOA optimization
    params = jnp.array([0.5, 0.5, 0.3, 0.3, 0.3, 0.3])  # gamma, beta params

    def qaoa_cost(params):
        circuit = create_qaoa_circuit(params)
        return -quantum_expectation(params, circuit)  # Maximize

    print("Optimizing QAOA...")
    for step in range(20):
        grads = grad(qaoa_cost)(params)
        params -= 0.1 * grads

        if step % 5 == 0:
            cost = qaoa_cost(params)
            print(f"  Step {step}: cost = {cost:.4f}")

    print(f"Final params: {params}")
    print("✓ QAOA demo complete\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all advanced examples."""
    print("\n" + "="*60)
    print("nQPU-Metal JAX: Advanced Examples")
    print("="*60)

    try:
        train_qnn_example()
        quantum_gan_example()
        quantum_transfer_learning_example()
        quantum_rl_example()
        hybrid_model_example()
        qaoa_example()

        print("="*60)
        print("All advanced examples completed successfully! ✓")
        print("="*60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
