"""
JAX Integration for nQPU-Metal Quantum Simulator

Provides JAX-compatible quantum circuit execution with automatic differentiation
via parameter-shift rule, VMAP support for batch execution, and JIT compilation.

Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from nqpu_jax import quantum_expectation, vmap_quantum, quantum_kernel_matrix
    >>>
    >>> # Define circuit configuration
    >>> circuit_config = {
    >>>     'n_qubits': 4,
    >>>     'gates': [
    >>>         ('H', 0),
    >>>         ('RY', 0, 'param_0'),
    >>>         ('RY', 1, 'param_1'),
    >>>         ('CNOT', 0, 1),
    >>>     ],
    >>>     'observable': 'Z0'
    >>> }
    >>>
    >>> # Compute expectation with gradients
    >>> params = jnp.array([0.5, 1.2])
    >>> exp_val = quantum_expectation(params, circuit_config)
    >>> grad_fn = jax.grad(quantum_expectation, argnums=0)
    >>> grads = grad_fn(params, circuit_config)
    >>>
    >>> # Batch execution via VMAP
    >>> batch_params = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    >>> batch_fn = vmap_quantum(circuit_config)
    >>> batch_results = batch_fn(batch_params)
    >>>
    >>> # Quantum kernel matrix for QSVM
    >>> @jax.jit
    >>> def kernel_fn(x1, x2):
    >>>     return quantum_kernel_matrix(x1, x2, circuit_config)
    >>> K = kernel_fn(batch_params[:2], batch_params[2:])
"""

from typing import Dict, List, Tuple, Any, Callable, Optional
import warnings

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn(
        "JAX not found. Install with: pip install jax jaxlib\n"
        "For GPU support: pip install jax[cuda12]",
        ImportWarning
    )

try:
    # Import nqpu-metal Rust bindings (built via maturin)
    from nqpu_metal import (
        PyJAXCircuit,
        py_jax_simulate,
        py_jax_expectation,
        py_jax_gradient,
        py_jax_vmap_simulate,
        py_jax_vmap_expectation,
    )
    HAS_RUST_BINDINGS = True
except ImportError:
    HAS_RUST_BINDINGS = False
    warnings.warn(
        "nqpu_metal Python bindings not found. Build with:\n"
        "  cd /path/to/nqpu-metal\n"
        "  maturin develop --release --features python",
        ImportWarning
    )

import numpy as np

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

CircuitConfig = Dict[str, Any]
"""Circuit configuration dictionary with keys:
- n_qubits: int - number of qubits
- gates: List[Tuple] - gate specifications
- observable: str - observable to measure (e.g., 'Z0', 'Z0Z1')
"""

# ============================================================================
# CIRCUIT COMPILATION
# ============================================================================

class JAXCircuitCompiler:
    """Compiles circuit configuration into Rust JAXCircuit object."""

    _cache = {}  # JIT cache for compiled circuits

    @staticmethod
    def compile(config: CircuitConfig) -> 'PyJAXCircuit':
        """Compile circuit config to Rust circuit object.

        Args:
            config: Circuit configuration dictionary

        Returns:
            Compiled PyJAXCircuit object

        Raises:
            ValueError: If circuit config is invalid
        """
        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Create cache key from config
        cache_key = JAXCircuitCompiler._make_cache_key(config)
        if cache_key in JAXCircuitCompiler._cache:
            return JAXCircuitCompiler._cache[cache_key]

        # Validate config
        if 'n_qubits' not in config:
            raise ValueError("Circuit config must have 'n_qubits' field")
        if 'gates' not in config:
            raise ValueError("Circuit config must have 'gates' field")

        n_qubits = config['n_qubits']
        gates = config['gates']

        # Create Rust circuit
        circuit = PyJAXCircuit(n_qubits)

        # Add gates
        for gate_spec in gates:
            gate_type = gate_spec[0].upper()

            if gate_type == 'H':
                circuit.h(gate_spec[1])
            elif gate_type == 'X':
                circuit.x(gate_spec[1])
            elif gate_type == 'Y':
                circuit.y(gate_spec[1])
            elif gate_type == 'Z':
                circuit.z(gate_spec[1])
            elif gate_type == 'RX':
                qubit, param_name = gate_spec[1], gate_spec[2]
                circuit.rx(qubit, param_name)
            elif gate_type == 'RY':
                qubit, param_name = gate_spec[1], gate_spec[2]
                circuit.ry(qubit, param_name)
            elif gate_type == 'RZ':
                qubit, param_name = gate_spec[1], gate_spec[2]
                circuit.rz(qubit, param_name)
            elif gate_type in ['CNOT', 'CX']:
                control, target = gate_spec[1], gate_spec[2]
                circuit.cx(control, target)
            elif gate_type == 'CZ':
                control, target = gate_spec[1], gate_spec[2]
                circuit.cz(control, target)
            else:
                raise ValueError(f"Unknown gate type: {gate_type}")

        # Cache compiled circuit
        JAXCircuitCompiler._cache[cache_key] = circuit
        return circuit

    @staticmethod
    def _make_cache_key(config: CircuitConfig) -> str:
        """Create hashable cache key from config."""
        n_qubits = config['n_qubits']
        gates_str = str(config['gates'])
        return f"{n_qubits}_{gates_str}"

# ============================================================================
# CORE JAX INTEGRATION - EXPECTATION VALUE WITH CUSTOM VJP
# ============================================================================

if HAS_JAX:

    @custom_vjp
    def quantum_expectation(params: jnp.ndarray, circuit_config: CircuitConfig) -> jnp.ndarray:
        """Compute quantum expectation value with automatic differentiation.

        This function uses JAX's custom_vjp to provide exact gradients via the
        parameter-shift rule, computed efficiently in Rust.

        Args:
            params: Parameter array of shape (n_params,)
            circuit_config: Circuit configuration dictionary

        Returns:
            Expectation value as scalar JAX array

        Example:
            >>> config = {
            ...     'n_qubits': 2,
            ...     'gates': [('RY', 0, 'theta'), ('RY', 1, 'phi'), ('CNOT', 0, 1)],
            ...     'observable': 'Z0'
            ... }
            >>> params = jnp.array([0.5, 1.2])
            >>> exp = quantum_expectation(params, config)
            >>> grad = jax.grad(quantum_expectation, argnums=0)(params, config)
        """
        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Compile circuit
        circuit = JAXCircuitCompiler.compile(circuit_config)

        # Get observable
        observable = circuit_config.get('observable', 'Z0')

        # Convert JAX array to numpy for Rust call
        params_np = np.asarray(params, dtype=np.float32)

        # Call Rust expectation computation
        exp_val = py_jax_expectation(circuit, params_np, observable)

        # Return as JAX array
        return jnp.asarray(exp_val, dtype=params.dtype)


    def quantum_expectation_fwd(params: jnp.ndarray, circuit_config: CircuitConfig) -> Tuple[jnp.ndarray, Tuple]:
        """Forward pass: compute expectation and save residuals for backward.

        Args:
            params: Parameter array
            circuit_config: Circuit configuration

        Returns:
            (expectation_value, residuals) where residuals = (params, circuit_config)
        """
        exp_val = quantum_expectation(params, circuit_config)
        residuals = (params, circuit_config)
        return exp_val, residuals


    def quantum_expectation_bwd(residuals: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
        """Backward pass: compute parameter-shift gradients via Rust.

        Args:
            residuals: Saved (params, circuit_config) from forward pass
            g: Gradient w.r.t. output (cotangent)

        Returns:
            (grad_params, None) where None is for circuit_config (non-differentiable)
        """
        params, circuit_config = residuals

        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Compile circuit
        circuit = JAXCircuitCompiler.compile(circuit_config)

        # Get observable qubit (parse from observable string)
        observable = circuit_config.get('observable', 'Z0')
        qubit = int(observable.replace('Z', '').replace('X', '').replace('Y', '')[0])

        # Convert to numpy
        params_np = np.asarray(params, dtype=np.float32)

        # Call Rust gradient computation (parameter-shift rule)
        grads_np = py_jax_gradient(circuit, params_np, qubit)

        # Convert back to JAX array and apply chain rule
        grads = jnp.asarray(grads_np, dtype=params.dtype)
        grads = grads * g  # Chain rule with upstream gradient

        return (grads, None)  # None for circuit_config (not differentiable)


    # Register custom VJP
    quantum_expectation.defvjp(quantum_expectation_fwd, quantum_expectation_bwd)


# ============================================================================
# VMAP SUPPORT - BATCH EXECUTION
# ============================================================================

if HAS_JAX:

    def vmap_quantum(circuit_config: CircuitConfig, observable: str = 'Z0') -> Callable:
        """Create a vmapped quantum expectation function.

        This function returns a batched version that uses Rust's efficient
        batch execution instead of JAX's default sequential vmap.

        Args:
            circuit_config: Circuit configuration
            observable: Observable to measure (default: 'Z0')

        Returns:
            Batched function that maps (batch_size, n_params) -> (batch_size,)

        Example:
            >>> config = {'n_qubits': 2, 'gates': [('RY', 0, 'theta')]}
            >>> batch_fn = vmap_quantum(config)
            >>> batch_params = jnp.array([[0.1], [0.2], [0.3]])
            >>> results = batch_fn(batch_params)  # shape: (3,)
        """
        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Compile circuit once
        circuit = JAXCircuitCompiler.compile(circuit_config)

        def batched_fn(batch_params: jnp.ndarray) -> jnp.ndarray:
            """Batched expectation computation.

            Args:
                batch_params: Array of shape (batch_size, n_params)

            Returns:
                Array of shape (batch_size,) with expectation values
            """
            # Convert to numpy
            batch_params_np = np.asarray(batch_params, dtype=np.float32)

            # Call Rust batch execution
            results_np = py_jax_vmap_expectation(
                circuit,
                batch_params_np,
                observable
            )

            # Convert back to JAX
            return jnp.asarray(results_np, dtype=batch_params.dtype)

        return batched_fn


    def vmap_quantum_simulate(circuit_config: CircuitConfig) -> Callable:
        """Create a vmapped quantum statevector simulation function.

        Args:
            circuit_config: Circuit configuration

        Returns:
            Batched function that maps (batch_size, n_params) -> (batch_size, 2^n_qubits)

        Example:
            >>> config = {'n_qubits': 2, 'gates': [('RY', 0, 'theta')]}
            >>> batch_fn = vmap_quantum_simulate(config)
            >>> batch_params = jnp.array([[0.1], [0.2]])
            >>> states = batch_fn(batch_params)  # shape: (2, 4) complex
        """
        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Compile circuit
        circuit = JAXCircuitCompiler.compile(circuit_config)

        def batched_fn(batch_params: jnp.ndarray) -> jnp.ndarray:
            """Batched statevector simulation.

            Args:
                batch_params: Array of shape (batch_size, n_params)

            Returns:
                Array of shape (batch_size, 2^n_qubits) with complex amplitudes
            """
            # Convert to numpy
            batch_params_np = np.asarray(batch_params, dtype=np.float32)

            # Call Rust batch simulation (returns complex arrays)
            results_np = py_jax_vmap_simulate(circuit, batch_params_np)

            # Convert back to JAX (complex64)
            return jnp.asarray(results_np, dtype=jnp.complex64)

        return batched_fn


# ============================================================================
# QUANTUM KERNEL MATRIX - FOR QSVM
# ============================================================================

if HAS_JAX:

    @jax.jit
    def quantum_kernel_matrix(
        params1: jnp.ndarray,
        params2: jnp.ndarray,
        circuit_config: CircuitConfig
    ) -> jnp.ndarray:
        """Compute quantum kernel matrix K[i,j] = |⟨ψ(x_i)|ψ(x_j)⟩|².

        This is the quantum kernel used in Quantum Support Vector Machines (QSVM).
        The kernel measures the overlap between quantum states prepared with
        different parameter sets.

        Args:
            params1: Parameter array of shape (n_samples1, n_params)
            params2: Parameter array of shape (n_samples2, n_params)
            circuit_config: Circuit configuration for state preparation

        Returns:
            Kernel matrix of shape (n_samples1, n_samples2)

        Example:
            >>> config = {
            ...     'n_qubits': 4,
            ...     'gates': [('RY', i, f'param_{i}') for i in range(4)]
            ... }
            >>> X_train = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
            >>> X_test = jnp.array([[0.2, 0.3, 0.4, 0.5]])
            >>> K = quantum_kernel_matrix(X_train, X_test, config)
            >>> # K.shape = (2, 1)
        """
        if not HAS_RUST_BINDINGS:
            raise RuntimeError("Rust bindings not available")

        # Get batched statevector function
        batch_simulate = vmap_quantum_simulate(circuit_config)

        # Simulate both batches
        states1 = batch_simulate(params1)  # (n1, 2^n)
        states2 = batch_simulate(params2)  # (n2, 2^n)

        # Compute overlap matrix: K[i,j] = |⟨ψ_i|ψ_j⟩|²
        # ⟨ψ_i|ψ_j⟩ = sum_k conj(ψ_i[k]) * ψ_j[k]
        overlaps = jnp.abs(jnp.dot(jnp.conj(states1), states2.T)) ** 2

        return overlaps


# ============================================================================
# VARIATIONAL QUANTUM EIGENSOLVER (VQE) HELPER
# ============================================================================

if HAS_JAX:

    def make_vqe_loss(circuit_config: CircuitConfig, hamiltonian_terms: List[Tuple[str, float]]):
        """Create a VQE loss function for a given Hamiltonian.

        Args:
            circuit_config: Variational ansatz circuit configuration
            hamiltonian_terms: List of (observable, coefficient) tuples
                Example: [('Z0', 0.5), ('Z0Z1', 0.3), ('X0X1', 0.2)]

        Returns:
            Loss function that maps params -> energy (scalar)

        Example:
            >>> ansatz = {
            ...     'n_qubits': 2,
            ...     'gates': [
            ...         ('RY', 0, 'theta_0'),
            ...         ('RY', 1, 'theta_1'),
            ...         ('CNOT', 0, 1),
            ...         ('RY', 0, 'theta_2'),
            ...         ('RY', 1, 'theta_3'),
            ...     ]
            ... }
            >>> H = [('Z0', 0.5), ('Z1', 0.5), ('Z0Z1', 0.25)]
            >>> loss_fn = make_vqe_loss(ansatz, H)
            >>> params = jnp.array([0.1, 0.2, 0.3, 0.4])
            >>> energy = loss_fn(params)
            >>> grad_fn = jax.grad(loss_fn)
            >>> grads = grad_fn(params)
        """

        @jax.jit
        def vqe_loss(params: jnp.ndarray) -> jnp.ndarray:
            """Compute VQE energy: E = sum_i c_i * ⟨ψ|H_i|ψ⟩"""
            energy = 0.0

            for observable, coefficient in hamiltonian_terms:
                # Create config for this term
                term_config = circuit_config.copy()
                term_config['observable'] = observable

                # Compute expectation
                exp_val = quantum_expectation(params, term_config)
                energy += coefficient * exp_val

            return energy

        return vqe_loss


# ============================================================================
# QUANTUM NATURAL GRADIENT
# ============================================================================

if HAS_JAX:

    def quantum_fisher_information(
        params: jnp.ndarray,
        circuit_config: CircuitConfig
    ) -> jnp.ndarray:
        """Compute quantum Fisher information matrix (FIM).

        The FIM captures the geometry of the parameter space and is used
        in quantum natural gradient optimization.

        Args:
            params: Parameter array of shape (n_params,)
            circuit_config: Circuit configuration

        Returns:
            Fisher information matrix of shape (n_params, n_params)

        Note:
            This uses the parameter-shift rule to compute the FIM efficiently.
        """
        n_params = len(params)

        # Compute gradients for each parameter
        def grad_i(i):
            def f(p):
                return quantum_expectation(p, circuit_config)
            return jax.grad(f, argnums=0)(params)[i]

        # FIM[i,j] = Re[⟨∂_i ψ|∂_j ψ⟩] (for pure states)
        # Approximated via parameter-shift gradients
        grads = jnp.array([grad_i(i) for i in range(n_params)])

        # Outer product approximation
        fim = jnp.outer(grads, grads)

        return fim


    def quantum_natural_gradient_step(
        params: jnp.ndarray,
        loss_fn: Callable,
        circuit_config: CircuitConfig,
        learning_rate: float = 0.01,
        regularization: float = 1e-4
    ) -> jnp.ndarray:
        """Perform one quantum natural gradient descent step.

        Args:
            params: Current parameters
            loss_fn: Loss function to minimize
            circuit_config: Circuit configuration (for FIM)
            learning_rate: Step size
            regularization: Regularization for FIM inversion

        Returns:
            Updated parameters

        Example:
            >>> params = jnp.array([0.1, 0.2])
            >>> circuit = {'n_qubits': 2, 'gates': [...]}
            >>> loss = make_vqe_loss(circuit, [('Z0', 1.0)])
            >>> new_params = quantum_natural_gradient_step(params, loss, circuit)
        """
        # Compute standard gradient
        grad = jax.grad(loss_fn)(params)

        # Compute quantum Fisher information
        fim = quantum_fisher_information(params, circuit_config)

        # Regularize FIM for numerical stability
        fim_reg = fim + regularization * jnp.eye(len(params))

        # Natural gradient: F^{-1} ∇L
        natural_grad = jnp.linalg.solve(fim_reg, grad)

        # Update parameters
        params_new = params - learning_rate * natural_grad

        return params_new


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_installation() -> Dict[str, bool]:
    """Check if JAX and Rust bindings are properly installed.

    Returns:
        Dictionary with installation status
    """
    return {
        'jax': HAS_JAX,
        'rust_bindings': HAS_RUST_BINDINGS,
    }


def get_version() -> str:
    """Get nqpu_jax version string."""
    return "0.1.0"


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Core functions
    'quantum_expectation',
    'vmap_quantum',
    'vmap_quantum_simulate',
    'quantum_kernel_matrix',

    # VQE
    'make_vqe_loss',

    # Quantum natural gradient
    'quantum_fisher_information',
    'quantum_natural_gradient_step',

    # Utilities
    'check_installation',
    'get_version',
    'JAXCircuitCompiler',
]
