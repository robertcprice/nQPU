"""Barren plateau detection and mitigation strategies.

Barren plateaus are the phenomenon where gradient magnitudes decrease
exponentially with circuit depth or system size, rendering variational
quantum algorithms untrainable.  This module provides:

1. **VarianceMonitor**: Online monitoring of gradient variance during
   training to detect onset of barren plateaus.
2. **IdentityBlockInit**: Parameter initialization strategy that keeps
   each circuit block close to the identity, preserving gradient flow.
3. **LayerwiseTraining**: Incremental training that adds layers one at
   a time, avoiding deep-circuit barren plateaus.
4. **BarrenPlateauAnalyzer**: Comprehensive analysis of gradient variance
   scaling with system size.

References
----------
- McClean et al., Nat. Commun. 9, 4812 (2018) [barren plateaus]
- Grant et al., Quantum 3, 214 (2019) [identity block initialization]
- Skolik et al., Quantum Machine Intelligence 3, 5 (2021) [layerwise training]
- Cerezo et al., Nat. Commun. 12, 1791 (2021) [cost function barren plateaus]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


# ------------------------------------------------------------------
# Variance monitoring
# ------------------------------------------------------------------


@dataclass
class VarianceStatus:
    """Status report from the variance monitor.

    Attributes
    ----------
    variance : float
        Current gradient variance (averaged over parameters).
    is_barren : bool
        Whether the gradient variance is below the barren plateau threshold.
    trend : str
        Direction of variance change: "decreasing", "stable", or "increasing".
    samples : int
        Number of gradient samples accumulated so far.
    """

    variance: float
    is_barren: bool
    trend: str  # "decreasing", "stable", "increasing"
    samples: int


@dataclass
class VarianceMonitor:
    """Monitor gradient variance during training to detect barren plateaus.

    Maintains a sliding window of gradient norms and computes running
    variance statistics.  Triggers a warning when the variance drops
    below the configured threshold, indicating that the optimizer is
    likely trapped in a barren plateau.

    Parameters
    ----------
    window_size : int
        Size of the sliding window for variance computation.
    threshold : float
        Variance below this value is flagged as barren.
    """

    window_size: int = 20
    threshold: float = 1e-6

    def __post_init__(self):
        self._history: List[np.ndarray] = []
        self._variance_values: List[float] = []

    def update(self, gradients: np.ndarray) -> VarianceStatus:
        """Add a gradient sample and check for barren plateau.

        Parameters
        ----------
        gradients : np.ndarray
            Gradient vector from the current training step.

        Returns
        -------
        VarianceStatus
            Current status including variance and trend information.
        """
        gradients = np.asarray(gradients, dtype=np.float64).ravel()
        self._history.append(gradients)

        # Compute variance over the sliding window
        window = self._history[-self.window_size:]
        if len(window) < 2:
            variance = float(np.var(gradients))
        else:
            grad_matrix = np.array(window)
            # Variance of each parameter across the window, then average
            per_param_var = np.var(grad_matrix, axis=0)
            variance = float(np.mean(per_param_var))

        self._variance_values.append(variance)

        # Determine trend
        trend = "stable"
        if len(self._variance_values) >= 3:
            recent = self._variance_values[-3:]
            if recent[-1] < recent[0] * 0.8:
                trend = "decreasing"
            elif recent[-1] > recent[0] * 1.2:
                trend = "increasing"

        return VarianceStatus(
            variance=variance,
            is_barren=variance < self.threshold,
            trend=trend,
            samples=len(self._history),
        )

    @property
    def is_barren(self) -> bool:
        """Whether the most recent variance is below threshold."""
        if not self._variance_values:
            return False
        return self._variance_values[-1] < self.threshold

    def variance_trajectory(self) -> np.ndarray:
        """Return the full history of variance values.

        Returns
        -------
        np.ndarray
            Array of variance values, one per update() call.
        """
        return np.array(self._variance_values, dtype=np.float64)

    def reset(self) -> None:
        """Clear all accumulated history."""
        self._history.clear()
        self._variance_values.clear()


# ------------------------------------------------------------------
# Identity block initialization
# ------------------------------------------------------------------


@dataclass
class IdentityBlockInit:
    """Initialize circuit parameters near identity to preserve gradient flow.

    Each block of parameters is initialized such that the corresponding
    unitary is close to the identity matrix.  For a block with
    Rx(a), Ry(b), Rz(c) rotations, setting a = b = c = 0 gives the
    identity.  Small random perturbations break symmetry while keeping
    the circuit near-identity, which helps avoid barren plateaus.

    Reference: Grant et al., "An initialization strategy for addressing
    barren plateaus in parametrized quantum circuits" (2019).

    Parameters
    ----------
    n_qubits : int
        Number of qubits per layer.
    n_layers : int
        Number of circuit layers.
    """

    n_qubits: int
    n_layers: int

    def initialize(self, rng=None) -> np.ndarray:
        """Generate parameters that make each layer approximately identity.

        For each layer, parameters are initialized as small perturbations
        around zero: params ~ N(0, epsilon^2) where epsilon scales
        inversely with the number of layers.

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        np.ndarray
            Parameter vector of shape (n_layers * n_qubits * 3,).
            Each group of 3 parameters corresponds to Rx, Ry, Rz angles
            for one qubit in one layer.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Perturbation scale: smaller for deeper circuits
        epsilon = 0.01 / max(self.n_layers, 1)
        n_params = self.n_layers * self.n_qubits * 3

        # Initialize near zero with paired cancellation
        params = np.zeros(n_params, dtype=np.float64)

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                base_idx = layer * self.n_qubits * 3 + qubit * 3
                # Small random perturbation around zero
                params[base_idx:base_idx + 3] = rng.normal(0, epsilon, size=3)

        return params

    def initialize_paired(self, rng=None) -> np.ndarray:
        """Generate paired parameters for identity-preserving initialization.

        Each pair of consecutive layers is initialized such that the
        second layer approximately cancels the first, making the combined
        two-layer block close to identity.

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        np.ndarray
            Parameter vector where pairs of layers cancel.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        params_per_layer = self.n_qubits * 3
        n_params = self.n_layers * params_per_layer
        params = np.zeros(n_params, dtype=np.float64)

        epsilon = 0.01
        for layer in range(0, self.n_layers, 2):
            # First layer: small random values
            random_block = rng.normal(0, epsilon, size=params_per_layer)
            start = layer * params_per_layer
            params[start:start + params_per_layer] = random_block

            # Second layer: negate to approximately cancel
            if layer + 1 < self.n_layers:
                start2 = (layer + 1) * params_per_layer
                perturbation = rng.normal(0, epsilon * 0.1, size=params_per_layer)
                params[start2:start2 + params_per_layer] = -random_block + perturbation

        return params


# ------------------------------------------------------------------
# Layerwise training
# ------------------------------------------------------------------


@dataclass
class LayerwiseResult:
    """Result of layerwise training.

    Attributes
    ----------
    params : np.ndarray
        Final parameter vector after all layers are trained.
    loss_history : list of float
        Loss values throughout the layerwise training process.
    layers_trained : int
        Number of layers successfully trained.
    variance_history : list of float
        Gradient variance at each training step.
    """

    params: np.ndarray
    loss_history: List[float] = field(default_factory=list)
    layers_trained: int = 0
    variance_history: List[float] = field(default_factory=list)


@dataclass
class LayerwiseTraining:
    """Train circuit layer-by-layer to avoid barren plateaus.

    Starts with 1 layer, trains it until convergence, then adds
    another layer (initialized near identity) and trains again.
    This builds up circuit depth gradually, maintaining trainability
    at each stage.

    Reference: Skolik et al., "Layerwise learning for quantum neural
    networks" (2021).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    max_layers : int
        Maximum number of layers to train.
    epochs_per_layer : int
        Number of training epochs for each new layer.
    """

    n_qubits: int
    max_layers: int
    epochs_per_layer: int = 50

    def train(
        self,
        loss_fn: Callable,
        grad_fn: Callable,
        rng=None,
    ) -> LayerwiseResult:
        """Execute layerwise training procedure.

        Parameters
        ----------
        loss_fn : callable
            ``loss_fn(params) -> float`` computes the loss for a given
            parameter vector.  The function must accept parameter vectors
            of varying length (corresponding to different numbers of layers).
        grad_fn : callable
            ``grad_fn(params) -> np.ndarray`` computes the gradient.
            Must accept the same variable-length parameter vectors.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        LayerwiseResult
            Final parameters, loss history, and diagnostic information.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        params_per_layer = self.n_qubits * 3  # Rx, Ry, Rz per qubit
        all_params = np.array([], dtype=np.float64)
        loss_history = []
        variance_history = []
        lr = 0.1

        for layer_idx in range(self.max_layers):
            # Add new layer parameters (near-identity initialization)
            epsilon = 0.01 / max(layer_idx + 1, 1)
            new_layer_params = rng.normal(0, epsilon, size=params_per_layer)
            all_params = np.concatenate([all_params, new_layer_params])

            # Adam state for current training phase
            m = np.zeros_like(all_params)
            v = np.zeros_like(all_params)
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

            for epoch in range(self.epochs_per_layer):
                grad = grad_fn(all_params)
                grad_var = float(np.var(grad))
                variance_history.append(grad_var)

                # Adam update
                step = layer_idx * self.epochs_per_layer + epoch + 1
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad ** 2
                m_hat = m / (1.0 - beta1 ** step)
                v_hat = v / (1.0 - beta2 ** step)
                all_params = all_params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

                current_loss = loss_fn(all_params)
                loss_history.append(current_loss)

        return LayerwiseResult(
            params=all_params,
            loss_history=loss_history,
            layers_trained=self.max_layers,
            variance_history=variance_history,
        )


# ------------------------------------------------------------------
# Barren plateau analyzer
# ------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Result of barren plateau analysis for a specific circuit.

    Attributes
    ----------
    mean_gradient : float
        Mean gradient magnitude across random initializations.
    gradient_variance : float
        Variance of gradient magnitudes.
    is_barren : bool
        Whether gradient variance indicates a barren plateau.
    severity : str
        Severity level: "none", "mild", or "severe".
    recommendation : str
        Human-readable recommendation for addressing the plateau.
    """

    mean_gradient: float
    gradient_variance: float
    is_barren: bool
    severity: str  # "none", "mild", "severe"
    recommendation: str


@dataclass
class ScalingResult:
    """Result of gradient variance scaling analysis.

    Attributes
    ----------
    qubit_counts : list of int
        Number of qubits tested.
    variances : list of float
        Gradient variance at each qubit count.
    scaling_exponent : float
        Exponent alpha from the fit: var ~ 2^(-alpha * n).
        Positive alpha indicates exponential decay (barren plateau).
    is_exponential_decay : bool
        True if the variance shows clear exponential decay.
    """

    qubit_counts: List[int]
    variances: List[float]
    scaling_exponent: float  # from fit: var ~ 2^(-alpha*n)
    is_exponential_decay: bool


@dataclass
class BarrenPlateauAnalyzer:
    """Comprehensive analysis of barren plateau risk for a circuit.

    Samples gradient vectors at many random parameter initializations
    and analyzes the variance statistics.  Can also perform scaling
    analysis to determine how gradient variance depends on the number
    of qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of circuit layers.
    n_samples : int
        Number of random initializations to sample.
    """

    n_qubits: int
    n_layers: int
    n_samples: int = 100

    def analyze(
        self,
        circuit_fn: Callable,
        rng=None,
    ) -> AnalysisResult:
        """Sample gradients and analyze variance scaling.

        Parameters
        ----------
        circuit_fn : callable
            ``circuit_fn(params) -> float`` cost function.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        AnalysisResult
            Comprehensive analysis including severity and recommendations.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_params = self.n_qubits * self.n_layers * 3
        shift = np.pi / 2
        denom = 2.0 * np.sin(shift)

        all_grads = np.zeros((self.n_samples, n_params), dtype=np.float64)

        for s in range(self.n_samples):
            params = rng.uniform(0, 2 * np.pi, size=n_params)
            grad = np.zeros(n_params)
            for i in range(n_params):
                e_i = np.zeros(n_params)
                e_i[i] = shift
                f_plus = circuit_fn(params + e_i)
                f_minus = circuit_fn(params - e_i)
                grad[i] = (f_plus - f_minus) / denom
            all_grads[s] = grad

        # Compute statistics
        mean_grad = float(np.mean(np.abs(all_grads)))
        per_param_var = np.var(all_grads, axis=0)
        gradient_variance = float(np.mean(per_param_var))

        # Determine severity
        if gradient_variance < 1e-8:
            severity = "severe"
            is_barren = True
            recommendation = (
                "Severe barren plateau detected. Gradient variance is extremely "
                "low. Consider: (1) reducing circuit depth, (2) using identity "
                "block initialization, (3) switching to layerwise training, or "
                "(4) using a local cost function."
            )
        elif gradient_variance < 1e-4:
            severity = "mild"
            is_barren = True
            recommendation = (
                "Mild barren plateau detected. Gradients are small but nonzero. "
                "Try: (1) identity block initialization, (2) increasing learning "
                "rate, or (3) reducing the number of layers."
            )
        else:
            severity = "none"
            is_barren = False
            recommendation = (
                "No barren plateau detected. Gradient variance is healthy. "
                "Proceed with standard training."
            )

        return AnalysisResult(
            mean_gradient=mean_grad,
            gradient_variance=gradient_variance,
            is_barren=is_barren,
            severity=severity,
            recommendation=recommendation,
        )

    def scaling_analysis(
        self,
        qubit_range: range = None,
        circuit_factory: Callable = None,
        rng=None,
    ) -> ScalingResult:
        """Study gradient variance vs number of qubits.

        For each qubit count, generates a cost function and samples
        gradient variances.  Then fits the data to determine whether
        the variance decays exponentially with qubit count.

        Parameters
        ----------
        qubit_range : range or None
            Range of qubit counts to test.  Defaults to range(2, n_qubits+1).
        circuit_factory : callable or None
            ``circuit_factory(n_qubits, n_layers) -> cost_fn``.
            Creates a cost function for the given qubit/layer configuration.
            If None, uses a simple parameterized rotation circuit.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        ScalingResult
            Variance values and scaling exponent.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        if qubit_range is None:
            qubit_range = range(2, self.n_qubits + 1)

        qubit_counts = list(qubit_range)
        variances = []
        shift = np.pi / 2
        denom = 2.0 * np.sin(shift)

        n_scaling_samples = min(self.n_samples, 50)

        for nq in qubit_counts:
            n_params = nq * self.n_layers * 3

            if circuit_factory is not None:
                cost_fn = circuit_factory(nq, self.n_layers)
            else:
                cost_fn = _default_cost_fn(nq, self.n_layers)

            # Sample gradients
            grad_samples = np.zeros((n_scaling_samples, n_params))
            for s in range(n_scaling_samples):
                params = rng.uniform(0, 2 * np.pi, size=n_params)
                grad = np.zeros(n_params)
                for i in range(n_params):
                    e_i = np.zeros(n_params)
                    e_i[i] = shift
                    f_plus = cost_fn(params + e_i)
                    f_minus = cost_fn(params - e_i)
                    grad[i] = (f_plus - f_minus) / denom
                grad_samples[s] = grad

            per_param_var = np.var(grad_samples, axis=0)
            mean_var = float(np.mean(per_param_var))
            variances.append(max(mean_var, 1e-20))

        # Fit exponential decay: log(var) = -alpha * n * log(2) + c
        # => linear regression of log(var) vs n
        log_variances = np.log(np.array(variances) + 1e-30)
        n_array = np.array(qubit_counts, dtype=np.float64)

        if len(n_array) >= 2:
            # Simple linear regression
            n_mean = n_array.mean()
            lv_mean = log_variances.mean()
            numerator = np.sum((n_array - n_mean) * (log_variances - lv_mean))
            denominator = np.sum((n_array - n_mean) ** 2)
            if abs(denominator) > 1e-15:
                slope = numerator / denominator
                # slope = -alpha * ln(2), so alpha = -slope / ln(2)
                scaling_exponent = -slope / np.log(2)
            else:
                scaling_exponent = 0.0
        else:
            scaling_exponent = 0.0

        # Determine if decay is exponential
        is_exponential = scaling_exponent > 0.5 and len(qubit_counts) >= 2

        return ScalingResult(
            qubit_counts=qubit_counts,
            variances=variances,
            scaling_exponent=float(scaling_exponent),
            is_exponential_decay=is_exponential,
        )


# ------------------------------------------------------------------
# Helper: default cost function for scaling analysis
# ------------------------------------------------------------------


def _default_cost_fn(n_qubits: int, n_layers: int) -> Callable:
    """Create a simple parameterized circuit cost function.

    Uses a hardware-efficient-style circuit with RY-RZ rotations
    and nearest-neighbor CNOT entangling, measuring <Z> on qubit 0.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of layers.

    Returns
    -------
    callable
        Cost function mapping parameter vector to scalar.
    """

    def cost_fn(params: np.ndarray) -> float:
        dim = 1 << n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0

        param_idx = 0
        for layer in range(n_layers):
            # RY rotations
            for q in range(n_qubits):
                angle = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                c = math.cos(angle / 2)
                s = math.sin(angle / 2)
                mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
                state = _apply_single_qubit(state, n_qubits, q, mat)

            # RZ rotations
            for q in range(n_qubits):
                angle = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                c = math.cos(angle / 2)
                s = math.sin(angle / 2)
                mat = np.array(
                    [[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128
                )
                state = _apply_single_qubit(state, n_qubits, q, mat)

            # CNOT chain
            for q in range(n_qubits - 1):
                state = _apply_cnot(state, n_qubits, q, q + 1)

            # Third rotation (Ry for third param per qubit)
            for q in range(n_qubits):
                angle = params[param_idx] if param_idx < len(params) else 0.0
                param_idx += 1
                c = math.cos(angle / 2)
                s = math.sin(angle / 2)
                mat = np.array([[c, -s], [s, c]], dtype=np.complex128)
                state = _apply_single_qubit(state, n_qubits, q, mat)

        # Expectation <Z> on qubit 0
        exp_val = 0.0
        for i in range(dim):
            p = abs(state[i]) ** 2
            if i & 1:
                exp_val -= p
            else:
                exp_val += p
        return exp_val

    return cost_fn


def _apply_single_qubit(
    state: np.ndarray, n_qubits: int, qubit: int, mat: np.ndarray
) -> np.ndarray:
    """Apply single-qubit gate to statevector."""
    dim = 1 << n_qubits
    new_state = np.zeros(dim, dtype=np.complex128)
    step = 1 << qubit
    for i in range(dim):
        if i & step == 0:
            j = i | step
            a, b = state[i], state[j]
            new_state[i] += mat[0, 0] * a + mat[0, 1] * b
            new_state[j] += mat[1, 0] * a + mat[1, 1] * b
    return new_state


def _apply_cnot(
    state: np.ndarray, n_qubits: int, control: int, target: int
) -> np.ndarray:
    """Apply CNOT gate."""
    dim = 1 << n_qubits
    new_state = state.copy()
    c_step = 1 << control
    t_step = 1 << target
    for i in range(dim):
        if (i & c_step) != 0 and (i & t_step) == 0:
            j = i | t_step
            new_state[i], new_state[j] = state[j], state[i]
    return new_state
