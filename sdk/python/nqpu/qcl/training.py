"""Training loops and objectives for quantum circuit learning.

Provides the main :class:`QCLTrainer` class that orchestrates the training
of parameterized quantum circuits for classification, regression, and
kernel-based tasks.  Includes built-in optimizers (Adam, SGD, SPSA) and
objective functions with loss computation.

Design principles
-----------------
- The trainer is optimizer-agnostic: it takes a cost function and a
  gradient method, then iterates.
- Objectives map raw circuit outputs (expectation values, probabilities)
  to task-specific scalar losses.
- Training history is recorded for analysis and early-stopping decisions.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .circuits import CircuitTemplate
from .gradients import (
    CostFn,
    FiniteDifferenceGradient,
    GradientResult,
    ParameterShiftRule,
)


# ------------------------------------------------------------------
# Training history
# ------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Container for training run metrics.

    Attributes
    ----------
    loss : list[float]
        Loss value at each epoch.
    accuracy : list[float]
        Accuracy at each epoch (classification only, empty otherwise).
    gradient_norms : list[float]
        Gradient L2 norm at each epoch.
    learning_rates : list[float]
        Effective learning rate at each epoch.
    """

    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    @property
    def best_loss(self) -> float:
        """Minimum loss achieved."""
        return min(self.loss) if self.loss else float("inf")

    @property
    def best_epoch(self) -> int:
        """Epoch index of the best loss."""
        if not self.loss:
            return 0
        return int(np.argmin(self.loss))

    def __repr__(self) -> str:
        n = len(self.loss)
        best = self.best_loss
        return f"TrainingHistory(epochs={n}, best_loss={best:.6g})"


# ------------------------------------------------------------------
# Objective functions
# ------------------------------------------------------------------


class ClassificationObjective:
    """Cross-entropy loss from measurement probabilities.

    Maps circuit output probabilities to a classification loss.
    For binary classification, uses the probability of the first qubit
    being |0> vs |1>.  For multi-class, uses the probabilities of the
    first ``n_classes`` computational basis states.

    Parameters
    ----------
    n_classes : int
        Number of classes (2 for binary).
    """

    def __init__(self, n_classes: int = 2) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes

    def loss(
        self,
        circuit: CircuitTemplate,
        x: np.ndarray,
        y: int,
        params: np.ndarray,
    ) -> float:
        """Compute cross-entropy loss for a single sample.

        Parameters
        ----------
        circuit : CircuitTemplate
            The QCL circuit.
        x : np.ndarray
            Input features.
        y : int
            True class label (0-indexed).
        params : np.ndarray
            Trainable parameters.

        Returns
        -------
        float
            Cross-entropy loss (lower is better).
        """
        probs = self._class_probs(circuit, x, params)
        # Clip for numerical stability
        eps = 1e-10
        p_true = np.clip(probs[y], eps, 1.0 - eps)
        return -math.log(p_true)

    def predict(
        self, circuit: CircuitTemplate, x: np.ndarray, params: np.ndarray
    ) -> int:
        """Predict the class label for a single input.

        Returns
        -------
        int
            Predicted class index.
        """
        probs = self._class_probs(circuit, x, params)
        return int(np.argmax(probs))

    def batch_loss(
        self,
        circuit: CircuitTemplate,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Average loss over a batch of samples."""
        total = 0.0
        for i in range(len(X)):
            total += self.loss(circuit, X[i], int(y[i]), params)
        return total / len(X)

    def batch_accuracy(
        self,
        circuit: CircuitTemplate,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Classification accuracy over a batch."""
        correct = 0
        for i in range(len(X)):
            if self.predict(circuit, X[i], params) == int(y[i]):
                correct += 1
        return correct / len(X)

    def _class_probs(
        self,
        circuit: CircuitTemplate,
        x: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Extract class probabilities from circuit output."""
        full_probs = circuit.probabilities(x, params)

        if self.n_classes == 2:
            # Binary: aggregate over first qubit
            n_qubits = circuit.n_qubits
            dim = 1 << n_qubits
            p0 = 0.0
            step = 1  # qubit 0
            for i in range(dim):
                if i & step == 0:
                    p0 += full_probs[i]
            return np.array([p0, 1.0 - p0])
        else:
            # Multi-class: first n_classes basis states
            probs = full_probs[: self.n_classes].copy()
            total = probs.sum()
            if total < 1e-15:
                return np.ones(self.n_classes) / self.n_classes
            return probs / total


class RegressionObjective:
    """MSE loss from expectation values.

    Uses the <Z> expectation value on a specified qubit as the circuit
    output, and computes mean squared error against the target.

    Parameters
    ----------
    output_qubit : int
        Which qubit's <Z> expectation to use as the model output.
    output_scale : float
        Scaling factor for the output (maps [-1,1] to target range).
    output_shift : float
        Shift added to scaled output.
    """

    def __init__(
        self,
        output_qubit: int = 0,
        output_scale: float = 1.0,
        output_shift: float = 0.0,
    ) -> None:
        self.output_qubit = output_qubit
        self.output_scale = output_scale
        self.output_shift = output_shift

    def predict_value(
        self,
        circuit: CircuitTemplate,
        x: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Compute the model output for a single input."""
        exp_z = circuit.expectation_z(x, params, qubit=self.output_qubit)
        return exp_z * self.output_scale + self.output_shift

    def loss(
        self,
        circuit: CircuitTemplate,
        x: np.ndarray,
        y_target: float,
        params: np.ndarray,
    ) -> float:
        """MSE loss for a single sample."""
        pred = self.predict_value(circuit, x, params)
        return (pred - y_target) ** 2

    def batch_loss(
        self,
        circuit: CircuitTemplate,
        X: np.ndarray,
        y: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Average MSE over a batch."""
        total = 0.0
        for i in range(len(X)):
            total += self.loss(circuit, X[i], float(y[i]), params)
        return total / len(X)


class KernelObjective:
    """Quantum kernel alignment objective.

    Maximizes the alignment between the quantum kernel matrix and
    the ideal kernel defined by the labels.  Used for training
    quantum kernel parameters.

    Parameters
    ----------
    regularization : float
        Regularization parameter for the alignment computation.
    """

    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = regularization

    def loss(
        self,
        kernel_matrix: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute negative kernel-target alignment (to minimize).

        Parameters
        ----------
        kernel_matrix : np.ndarray
            Gram matrix K[i,j] (shape (n, n)).
        y : np.ndarray
            Labels (integer or float).

        Returns
        -------
        float
            Negative alignment (lower is better).
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        Y = np.outer(y, y)

        # Frobenius inner product
        kta_num = np.sum(kernel_matrix * Y)
        kta_den = np.sqrt(np.sum(kernel_matrix ** 2) * np.sum(Y ** 2))
        if kta_den < 1e-15:
            return 0.0
        alignment = kta_num / kta_den
        return -alignment  # negate so minimizing = improving alignment


# ------------------------------------------------------------------
# Main trainer
# ------------------------------------------------------------------


class QCLTrainer:
    """Main training loop for quantum circuit learning.

    Supports classification, regression, and custom objectives.
    Includes built-in optimizers (Adam, SGD, SPSA) and can use
    gradient methods from :mod:`nqpu.qcl.gradients`.

    Parameters
    ----------
    circuit : CircuitTemplate
        The quantum circuit to train.
    objective : ClassificationObjective or RegressionObjective
        Task-specific loss computation.
    gradient_method : str
        ``'parameter_shift'`` (default) or ``'finite_difference'``.
    seed : int or None
        Random seed for parameter initialization and SPSA.
    """

    def __init__(
        self,
        circuit: CircuitTemplate,
        objective: ClassificationObjective | RegressionObjective,
        gradient_method: str = "parameter_shift",
        seed: int | None = None,
    ) -> None:
        self.circuit = circuit
        self.objective = objective
        self.gradient_method = gradient_method
        self.rng = np.random.default_rng(seed)
        self.params: np.ndarray | None = None

        if gradient_method == "parameter_shift":
            self._grad_engine = ParameterShiftRule()
        elif gradient_method == "finite_difference":
            self._grad_engine = FiniteDifferenceGradient(epsilon=1e-5)
        else:
            raise ValueError(
                f"Unknown gradient method: {gradient_method!r}. "
                f"Use 'parameter_shift' or 'finite_difference'."
            )

    def _init_params(self, init_params: np.ndarray | None = None) -> np.ndarray:
        """Initialize parameters, either from user input or randomly."""
        n = self.circuit.n_params
        if init_params is not None:
            p = np.asarray(init_params, dtype=np.float64)
            if len(p) != n:
                raise ValueError(f"Expected {n} params, got {len(p)}")
            return p.copy()
        return self.rng.uniform(0, 2 * np.pi, size=n).astype(np.float64)

    def _build_cost_fn(
        self, X: np.ndarray, y: np.ndarray
    ) -> CostFn:
        """Build a scalar cost function from data and objective."""
        if isinstance(self.objective, ClassificationObjective):
            def cost(params: np.ndarray) -> float:
                return self.objective.batch_loss(self.circuit, X, y, params)
        elif isinstance(self.objective, RegressionObjective):
            def cost(params: np.ndarray) -> float:
                return self.objective.batch_loss(self.circuit, X, y, params)
        else:
            raise TypeError(f"Unsupported objective type: {type(self.objective)}")
        return cost

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.1,
        optimizer: str = "adam",
        init_params: np.ndarray | None = None,
        early_stopping_patience: int = 0,
        lr_decay: float = 1.0,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> TrainingHistory:
        """Train the circuit parameters.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Training labels, shape (n_samples,).
        epochs : int
            Number of training epochs.
        lr : float
            Initial learning rate.
        optimizer : str
            ``'adam'``, ``'sgd'``, or ``'spsa'``.
        init_params : np.ndarray or None
            Initial parameters.  Randomly initialized if None.
        early_stopping_patience : int
            Stop after this many epochs without improvement (0 = disabled).
        lr_decay : float
            Multiply LR by this factor each epoch (1.0 = no decay).
        callback : callable or None
            ``callback(epoch, params, loss)`` hook.

        Returns
        -------
        TrainingHistory
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.params = self._init_params(init_params)
        cost_fn = self._build_cost_fn(X, y)
        history = TrainingHistory()

        # Optimizer state
        if optimizer == "adam":
            m = np.zeros_like(self.params)
            v = np.zeros_like(self.params)
            beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        elif optimizer == "sgd":
            pass
        elif optimizer == "spsa":
            pass
        else:
            raise ValueError(
                f"Unknown optimizer '{optimizer}'. Use 'adam', 'sgd', or 'spsa'."
            )

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            current_lr = lr * (lr_decay ** (epoch - 1))

            if optimizer == "spsa":
                # SPSA: simultaneous perturbation
                c_k = current_lr * 0.1
                delta = self.rng.choice([-1, 1], size=len(self.params))
                f_plus = cost_fn(self.params + c_k * delta)
                f_minus = cost_fn(self.params - c_k * delta)
                g_hat = (f_plus - f_minus) / (2.0 * c_k * delta)
                self.params = self.params - current_lr * g_hat
                grad_norm = float(np.linalg.norm(g_hat))
            else:
                # Gradient-based
                result = self._grad_engine.compute(cost_fn, self.params)
                grad = result.gradient
                grad_norm = float(np.linalg.norm(grad))

                if optimizer == "adam":
                    m = beta1 * m + (1.0 - beta1) * grad
                    v = beta2 * v + (1.0 - beta2) * grad ** 2
                    m_hat = m / (1.0 - beta1 ** epoch)
                    v_hat = v / (1.0 - beta2 ** epoch)
                    self.params = self.params - current_lr * m_hat / (
                        np.sqrt(v_hat) + eps_adam
                    )
                else:  # sgd
                    self.params = self.params - current_lr * grad

            loss_val = cost_fn(self.params)
            history.loss.append(loss_val)
            history.gradient_norms.append(grad_norm)
            history.learning_rates.append(current_lr)

            # Accuracy for classification
            if isinstance(self.objective, ClassificationObjective):
                acc = self.objective.batch_accuracy(
                    self.circuit, X, y, self.params
                )
                history.accuracy.append(acc)

            if callback is not None:
                callback(epoch, self.params.copy(), loss_val)

            # Early stopping
            if early_stopping_patience > 0:
                if loss_val < best_loss - 1e-8:
                    best_loss = loss_val
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for new inputs.

        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions (class labels or regression values).

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if self.params is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        predictions = []
        for i in range(len(X)):
            if isinstance(self.objective, ClassificationObjective):
                predictions.append(
                    self.objective.predict(self.circuit, X[i], self.params)
                )
            elif isinstance(self.objective, RegressionObjective):
                predictions.append(
                    self.objective.predict_value(
                        self.circuit, X[i], self.params
                    )
                )
        return np.array(predictions)
