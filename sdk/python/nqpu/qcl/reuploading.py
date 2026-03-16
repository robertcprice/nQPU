"""Data re-uploading: single-qubit universal quantum classifier.

Implements the data re-uploading strategy from Perez-Salinas et al. (2020),
where classical data is encoded multiple times into a quantum circuit with
interleaved trainable rotations.  The key insight is that a single qubit
with L layers of data re-uploading can approximate any function from
R^d -> {0,1,...,C-1}, achieving universal classification.

Each layer:
  1. Encodes input features as rotations: Rx(w_d * x[0]) Ry(w_d * x[1]) Rz(w_d * x[2])
  2. Applies trainable rotations:         Rx(p[0])       Ry(p[1])       Rz(p[2])

Class probabilities are extracted from the Bloch sphere representation
of the final single-qubit state.

References
----------
- Perez-Salinas et al., Quantum 4, 226 (2020)
  "Data re-uploading for a universal quantum classifier"
- Schuld et al., Phys. Rev. A 101, 032308 (2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# Gate primitives (pure numpy, no external deps)
# ------------------------------------------------------------------


def _rx(angle: float) -> np.ndarray:
    """Single-qubit Rx rotation matrix."""
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(angle: float) -> np.ndarray:
    """Single-qubit Ry rotation matrix."""
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(angle: float) -> np.ndarray:
    """Single-qubit Rz rotation matrix."""
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=np.complex128)


def _cnot_state(state: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    """Apply CNOT to a multi-qubit statevector."""
    dim = 1 << n_qubits
    new_state = state.copy()
    c_step = 1 << control
    t_step = 1 << target
    for i in range(dim):
        if (i & c_step) != 0 and (i & t_step) == 0:
            j = i | t_step
            new_state[i], new_state[j] = state[j], state[i]
    return new_state


def _apply_single_qubit(state: np.ndarray, n_qubits: int, qubit: int, mat: np.ndarray) -> np.ndarray:
    """Apply a 2x2 unitary to a single qubit in a multi-qubit state."""
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


def _apply_rotation_triple(state: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Apply Rx(a0) Ry(a1) Rz(a2) to a single-qubit state."""
    state = _rx(angles[0]) @ state
    state = _ry(angles[1]) @ state
    state = _rz(angles[2]) @ state
    return state


def _bloch_coordinates(state: np.ndarray) -> Tuple[float, float, float]:
    """Extract Bloch sphere coordinates (x, y, z) from a single-qubit state.

    Returns
    -------
    tuple of float
        (bx, by, bz) where bx = <X>, by = <Y>, bz = <Z>.
    """
    rho = np.outer(state, np.conj(state))
    bx = float(2.0 * np.real(rho[0, 1]))
    by = float(2.0 * np.imag(rho[1, 0]))
    bz = float(np.real(rho[0, 0] - rho[1, 1]))
    return bx, by, bz


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass
class ReuploadingHistory:
    """Training history for a re-uploading classifier.

    Attributes
    ----------
    losses : list of float
        Loss value at each epoch.
    accuracies : list of float
        Training accuracy at each epoch.
    params : np.ndarray
        Best parameters found during training.
    epochs : int
        Total number of epochs trained.
    """

    losses: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    params: np.ndarray = field(default_factory=lambda: np.array([]))
    epochs: int = 0


# ------------------------------------------------------------------
# Single re-uploading layer
# ------------------------------------------------------------------


@dataclass
class ReuploadingLayer:
    """Single layer: data encoding followed by trainable rotation.

    Each layer applies 6 rotation gates to a single qubit:
    3 data-encoding rotations (Rx, Ry, Rz with scaled input features)
    followed by 3 trainable rotations (Rx, Ry, Rz with free parameters).

    Parameters
    ----------
    n_features : int
        Dimensionality of the input data.
    n_qubits : int
        Number of qubits (default 1 for standard re-uploading).
    """

    n_features: int
    n_qubits: int = 1

    def circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Apply one re-uploading layer to a single-qubit state.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector (will be padded/wrapped to length 3).
        params : np.ndarray
            6 parameters: [w0, w1, w2, p0, p1, p2].
            First 3 are data-encoding weights, last 3 are trainable angles.

        Returns
        -------
        np.ndarray
            2x2 unitary matrix for this layer.
        """
        # Pad or wrap features to length 3
        x_padded = np.zeros(3)
        for i in range(3):
            if i < len(x):
                x_padded[i] = x[i]

        # Data encoding angles: weight * feature
        data_angles = params[:3] * x_padded
        # Trainable angles
        train_angles = params[3:6]

        # Build layer unitary: trainable * data_encoding
        u_data = _rz(data_angles[2]) @ _ry(data_angles[1]) @ _rx(data_angles[0])
        u_train = _rz(train_angles[2]) @ _ry(train_angles[1]) @ _rx(train_angles[0])
        return u_train @ u_data


# ------------------------------------------------------------------
# Single-qubit re-uploading classifier
# ------------------------------------------------------------------


@dataclass
class ReuploadingClassifier:
    """Universal quantum classifier via data re-uploading.

    Key insight: a single qubit with L layers of data re-uploading
    can approximate any function from R^d -> {0,1,...,C-1}.
    Each layer encodes the input data again, allowing the quantum
    state to build up a complex function of the input.

    Class probabilities are obtained from the Bloch sphere coordinates
    of the final state.  For binary classification, the z-coordinate
    maps directly to class probabilities.  For multi-class, the full
    Bloch vector is used with learned class anchors.

    Parameters
    ----------
    n_features : int
        Dimensionality of input data.
    n_classes : int
        Number of output classes.
    n_layers : int
        Number of re-uploading layers.
    n_qubits : int
        Number of qubits (1 for standard single-qubit classifier).
    """

    n_features: int
    n_classes: int
    n_layers: int = 6
    n_qubits: int = 1

    def __post_init__(self):
        # Each layer has 6 params (3 data weights + 3 trainable)
        self.n_params = self.n_layers * 6
        self._layers = [
            ReuploadingLayer(n_features=self.n_features, n_qubits=self.n_qubits)
            for _ in range(self.n_layers)
        ]

    def initialize_params(self, rng=None) -> np.ndarray:
        """Random parameter initialization.

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.  If None, creates one with default seed.

        Returns
        -------
        np.ndarray
            Initial parameter vector of length n_params.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        # Data weights initialized near 1, trainable angles near 0
        params = np.zeros(self.n_params)
        for layer_idx in range(self.n_layers):
            offset = layer_idx * 6
            # Data encoding weights ~ uniform(0.5, 1.5)
            params[offset:offset + 3] = rng.uniform(0.5, 1.5, size=3)
            # Trainable angles ~ uniform(-pi/4, pi/4)
            params[offset + 3:offset + 6] = rng.uniform(-np.pi / 4, np.pi / 4, size=3)
        return params

    def _run_circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Execute the full re-uploading circuit and return the final state.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray
            Full parameter vector.

        Returns
        -------
        np.ndarray
            Final single-qubit statevector (length 2).
        """
        state = np.array([1.0, 0.0], dtype=np.complex128)  # |0>
        x = np.asarray(x, dtype=np.float64).ravel()
        for layer_idx, layer in enumerate(self._layers):
            offset = layer_idx * 6
            layer_params = params[offset:offset + 6]
            u = layer.circuit(x, layer_params)
            state = u @ state
        return state

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single input.

        For binary classification, uses the Bloch z-coordinate:
            P(class 0) = (1 + bz) / 2
            P(class 1) = (1 - bz) / 2

        For multi-class, uses all Bloch coordinates and softmax
        over distances to equally-spaced anchor points.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Class probability vector of length n_classes.
        """
        state = self._run_circuit(x, params)
        bx, by, bz = _bloch_coordinates(state)

        if self.n_classes == 2:
            p0 = np.clip((1.0 + bz) / 2.0, 1e-10, 1.0 - 1e-10)
            return np.array([p0, 1.0 - p0])
        else:
            # Multi-class: anchor points on Bloch sphere
            bloch = np.array([bx, by, bz])
            anchors = self._class_anchors()
            # Negative squared distances as logits
            logits = np.array([
                -np.sum((bloch - a) ** 2) for a in anchors
            ])
            # Softmax
            logits -= logits.max()
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum()
            return np.clip(probs, 1e-10, 1.0 - 1e-10)

    def _class_anchors(self) -> np.ndarray:
        """Generate equally-spaced anchor points on the Bloch sphere.

        Returns
        -------
        np.ndarray
            Array of shape (n_classes, 3) with anchor Bloch vectors.
        """
        anchors = np.zeros((self.n_classes, 3))
        for c in range(self.n_classes):
            angle = 2.0 * np.pi * c / self.n_classes
            anchors[c, 0] = np.cos(angle)  # bx
            anchors[c, 1] = np.sin(angle)  # by
            anchors[c, 2] = 0.0            # bz
        return anchors

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class labels for a batch of inputs.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            probs = self.predict_proba(X[i], params)
            predictions[i] = int(np.argmax(probs))
        return predictions

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Cross-entropy loss over a batch.

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features).
        y : np.ndarray
            True class labels (n_samples,).
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        float
            Average cross-entropy loss.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        total_loss = 0.0
        for i in range(len(X)):
            probs = self.predict_proba(X[i], params)
            p_true = np.clip(probs[y[i]], 1e-10, 1.0 - 1e-10)
            total_loss -= math.log(p_true)
        return total_loss / len(X)

    def gradient(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parameter-shift gradient of the cross-entropy loss.

        Uses the standard parameter-shift rule with shift = pi/2
        applied to each parameter independently.

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features).
        y : np.ndarray
            True class labels (n_samples,).
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector of length n_params.
        """
        params = np.asarray(params, dtype=np.float64)
        grad = np.zeros(len(params))
        shift = np.pi / 2
        denom = 2.0 * np.sin(shift)

        for i in range(len(params)):
            e_i = np.zeros(len(params))
            e_i[i] = shift
            loss_plus = self.loss(X, y, params + e_i)
            loss_minus = self.loss(X, y, params - e_i)
            grad[i] = (loss_plus - loss_minus) / denom
        return grad

    def accuracy(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Classification accuracy over a batch.

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features).
        y : np.ndarray
            True class labels (n_samples,).
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        float
            Accuracy in [0, 1].
        """
        predictions = self.predict(X, params)
        y = np.asarray(y, dtype=int).ravel()
        return float(np.mean(predictions == y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        rng=None,
    ) -> ReuploadingHistory:
        """Train the classifier using Adam optimizer with parameter-shift gradients.

        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features).
        y : np.ndarray
            Training labels (n_samples,).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        rng : np.random.Generator or None
            Random number generator for initialization.

        Returns
        -------
        ReuploadingHistory
            Training history with losses, accuracies, and best params.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        params = self.initialize_params(rng=rng)
        history = ReuploadingHistory(epochs=epochs)

        # Adam state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        best_loss = float("inf")
        best_params = params.copy()

        for epoch in range(1, epochs + 1):
            grad = self.gradient(X, y, params)

            # Adam update
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad ** 2
            m_hat = m / (1.0 - beta1 ** epoch)
            v_hat = v / (1.0 - beta2 ** epoch)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            current_loss = self.loss(X, y, params)
            current_acc = self.accuracy(X, y, params)
            history.losses.append(current_loss)
            history.accuracies.append(current_acc)

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()

        history.params = best_params
        return history


# ------------------------------------------------------------------
# Multi-qubit re-uploading classifier
# ------------------------------------------------------------------


@dataclass
class MultiQubitReuploading:
    """Multi-qubit re-uploading classifier with entangling layers.

    Extends the single-qubit re-uploading approach to multiple qubits
    with CNOT entangling layers between re-uploading blocks.  This
    increases the expressiveness and can handle more complex
    classification boundaries.

    Parameters
    ----------
    n_features : int
        Dimensionality of input data.
    n_classes : int
        Number of output classes.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of re-uploading layers.
    """

    n_features: int
    n_classes: int
    n_qubits: int = 4
    n_layers: int = 4

    def __post_init__(self):
        # Per layer: n_qubits * 6 (data+trainable per qubit)
        self._params_per_layer = self.n_qubits * 6
        self.n_params = self.n_layers * self._params_per_layer

    def initialize_params(self, rng=None) -> np.ndarray:
        """Random parameter initialization for multi-qubit circuit.

        Parameters
        ----------
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        np.ndarray
            Initial parameter vector.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        params = np.zeros(self.n_params)
        for layer_idx in range(self.n_layers):
            for q in range(self.n_qubits):
                offset = layer_idx * self._params_per_layer + q * 6
                params[offset:offset + 3] = rng.uniform(0.5, 1.5, size=3)
                params[offset + 3:offset + 6] = rng.uniform(
                    -np.pi / 4, np.pi / 4, size=3
                )
        return params

    def _entangling_layer(self, state: np.ndarray) -> np.ndarray:
        """Apply CNOT chain for entanglement.

        Connects qubit i to qubit (i+1) for i = 0, ..., n_qubits-2.

        Parameters
        ----------
        state : np.ndarray
            Multi-qubit statevector.

        Returns
        -------
        np.ndarray
            State after entangling layer.
        """
        for q in range(self.n_qubits - 1):
            state = _cnot_state(state, self.n_qubits, q, q + 1)
        return state

    def _run_circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Execute the multi-qubit re-uploading circuit.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray
            Full parameter vector.

        Returns
        -------
        np.ndarray
            Final statevector of length 2^n_qubits.
        """
        dim = 1 << self.n_qubits
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0  # |00...0>

        x = np.asarray(x, dtype=np.float64).ravel()
        x_padded = np.zeros(3)
        for i in range(min(3, len(x))):
            x_padded[i] = x[i]

        for layer_idx in range(self.n_layers):
            # Apply single-qubit re-uploading rotations to each qubit
            for q in range(self.n_qubits):
                offset = layer_idx * self._params_per_layer + q * 6
                layer_params = params[offset:offset + 6]

                # Data encoding
                data_angles = layer_params[:3] * x_padded
                state = _apply_single_qubit(state, self.n_qubits, q, _rx(data_angles[0]))
                state = _apply_single_qubit(state, self.n_qubits, q, _ry(data_angles[1]))
                state = _apply_single_qubit(state, self.n_qubits, q, _rz(data_angles[2]))

                # Trainable rotations
                state = _apply_single_qubit(state, self.n_qubits, q, _rx(layer_params[3]))
                state = _apply_single_qubit(state, self.n_qubits, q, _ry(layer_params[4]))
                state = _apply_single_qubit(state, self.n_qubits, q, _rz(layer_params[5]))

            # Entangling layer
            if self.n_qubits > 1:
                state = self._entangling_layer(state)

        return state

    def predict_proba(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single input.

        Uses measurement probabilities on the first qubit for binary
        classification, or the first n_classes basis states for
        multi-class problems.

        Parameters
        ----------
        x : np.ndarray
            Input feature vector.
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Class probability vector of length n_classes.
        """
        state = self._run_circuit(x, params)
        full_probs = np.abs(state) ** 2

        if self.n_classes == 2:
            # Binary: aggregate over first qubit
            dim = 1 << self.n_qubits
            p0 = 0.0
            for i in range(dim):
                if (i & 1) == 0:
                    p0 += full_probs[i]
            p0 = np.clip(p0, 1e-10, 1.0 - 1e-10)
            return np.array([p0, 1.0 - p0])
        else:
            probs = full_probs[:self.n_classes].copy()
            total = probs.sum()
            if total < 1e-15:
                return np.ones(self.n_classes) / self.n_classes
            probs = probs / total
            return np.clip(probs, 1e-10, 1.0 - 1e-10)

    def predict(self, X: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Predict class labels for a batch of inputs.

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features).
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = np.zeros(len(X), dtype=int)
        for i in range(len(X)):
            probs = self.predict_proba(X[i], params)
            predictions[i] = int(np.argmax(probs))
        return predictions

    def loss(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        """Cross-entropy loss over a batch.

        Parameters
        ----------
        X : np.ndarray
            Input features (n_samples, n_features).
        y : np.ndarray
            True class labels.
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        float
            Average cross-entropy loss.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        total_loss = 0.0
        for i in range(len(X)):
            probs = self.predict_proba(X[i], params)
            p_true = np.clip(probs[y[i]], 1e-10, 1.0 - 1e-10)
            total_loss -= math.log(p_true)
        return total_loss / len(X)

    def gradient(self, X: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parameter-shift gradient of the cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True class labels.
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Gradient vector.
        """
        params = np.asarray(params, dtype=np.float64)
        grad = np.zeros(len(params))
        shift = np.pi / 2
        denom = 2.0 * np.sin(shift)

        for i in range(len(params)):
            e_i = np.zeros(len(params))
            e_i[i] = shift
            loss_plus = self.loss(X, y, params + e_i)
            loss_minus = self.loss(X, y, params - e_i)
            grad[i] = (loss_plus - loss_minus) / denom
        return grad

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        rng=None,
    ) -> ReuploadingHistory:
        """Train the multi-qubit classifier using Adam optimizer.

        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features).
        y : np.ndarray
            Training labels (n_samples,).
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        rng : np.random.Generator or None
            Random number generator.

        Returns
        -------
        ReuploadingHistory
            Training history.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        params = self.initialize_params(rng=rng)
        history = ReuploadingHistory(epochs=epochs)

        # Adam state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        best_loss = float("inf")
        best_params = params.copy()

        for epoch in range(1, epochs + 1):
            grad = self.gradient(X, y, params)

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad ** 2
            m_hat = m / (1.0 - beta1 ** epoch)
            v_hat = v / (1.0 - beta2 ** epoch)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            current_loss = self.loss(X, y, params)
            current_acc = float(np.mean(self.predict(X, params) == y))
            history.losses.append(current_loss)
            history.accuracies.append(current_acc)

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = params.copy()

        history.params = best_params
        return history
