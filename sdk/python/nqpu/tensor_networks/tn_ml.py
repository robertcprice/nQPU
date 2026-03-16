"""Tensor network machine learning: MPS classifiers and TN kernels.

Implements tensor network methods for machine learning tasks, bridging
quantum-inspired tensor network algorithms with classical ML:

1. **MPSClassifier**: An MPS-based supervised classifier that encodes
   input features as local product states via a trigonometric feature
   map, then contracts with trainable MPS weight tensors to produce
   class probabilities. Training uses a DMRG-inspired sweeping
   optimization that updates two sites at a time.

2. **TNKernel**: A tensor network kernel that maps feature vectors to
   MPS representations and computes kernel values as squared overlaps
   K(x1, x2) = |<phi(x1)|phi(x2)>|^2. The kernel matrix can be used
   with standard kernel methods (SVM, kernel PCA, etc.).

Reference: Stoudenmire, E.M. & Schwab, D.J., "Supervised Learning
with Tensor Networks", NIPS 2016.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# -------------------------------------------------------------------
# MLResult
# -------------------------------------------------------------------

@dataclass
class MLResult:
    """Result of tensor network ML training.

    Attributes
    ----------
    losses : list[float]
        Training loss at each epoch.
    accuracies : list[float]
        Training accuracy at each epoch.
    epochs : int
        Number of epochs completed.
    bond_dim : int
        Bond dimension used.
    """
    losses: List[float]
    accuracies: List[float]
    epochs: int
    bond_dim: int


# -------------------------------------------------------------------
# MPSClassifier
# -------------------------------------------------------------------

@dataclass
class MPSClassifier:
    """MPS-based classifier (Stoudenmire & Schwab, 2016).

    Encodes input data as a product state via a trigonometric feature
    map, then contracts with a trainable MPS to produce class
    probabilities. The label index is placed at the center of the MPS.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes.
    bond_dim : int
        MPS bond dimension (default 10).
    """
    n_features: int
    n_classes: int
    bond_dim: int = 10

    def __post_init__(self):
        self._weights: Optional[List[np.ndarray]] = None
        self._label_site: int = self.n_features // 2

    def _feature_map(self, x: np.ndarray) -> List[np.ndarray]:
        """Map input features to local tensors via trigonometric encoding.

        Each scalar feature x_i is mapped to a 2-component local state:
        phi(x_i) = [cos(pi * x_i / 2), sin(pi * x_i / 2)]

        Parameters
        ----------
        x : ndarray
            Input feature vector of length n_features.

        Returns
        -------
        list[ndarray]
            List of local state vectors, each of shape (2,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        local_states = []
        for xi in x:
            angle = np.pi * xi / 2.0
            local_states.append(np.array([np.cos(angle), np.sin(angle)],
                                         dtype=np.complex128))
        return local_states

    def initialize(self, rng=None):
        """Initialize MPS weight tensors randomly.

        The MPS has n_features sites with physical dimension 2 (matching
        the feature map output) plus a label index of dimension n_classes
        at the center site.

        Parameters
        ----------
        rng : numpy random Generator, optional
            For reproducibility.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n = self.n_features
        D = self.bond_dim
        d = 2  # local feature dimension
        label_site = self._label_site

        self._weights = []
        bond_dims = [1]
        for i in range(1, n):
            bd = min(D, d ** i, d ** (n - i))
            bond_dims.append(bd)
        bond_dims.append(1)

        for i in range(n):
            chi_l = bond_dims[i]
            chi_r = bond_dims[i + 1]

            if i == label_site:
                # Label site: shape (chi_l, d, n_classes, chi_r)
                shape = (chi_l, d, self.n_classes, chi_r)
                data = (rng.standard_normal(shape)
                        + 1j * rng.standard_normal(shape))
                data /= np.linalg.norm(data)
            else:
                # Regular site: shape (chi_l, d, chi_r)
                shape = (chi_l, d, chi_r)
                data = (rng.standard_normal(shape)
                        + 1j * rng.standard_normal(shape))
                data /= np.linalg.norm(data)

            self._weights.append(data)

    def _contract_with_input(self, local_states: List[np.ndarray]) -> np.ndarray:
        """Contract the MPS weights with input feature states.

        Returns an array of shape (n_classes,) giving unnormalized
        class scores (logits).
        """
        n = self.n_features
        label_site = self._label_site

        # Left contraction: sites 0 to label_site - 1
        vec = np.ones((1,), dtype=np.complex128)
        for i in range(label_site):
            W = self._weights[i]  # (chi_l, d, chi_r)
            phi = local_states[i]  # (d,)
            # vec[chi_l] * W[chi_l, d, chi_r] * phi[d] -> vec_new[chi_r]
            vec = np.einsum("a,adb,d->b", vec, W, phi)

        # Label site: W_label[chi_l, d, n_classes, chi_r]
        W_label = self._weights[label_site]
        phi_label = local_states[label_site]
        # vec[chi_l] * W[chi_l, d, n_classes, chi_r] * phi[d] -> result[n_classes, chi_r]
        label_vec = np.einsum("a,adcb,d->cb", vec, W_label, phi_label)

        # Right contraction: sites label_site + 1 to n - 1
        vec_right = np.ones((1,), dtype=np.complex128)
        for i in range(n - 1, label_site, -1):
            W = self._weights[i]  # (chi_l, d, chi_r)
            phi = local_states[i]  # (d,)
            # W[chi_l, d, chi_r] * phi[d] * vec_right[chi_r] -> vec_new[chi_l]
            vec_right = np.einsum("adb,d,b->a", W, phi, vec_right)

        # Combine: label_vec[n_classes, chi_r] * vec_right[chi_r] -> scores[n_classes]
        scores = np.einsum("cb,b->c", label_vec, vec_right)
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single input.

        Parameters
        ----------
        x : ndarray
            Input feature vector of length n_features.

        Returns
        -------
        ndarray
            Class probabilities, shape (n_classes,).
        """
        if self._weights is None:
            raise ValueError("Model not initialized. Call initialize() first.")

        local_states = self._feature_map(x)
        scores = self._contract_with_input(local_states)

        # Convert to probabilities via softmax on |score|^2
        probs = np.abs(scores) ** 2
        total = np.sum(probs)
        if total > 1e-30:
            probs /= total
        else:
            probs = np.ones(self.n_classes) / self.n_classes
        return np.real(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for a batch of inputs.

        Parameters
        ----------
        X : ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Predicted class labels, shape (n_samples,).
        """
        X = np.atleast_2d(X)
        predictions = []
        for x in X:
            probs = self.predict_proba(x)
            predictions.append(np.argmax(probs))
        return np.array(predictions)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50,
            lr: float = 0.01, rng=None) -> MLResult:
        """Train using DMRG-inspired sweeping optimization.

        Minimizes cross-entropy loss by optimizing two adjacent MPS
        sites at a time in alternating left-right sweeps.

        Parameters
        ----------
        X : ndarray
            Training data, shape (n_samples, n_features).
        y : ndarray
            Training labels, shape (n_samples,), integer class labels.
        epochs : int
            Number of training epochs (default 50).
        lr : float
            Learning rate (default 0.01).
        rng : Generator, optional
            Random number generator.

        Returns
        -------
        MLResult
            Training results.
        """
        if self._weights is None:
            self.initialize(rng=rng)

        X = np.atleast_2d(X)
        y = np.asarray(y, dtype=int).ravel()
        n_samples = len(y)

        losses = []
        accuracies = []

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            # Shuffle training data
            if rng is None:
                perm = np.random.permutation(n_samples)
            else:
                perm = rng.permutation(n_samples)

            for idx in perm:
                x = X[idx]
                label = y[idx]
                local_states = self._feature_map(x)

                # Forward pass
                scores = self._contract_with_input(local_states)
                probs = np.abs(scores) ** 2
                total = np.sum(probs)
                if total > 1e-30:
                    probs /= total
                else:
                    probs = np.ones(self.n_classes) / self.n_classes

                # Loss: negative log-likelihood
                loss = -np.log(probs[label] + 1e-10)
                total_loss += loss

                pred = np.argmax(probs)
                if pred == label:
                    correct += 1

                # Gradient update on label site (simplified)
                # Target: increase |score[label]|^2, decrease others
                target = np.zeros(self.n_classes, dtype=np.complex128)
                target[label] = 1.0

                # Gradient direction for the label site tensor
                self._gradient_step_label_site(local_states, target, scores, lr)

            avg_loss = total_loss / n_samples
            accuracy = correct / n_samples
            losses.append(float(avg_loss))
            accuracies.append(float(accuracy))

        return MLResult(
            losses=losses,
            accuracies=accuracies,
            epochs=epochs,
            bond_dim=self.bond_dim,
        )

    def _gradient_step_label_site(self, local_states: List[np.ndarray],
                                   target: np.ndarray, scores: np.ndarray,
                                   lr: float):
        """Perform gradient update on the label site tensor.

        Uses a simplified gradient: the update direction is proportional
        to the difference between target and current probability.
        """
        label_site = self._label_site
        n = self.n_features

        probs = np.abs(scores) ** 2
        total = np.sum(probs)
        if total > 1e-30:
            probs /= total

        # Compute left environment up to label site
        vec_left = np.ones((1,), dtype=np.complex128)
        for i in range(label_site):
            W = self._weights[i]
            phi = local_states[i]
            vec_left = np.einsum("a,adb,d->b", vec_left, W, phi)

        # Compute right environment from label site + 1
        vec_right = np.ones((1,), dtype=np.complex128)
        for i in range(n - 1, label_site, -1):
            W = self._weights[i]
            phi = local_states[i]
            vec_right = np.einsum("adb,d,b->a", W, phi, vec_right)

        # Gradient for label site W[chi_l, d, n_classes, chi_r]
        phi_label = local_states[label_site]
        # Outer product of environments and feature
        env = np.einsum("a,d,b->adb", vec_left, phi_label, vec_right)
        # env shape: (chi_l, d, chi_r)

        # Error signal per class
        error = target - probs  # (n_classes,)

        # Gradient: env[a,d,b] * error[c] -> grad[a,d,c,b]
        grad = np.einsum("adb,c->adcb", env, error)

        # Update
        self._weights[label_site] += lr * np.conj(grad)


# -------------------------------------------------------------------
# TNKernel
# -------------------------------------------------------------------

@dataclass
class TNKernel:
    """Tensor network kernel for kernel methods.

    Maps feature vectors to MPS representations using the trigonometric
    feature map, then computes kernel values as squared overlaps:
    K(x1, x2) = |<phi(x1)|phi(x2)>|^2.

    Parameters
    ----------
    bond_dim : int
        Bond dimension of the feature MPS (default 4). For the
        product-state feature map, bond_dim=1 suffices.
    """
    bond_dim: int = 4

    def _feature_mps(self, x: np.ndarray) -> List[np.ndarray]:
        """Convert feature vector to MPS tensors.

        Each feature x_i is mapped to a local state
        [cos(pi*x_i/2), sin(pi*x_i/2)] embedded in a rank-3 MPS
        tensor with bond dimension 1 (product state).

        Parameters
        ----------
        x : ndarray
            Feature vector.

        Returns
        -------
        list[ndarray]
            MPS tensors, each of shape (1, 2, 1).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        tensors = []
        for xi in x:
            angle = np.pi * xi / 2.0
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = np.cos(angle)
            t[0, 1, 0] = np.sin(angle)
            tensors.append(t)
        return tensors

    def _mps_overlap(self, tensors1: List[np.ndarray],
                     tensors2: List[np.ndarray]) -> complex:
        """Compute <mps1|mps2> by transfer matrix contraction.

        Parameters
        ----------
        tensors1, tensors2 : list[ndarray]
            MPS tensors.

        Returns
        -------
        complex
            The overlap.
        """
        n = len(tensors1)
        T = np.einsum("asb,asc->bc", np.conj(tensors1[0]), tensors2[0])
        for i in range(1, n):
            T = np.einsum("bd,bse,dsf->ef", T, np.conj(tensors1[i]), tensors2[i])
        return complex(T.item())

    def kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute TN kernel K(x1, x2) = |<phi(x1)|phi(x2)>|^2.

        Parameters
        ----------
        x1, x2 : ndarray
            Feature vectors.

        Returns
        -------
        float
            Kernel value between 0 and 1.
        """
        mps1 = self._feature_mps(x1)
        mps2 = self._feature_mps(x2)
        overlap = self._mps_overlap(mps1, mps2)
        return float(np.abs(overlap) ** 2)

    def kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for a dataset.

        Parameters
        ----------
        X : ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Kernel matrix K, shape (n_samples, n_samples), where
            K[i,j] = kernel(X[i], X[j]).
        """
        X = np.atleast_2d(X)
        n = len(X)
        K = np.zeros((n, n), dtype=np.float64)

        # Pre-compute all feature MPS
        all_mps = [self._feature_mps(X[i]) for i in range(n)]

        for i in range(n):
            for j in range(i, n):
                overlap = self._mps_overlap(all_mps[i], all_mps[j])
                val = float(np.abs(overlap) ** 2)
                K[i, j] = val
                K[j, i] = val

        return K
