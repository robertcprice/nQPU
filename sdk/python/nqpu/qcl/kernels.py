"""Quantum kernel methods for quantum machine learning.

Provides quantum kernel computation, quantum SVM, and kernel PCA.  Kernel
methods avoid the barren plateau problem entirely by working with
fidelity-based similarity measures instead of variational optimization.

Classes
-------
- :class:`QuantumKernel` -- standard fidelity kernel |<phi(x)|phi(y)>|^2
- :class:`ProjectedQuantumKernel` -- classical post-processing of quantum features
- :class:`TrainableKernel` -- kernel with optimizable circuit parameters
- :class:`QSVM` -- quantum support vector machine (simplified dual solver)
- :class:`QKernelPCA` -- kernel PCA for dimensionality reduction

References
----------
- Havlicek et al., Nature 567, 209 (2019) [quantum kernel estimation]
- Schuld & Killoran, Phys. Rev. Lett. 122, 040504 (2019) [quantum kernels]
- Huang et al., Nat. Commun. 12, 2631 (2021) [projected quantum kernels]
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .circuits import (
    AnsatzCircuit,
    CircuitTemplate,
    DataEncodingCircuit,
    HardwareEfficientAnsatz,
    ParameterizedCircuit,
    StatevectorSimulator,
)


# ------------------------------------------------------------------
# Kernel target alignment metric
# ------------------------------------------------------------------


def kernel_target_alignment(
    K: np.ndarray, y: np.ndarray
) -> float:
    """Compute kernel-target alignment (KTA).

    KTA measures how well a kernel matrix aligns with the ideal kernel
    defined by the labels.  Values near 1 indicate strong alignment.

    Parameters
    ----------
    K : np.ndarray
        Kernel (Gram) matrix of shape (n, n).
    y : np.ndarray
        Labels (will be converted to +1/-1 for binary).

    Returns
    -------
    float
        KTA value in [-1, 1].
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    # Convert to +/-1 if binary labels
    unique = np.unique(y)
    if len(unique) == 2:
        y = np.where(y == unique[0], -1.0, 1.0)

    Y = np.outer(y, y)
    num = np.sum(K * Y)
    den = np.sqrt(np.sum(K ** 2) * np.sum(Y ** 2))
    if den < 1e-15:
        return 0.0
    return float(num / den)


# ------------------------------------------------------------------
# Quantum kernel
# ------------------------------------------------------------------


class QuantumKernel:
    """Quantum kernel: K(x, y) = |<phi(x)|phi(y)>|^2.

    Computes the fidelity between quantum feature states as a kernel
    function.  This is the standard quantum kernel used in QSVM and
    kernel methods.

    Parameters
    ----------
    encoding : DataEncodingCircuit
        Circuit that encodes classical data into quantum states.
    """

    def __init__(self, encoding: DataEncodingCircuit) -> None:
        self.encoding = encoding

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the kernel value K(x1, x2).

        Parameters
        ----------
        x1, x2 : np.ndarray
            Input feature vectors.

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        sv1 = self.encoding.encode(x1)
        sv2 = self.encoding.encode(x2)
        return float(abs(np.vdot(sv1, sv2)) ** 2)

    def matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the full kernel matrix for a dataset.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Symmetric PSD kernel matrix of shape (n, n).
        """
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        K = np.zeros((n, n), dtype=np.float64)
        # Cache statevectors for efficiency
        states = [self.encoding.encode(X[i]) for i in range(n)]
        for i in range(n):
            K[i, i] = 1.0  # <phi|phi> = 1
            for j in range(i + 1, n):
                val = float(abs(np.vdot(states[i], states[j])) ** 2)
                K[i, j] = val
                K[j, i] = val
        return K

    def cross_matrix(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute kernel values between two different datasets.

        Parameters
        ----------
        X1 : np.ndarray
            First data matrix (n1, d).
        X2 : np.ndarray
            Second data matrix (n2, d).

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (n1, n2).
        """
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2), dtype=np.float64)
        states1 = [self.encoding.encode(X1[i]) for i in range(n1)]
        states2 = [self.encoding.encode(X2[j]) for j in range(n2)]
        for i in range(n1):
            for j in range(n2):
                K[i, j] = float(abs(np.vdot(states1[i], states2[j])) ** 2)
        return K


# ------------------------------------------------------------------
# Projected quantum kernel
# ------------------------------------------------------------------


class ProjectedQuantumKernel:
    """Projected quantum kernel with classical post-processing.

    Instead of computing fidelity directly, this kernel first extracts
    classical feature vectors from the quantum state (via expectation
    values of local observables), then uses a classical kernel on those
    features.  This can improve trainability and generalization.

    Parameters
    ----------
    encoding : DataEncodingCircuit
        Circuit for quantum feature mapping.
    n_projections : int
        Number of expectation-value features to extract.
        Defaults to n_qubits (one <Z> per qubit).
    gamma : float
        RBF kernel bandwidth parameter.
    """

    def __init__(
        self,
        encoding: DataEncodingCircuit,
        n_projections: int | None = None,
        gamma: float = 1.0,
    ) -> None:
        self.encoding = encoding
        self.n_projections = n_projections or encoding.n_qubits
        self.gamma = gamma

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Extract classical features via qubit-wise Z expectations."""
        state = self.encoding.encode(x)
        n = self.encoding.n_qubits
        features = np.zeros(min(self.n_projections, n), dtype=np.float64)
        dim = 1 << n
        for q in range(len(features)):
            exp_val = 0.0
            step = 1 << q
            for i in range(dim):
                p = abs(state[i]) ** 2
                if i & step:
                    exp_val -= p
                else:
                    exp_val += p
            features[q] = exp_val
        return features

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the projected kernel value."""
        f1 = self._project(x1)
        f2 = self._project(x2)
        diff = f1 - f2
        return float(np.exp(-self.gamma * np.dot(diff, diff)))

    def matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the projected kernel matrix."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        features = np.array([self._project(X[i]) for i in range(n)])
        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                diff = features[i] - features[j]
                val = float(np.exp(-self.gamma * np.dot(diff, diff)))
                K[i, j] = val
                K[j, i] = val
        return K


# ------------------------------------------------------------------
# Trainable kernel
# ------------------------------------------------------------------


class TrainableKernel:
    """Quantum kernel with optimizable circuit parameters.

    Wraps a :class:`CircuitTemplate` and uses both encoding and ansatz
    to compute kernel values.  The ansatz parameters are optimized to
    maximize kernel-target alignment.

    Parameters
    ----------
    circuit : CircuitTemplate
        Combined encoding + ansatz circuit.
    """

    def __init__(self, circuit: CircuitTemplate) -> None:
        self.circuit = circuit
        self.params: np.ndarray | None = None

    def set_params(self, params: np.ndarray) -> None:
        """Set the trainable kernel parameters."""
        self.params = np.asarray(params, dtype=np.float64).copy()

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute the trainable kernel value."""
        if self.params is None:
            raise RuntimeError("Parameters not set. Call set_params() first.")
        return self.circuit.overlap(x1, x2, self.params)

    def matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix with current parameters."""
        if self.params is None:
            raise RuntimeError("Parameters not set. Call set_params() first.")
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        K = np.zeros((n, n), dtype=np.float64)
        states = [self.circuit.run(X[i], self.params) for i in range(n)]
        for i in range(n):
            K[i, i] = 1.0
            for j in range(i + 1, n):
                val = float(abs(np.vdot(states[i], states[j])) ** 2)
                K[i, j] = val
                K[j, i] = val
        return K


# ------------------------------------------------------------------
# QSVM (simplified dual solver)
# ------------------------------------------------------------------


class QSVM:
    """Quantum Support Vector Machine.

    Uses a quantum kernel to compute the Gram matrix, then solves the
    SVM dual problem via a simplified sequential minimal optimization
    (SMO) algorithm.  Supports binary classification only.

    Parameters
    ----------
    kernel : QuantumKernel or ProjectedQuantumKernel or TrainableKernel
        Quantum kernel to use.
    C : float
        Regularization parameter (upper bound on Lagrange multipliers).
    max_iter : int
        Maximum number of SMO iterations.
    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        kernel: QuantumKernel | ProjectedQuantumKernel | TrainableKernel,
        C: float = 1.0,
        max_iter: int = 200,
        tol: float = 1e-3,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.alphas: np.ndarray | None = None
        self.bias: float = 0.0
        self.support_labels: np.ndarray | None = None
        self.K_train: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVM":
        """Fit the QSVM on training data.

        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features).
        y : np.ndarray
            Binary labels (two unique values).

        Returns
        -------
        QSVM
            Self (for chaining).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        unique = np.unique(y)
        if len(unique) != 2:
            raise ValueError(f"QSVM supports binary classification, got {len(unique)} classes")

        # Map to +1/-1
        self._label_map = {unique[0]: -1.0, unique[1]: 1.0}
        self._inv_label_map = {-1.0: unique[0], 1.0: unique[1]}
        y_svm = np.array([self._label_map[yi] for yi in y])

        self.X_train = X.copy()
        self.y_train = y_svm.copy()
        n = len(X)

        # Compute kernel matrix
        if isinstance(self.kernel, QuantumKernel):
            K = self.kernel.matrix(X)
        elif isinstance(self.kernel, ProjectedQuantumKernel):
            K = self.kernel.matrix(X)
        elif isinstance(self.kernel, TrainableKernel):
            K = self.kernel.matrix(X)
        else:
            K = self.kernel.matrix(X)
        self.K_train = K

        # SMO-style dual solver
        alphas = np.zeros(n, dtype=np.float64)
        bias = 0.0

        for iteration in range(self.max_iter):
            changed = 0
            for i in range(n):
                # Decision function value
                f_i = np.sum(alphas * y_svm * K[i]) + bias
                e_i = f_i - y_svm[i]

                # Check KKT violations
                if (y_svm[i] * e_i < -self.tol and alphas[i] < self.C) or (
                    y_svm[i] * e_i > self.tol and alphas[i] > 0
                ):
                    # Pick j != i randomly
                    j = i
                    while j == i:
                        j = int(np.random.randint(0, n))

                    f_j = np.sum(alphas * y_svm * K[j]) + bias
                    e_j = f_j - y_svm[j]

                    # Save old alphas
                    ai_old, aj_old = alphas[i], alphas[j]

                    # Compute bounds
                    if y_svm[i] != y_svm[j]:
                        L = max(0.0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0.0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])

                    if L >= H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    alphas[j] -= y_svm[j] * (e_i - e_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - aj_old) < 1e-8:
                        continue

                    # Update alpha_i
                    alphas[i] += y_svm[i] * y_svm[j] * (aj_old - alphas[j])

                    # Update bias
                    b1 = (
                        bias
                        - e_i
                        - y_svm[i] * (alphas[i] - ai_old) * K[i, i]
                        - y_svm[j] * (alphas[j] - aj_old) * K[i, j]
                    )
                    b2 = (
                        bias
                        - e_j
                        - y_svm[i] * (alphas[i] - ai_old) * K[i, j]
                        - y_svm[j] * (alphas[j] - aj_old) * K[j, j]
                    )

                    if 0 < alphas[i] < self.C:
                        bias = b1
                    elif 0 < alphas[j] < self.C:
                        bias = b2
                    else:
                        bias = (b1 + b2) / 2.0

                    changed += 1

            if changed == 0:
                break

        self.alphas = alphas
        self.bias = bias
        self.support_labels = y_svm
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new data.

        Parameters
        ----------
        X : np.ndarray
            Test features (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels (original label values, not +/-1).
        """
        if self.alphas is None or self.X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if isinstance(self.kernel, (QuantumKernel, ProjectedQuantumKernel, TrainableKernel)):
            K_test = self.kernel.cross_matrix(X, self.X_train) if hasattr(self.kernel, 'cross_matrix') else self._compute_cross_kernel(X)
        else:
            K_test = self._compute_cross_kernel(X)

        decision = K_test @ (self.alphas * self.y_train) + self.bias
        raw_preds = np.sign(decision)
        # Map back to original labels
        return np.array(
            [self._inv_label_map.get(p, self._inv_label_map[1.0]) for p in raw_preds]
        )

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values (signed distance to hyperplane)."""
        if self.alphas is None or self.X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        if hasattr(self.kernel, 'cross_matrix'):
            K_test = self.kernel.cross_matrix(X, self.X_train)
        else:
            K_test = self._compute_cross_kernel(X)
        return K_test @ (self.alphas * self.y_train) + self.bias

    def _compute_cross_kernel(self, X_test: np.ndarray) -> np.ndarray:
        """Fallback cross-kernel computation."""
        n_test = len(X_test)
        n_train = len(self.X_train)
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in range(n_test):
            for j in range(n_train):
                K[i, j] = self.kernel.evaluate(X_test[i], self.X_train[j])
        return K

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Classification accuracy on test data."""
        preds = self.predict(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        return float(np.mean(preds == y))


# ------------------------------------------------------------------
# Quantum kernel PCA
# ------------------------------------------------------------------


class QKernelPCA:
    """Quantum Kernel Principal Component Analysis.

    Performs PCA in the quantum feature space by eigendecomposing the
    centered kernel matrix.  Useful for dimensionality reduction and
    visualization of quantum feature maps.

    Parameters
    ----------
    kernel : QuantumKernel or ProjectedQuantumKernel
        Quantum kernel for computing the Gram matrix.
    n_components : int
        Number of principal components to keep.
    """

    def __init__(
        self,
        kernel: QuantumKernel | ProjectedQuantumKernel,
        n_components: int = 2,
    ) -> None:
        self.kernel = kernel
        self.n_components = n_components
        self.eigenvalues: np.ndarray | None = None
        self.eigenvectors: np.ndarray | None = None
        self.K_train: np.ndarray | None = None
        self.X_train: np.ndarray | None = None
        self._row_mean: np.ndarray | None = None
        self._total_mean: float = 0.0

    def fit(self, X: np.ndarray) -> "QKernelPCA":
        """Fit kernel PCA on training data.

        Parameters
        ----------
        X : np.ndarray
            Training data (n_samples, n_features).

        Returns
        -------
        QKernelPCA
            Self.
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_train = X.copy()
        n = len(X)

        K = self.kernel.matrix(X)
        self.K_train = K.copy()

        # Center the kernel matrix
        self._row_mean = K.mean(axis=0)
        self._total_mean = K.mean()
        K_centered = (
            K
            - self._row_mean[np.newaxis, :]
            - self._row_mean[:, np.newaxis]
            + self._total_mean
        )

        # Eigendecompose
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues = eigenvalues[: self.n_components]
        self.eigenvectors = eigenvectors[:, : self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project data onto the principal components.

        Parameters
        ----------
        X : np.ndarray
            Data to transform (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Projected data (n_samples, n_components).
        """
        if self.eigenvectors is None or self.X_train is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)

        # Compute cross-kernel
        if hasattr(self.kernel, 'cross_matrix'):
            K_test = self.kernel.cross_matrix(X, self.X_train)
        else:
            n_test = len(X)
            n_train = len(self.X_train)
            K_test = np.zeros((n_test, n_train), dtype=np.float64)
            for i in range(n_test):
                for j in range(n_train):
                    K_test[i, j] = self.kernel.evaluate(X[i], self.X_train[j])

        # Center the test kernel
        K_test_centered = (
            K_test
            - K_test.mean(axis=1, keepdims=True)
            - self._row_mean[np.newaxis, :]
            + self._total_mean
        )

        # Project
        result = np.zeros((len(X), self.n_components), dtype=np.float64)
        for c in range(self.n_components):
            if self.eigenvalues[c] > 1e-10:
                result[:, c] = (
                    K_test_centered @ self.eigenvectors[:, c]
                    / np.sqrt(self.eigenvalues[c])
                )
        return result

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Fraction of variance explained by each component."""
        if self.eigenvalues is None:
            raise RuntimeError("Model not fitted.")
        total = max(self.eigenvalues.sum(), 1e-15)
        return np.clip(self.eigenvalues / total, 0.0, 1.0)
