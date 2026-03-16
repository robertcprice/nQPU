"""Quantum-inspired linear algebra algorithms.

Classical algorithms that achieve sub-linear or improved scaling by
borrowing techniques from quantum machine learning:

1. **QISVD** -- Randomised SVD with length-squared sampling (inspired by
   quantum singular value estimation).
2. **QIRegression** -- Low-rank regression via dequantized row sampling.
3. **QIPCA** -- Principal component analysis via Nystrom approximation
   with quantum-inspired sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class QISVDResult:
    """Result of quantum-inspired SVD."""

    U: np.ndarray  # (m, k)
    S: np.ndarray  # (k,)
    Vt: np.ndarray  # (k, n)
    explained_variance: np.ndarray  # (k,)
    n_samples_used: int
    relative_error: float


@dataclass
class QIRegressionResult:
    """Result of quantum-inspired regression."""

    coefficients: np.ndarray  # (n,)
    residual: float
    r_squared: float
    n_samples_used: int


@dataclass
class QIPCAResult:
    """Result of quantum-inspired PCA."""

    components: np.ndarray  # (k, n)
    explained_variance_ratio: np.ndarray  # (k,)
    n_samples: int
    total_variance: float


# ---------------------------------------------------------------------------
# Quantum-Inspired SVD
# ---------------------------------------------------------------------------

class QISVD:
    """Randomised SVD using length-squared (quantum-inspired) sampling.

    For an m x n matrix, samples O(k * polylog(mn) / eps^2) rows and
    columns to form a sketch, then computes the rank-k SVD of the sketch.
    """

    def __init__(
        self,
        oversampling: int = 10,
        seed: int = 42,
    ) -> None:
        self.oversampling = oversampling
        self.seed = seed

    def fit(self, A: np.ndarray, k: int = 5) -> QISVDResult:
        """Compute a rank-*k* approximate SVD of *A*.

        Parameters
        ----------
        A : ndarray of shape (m, n)
        k : target rank

        Returns
        -------
        QISVDResult
        """
        rng = np.random.default_rng(self.seed)
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape
        actual_k = min(k, m, n)
        n_samples = min(actual_k + self.oversampling, m, n)

        # Step 1: length-squared row sampling
        row_norms_sq = np.sum(A ** 2, axis=1)
        total_row = row_norms_sq.sum()
        if total_row < 1e-15:
            row_probs = np.ones(m) / m
        else:
            row_probs = row_norms_sq / total_row

        row_indices = rng.choice(m, size=n_samples, replace=True, p=row_probs)

        # Step 2: length-squared column sampling
        col_norms_sq = np.sum(A ** 2, axis=0)
        total_col = col_norms_sq.sum()
        if total_col < 1e-15:
            col_probs = np.ones(n) / n
        else:
            col_probs = col_norms_sq / total_col

        col_indices = rng.choice(n, size=n_samples, replace=True, p=col_probs)

        # Step 3: form rescaled submatrix C = sampled rows, R = sampled cols
        # Use the Frieze-Kannan-Vempala approach:
        #   C (m x c):  columns of A sampled and rescaled
        #   R (r x n):  rows of A sampled and rescaled
        C = A[:, col_indices] / np.sqrt(n_samples * col_probs[col_indices])
        R = A[row_indices, :] / np.sqrt(
            n_samples * row_probs[row_indices]
        )[:, np.newaxis]

        # Step 4: intersection matrix W = C[row_indices, :]
        W = C[row_indices, :]

        # Step 5: pseudo-inverse of W for CUR decomposition
        U_w, S_w, Vt_w = np.linalg.svd(W, full_matrices=False)
        # Truncate tiny singular values
        threshold = 1e-10 * (S_w[0] if len(S_w) > 0 else 1.0)
        keep = S_w > threshold
        S_w_inv = np.zeros_like(S_w)
        S_w_inv[keep] = 1.0 / S_w[keep]
        W_pinv = (Vt_w.T * S_w_inv) @ U_w.T

        # Step 6: form the core and get final SVD
        # A ~ C @ W^+ @ R
        core = C @ W_pinv @ R
        U_full, S_full, Vt_full = np.linalg.svd(core, full_matrices=False)

        # Truncate to rank k
        U_k = U_full[:, :actual_k]
        S_k = S_full[:actual_k]
        Vt_k = Vt_full[:actual_k, :]

        # Explained variance
        total_var = np.sum(A ** 2)
        explained = S_k ** 2
        if total_var > 1e-15:
            explained_norm = explained / total_var
        else:
            explained_norm = explained

        # Relative error (Frobenius)
        recon = U_k * S_k @ Vt_k
        rel_error = float(
            np.linalg.norm(A - recon) / max(np.linalg.norm(A), 1e-15)
        )

        return QISVDResult(
            U=U_k,
            S=S_k,
            Vt=Vt_k,
            explained_variance=explained_norm,
            n_samples_used=n_samples,
            relative_error=rel_error,
        )


# ---------------------------------------------------------------------------
# Quantum-Inspired Regression
# ---------------------------------------------------------------------------

class QIRegression:
    """Quantum-inspired least-squares regression.

    Samples rows of the augmented matrix [A | b] proportional to their
    squared norms, solves the smaller system, and extrapolates to the
    full solution.  For tall-and-skinny systems this can be much faster
    than direct solves.
    """

    def __init__(
        self,
        oversampling_factor: int = 4,
        seed: int = 42,
    ) -> None:
        self.oversampling_factor = oversampling_factor
        self.seed = seed

    def fit(self, A: np.ndarray, b: np.ndarray) -> QIRegressionResult:
        """Solve min ||Ax - b||_2 using quantum-inspired sampling.

        Parameters
        ----------
        A : ndarray of shape (m, n)
        b : ndarray of shape (m,)

        Returns
        -------
        QIRegressionResult
        """
        rng = np.random.default_rng(self.seed)
        A = np.asarray(A, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64).ravel()
        m, n = A.shape

        # Augmented matrix [A | b]
        Ab = np.column_stack([A, b])

        # Row sampling proportional to squared norms
        row_norms_sq = np.sum(Ab ** 2, axis=1)
        total = row_norms_sq.sum()
        if total < 1e-15:
            row_probs = np.ones(m) / m
        else:
            row_probs = row_norms_sq / total

        n_samples = min(self.oversampling_factor * n, m)
        row_indices = rng.choice(m, size=n_samples, replace=True, p=row_probs)

        # Rescale sampled rows
        scale = 1.0 / np.sqrt(n_samples * row_probs[row_indices])
        A_sampled = A[row_indices] * scale[:, np.newaxis]
        b_sampled = b[row_indices] * scale

        # Solve the smaller system
        x, residuals, rank, sv = np.linalg.lstsq(A_sampled, b_sampled, rcond=None)

        # Compute full residual and R^2
        residual_vec = A @ x - b
        ss_res = float(np.dot(residual_vec, residual_vec))
        ss_tot = float(np.sum((b - b.mean()) ** 2))
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-15) if ss_tot > 1e-15 else 0.0

        return QIRegressionResult(
            coefficients=x,
            residual=np.sqrt(ss_res),
            r_squared=float(np.clip(r_squared, 0.0, 1.0)),
            n_samples_used=n_samples,
        )


# ---------------------------------------------------------------------------
# Quantum-Inspired PCA
# ---------------------------------------------------------------------------

class QIPCA:
    """Quantum-inspired Principal Component Analysis.

    Uses the Nystrom approximation with length-squared column sampling
    to find the top-k principal components without computing the full
    covariance matrix.
    """

    def __init__(
        self,
        oversampling: int = 10,
        seed: int = 42,
    ) -> None:
        self.oversampling = oversampling
        self.seed = seed

    def fit(self, X: np.ndarray, k: int = 5) -> QIPCAResult:
        """Find top-*k* principal components of *X*.

        Parameters
        ----------
        X : ndarray of shape (m, n)
            Data matrix (m samples, n features).  Centered internally.
        k : target number of principal components.

        Returns
        -------
        QIPCAResult
        """
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        m, n = X.shape

        # Center the data
        mean = X.mean(axis=0)
        Xc = X - mean

        actual_k = min(k, m, n)
        n_samples = min(actual_k + self.oversampling, n)

        # Length-squared column sampling from X^T X (via column norms of Xc)
        col_norms_sq = np.sum(Xc ** 2, axis=0)
        total = col_norms_sq.sum()
        if total < 1e-15:
            col_probs = np.ones(n) / n
        else:
            col_probs = col_norms_sq / total

        col_indices = rng.choice(n, size=n_samples, replace=False if n_samples <= n else True, p=col_probs)

        # Form the Nystrom sketch
        # C = Xc[:, col_indices] (m x c)
        C = Xc[:, col_indices]

        # Gram matrix of sampled columns in feature space
        # W = C^T C / m  (c x c) -- approximate covariance submatrix
        W = (C.T @ C) / m

        # Eigendecompose W
        eigvals, eigvecs = np.linalg.eigh(W)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep top-k
        eigvals_k = eigvals[:actual_k]
        eigvecs_k = eigvecs[:, :actual_k]

        # Nystrom extension: approximate eigenvectors in full feature space
        # V_full ~ Xc^T @ C @ eigvecs_k / (m * sqrt(eigvals_k))
        safe_eigvals = np.maximum(eigvals_k, 1e-15)
        components = (Xc.T @ (C @ eigvecs_k)) / (m * np.sqrt(safe_eigvals))

        # Normalise each component
        for j in range(actual_k):
            norm = np.linalg.norm(components[:, j])
            if norm > 1e-15:
                components[:, j] /= norm

        # components is (n, k) -- transpose to (k, n) for convention
        components = components.T

        # Explained variance ratio
        total_variance = total / m  # total variance = sum of col_norms_sq / m
        evr = eigvals_k / max(total_variance, 1e-15)
        # Clip ratios to [0, 1]
        evr = np.clip(evr, 0.0, 1.0)

        return QIPCAResult(
            components=components,
            explained_variance_ratio=evr,
            n_samples=n_samples,
            total_variance=float(total_variance),
        )
