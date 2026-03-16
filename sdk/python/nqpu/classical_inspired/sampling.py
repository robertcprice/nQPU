"""Quantum-inspired classical sampling algorithms.

Three sampling techniques that borrow ideas from quantum computing:

1. **DequantizedSampler** -- Tang's dequantization technique for low-rank
   matrix operations using length-squared sampling.
2. **TNSampler** -- Tensor-network (MPS) sequential conditional sampling.
3. **QIMonteCarlo** -- Quantum-inspired lifted Markov chain Monte Carlo
   based on Szegedy's quantum walk framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class DequantizedSample:
    """Result of dequantized sampling."""

    sampled_row_indices: np.ndarray
    sampled_col_indices: np.ndarray
    weights: np.ndarray
    reconstruction_error: float
    n_queries: int


@dataclass
class TNSampleResult:
    """Result of tensor-network sampling."""

    samples: np.ndarray  # (n_samples, n_sites)
    log_probs: np.ndarray  # (n_samples,)
    acceptance_rate: float
    bond_dimension: int


@dataclass
class QIMCResult:
    """Result of quantum-inspired Monte Carlo."""

    samples: np.ndarray
    effective_sample_size: float
    mixing_diagnostics: dict
    n_steps: int
    acceptance_rate: float


# ---------------------------------------------------------------------------
# Dequantized Sampler (Tang's technique)
# ---------------------------------------------------------------------------

class DequantizedSampler:
    """Tang's dequantized sampling for low-rank matrices.

    Provides sub-linear-time approximate access to a low-rank matrix by
    maintaining a length-squared sampling data structure.  Core operations:

    - ``sample_row``: sample row index proportional to squared row norm.
    - ``sample_entry``: sample column from a given row proportional to |A_{ij}|^2.
    - ``query``: approximate entry retrieval for low-rank A.
    - ``recommend``: low-rank recommendation using dequantized SVD.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def sample_row(
        self, A: np.ndarray, n_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample row indices proportional to ||A[i, :]||^2.

        Returns (indices, probabilities).
        """
        rng = np.random.default_rng(self.seed)
        A = np.asarray(A, dtype=np.float64)
        row_norms_sq = np.sum(A ** 2, axis=1)
        total = row_norms_sq.sum()
        if total < 1e-15:
            # Uniform fallback for zero matrix
            probs = np.ones(A.shape[0]) / A.shape[0]
        else:
            probs = row_norms_sq / total
        indices = rng.choice(A.shape[0], size=n_samples, p=probs)
        return indices, probs[indices]

    def sample_entry(
        self, A: np.ndarray, row_idx: int, n_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample column indices from row ``row_idx`` proportional to |A[i,j]|^2.

        Returns (col_indices, probabilities).
        """
        rng = np.random.default_rng(self.seed + row_idx + 1)
        row = np.asarray(A[row_idx], dtype=np.float64)
        norms_sq = row ** 2
        total = norms_sq.sum()
        if total < 1e-15:
            probs = np.ones(row.shape[0]) / row.shape[0]
        else:
            probs = norms_sq / total
        cols = rng.choice(row.shape[0], size=n_samples, p=probs)
        return cols, probs[cols]

    def query(self, A: np.ndarray, i: int, j: int) -> float:
        """Return A[i, j] directly (exact for demonstration).

        In the full dequantized model this would use the sampling data
        structure for O(polylog) access on a low-rank representation.
        """
        return float(A[i, j])

    def recommend(
        self,
        A: np.ndarray,
        k: int = 5,
        n_row_samples: int = 0,
        n_col_samples: int = 0,
    ) -> DequantizedSample:
        """Low-rank recommendation via dequantized sampling.

        Approximates the rank-*k* SVD of *A* using length-squared
        sampling of rows and columns, then reconstructs.

        Parameters
        ----------
        A : ndarray of shape (m, n)
        k : target rank
        n_row_samples : rows to sample (default: 4*k)
        n_col_samples : cols to sample per row (default: 4*k)
        """
        rng = np.random.default_rng(self.seed)
        A = np.asarray(A, dtype=np.float64)
        m, n = A.shape

        if n_row_samples <= 0:
            n_row_samples = min(4 * k, m)
        if n_col_samples <= 0:
            n_col_samples = min(4 * k, n)

        # Step 1: sample rows proportional to squared norms
        row_norms_sq = np.sum(A ** 2, axis=1)
        total_norm_sq = row_norms_sq.sum()
        if total_norm_sq < 1e-15:
            row_probs = np.ones(m) / m
        else:
            row_probs = row_norms_sq / total_norm_sq
        row_indices = rng.choice(m, size=n_row_samples, p=row_probs)

        # Step 2: sample columns from each selected row
        col_indices_all = []
        for ri in row_indices:
            row = A[ri]
            row_sq = row ** 2
            row_total = row_sq.sum()
            if row_total < 1e-15:
                col_probs = np.ones(n) / n
            else:
                col_probs = row_sq / row_total
            ci = rng.choice(n, size=n_col_samples, p=col_probs)
            col_indices_all.append(ci)
        col_indices_flat = np.unique(np.concatenate(col_indices_all))

        # Step 3: form submatrix and compute low-rank SVD
        sub = A[np.ix_(row_indices, col_indices_flat)]
        actual_k = min(k, *sub.shape)
        U_sub, S_sub, Vt_sub = np.linalg.svd(sub, full_matrices=False)
        U_k = U_sub[:, :actual_k]
        S_k = S_sub[:actual_k]
        Vt_k = Vt_sub[:actual_k, :]

        # Step 4: reconstruct and measure error
        recon_sub = U_k * S_k @ Vt_k
        # Map back to full matrix for error estimation
        full_recon = np.zeros_like(A)
        for i_loc, i_glob in enumerate(row_indices):
            for j_loc, j_glob in enumerate(col_indices_flat):
                full_recon[i_glob, j_glob] = recon_sub[i_loc, j_loc]

        # Frobenius error on sampled entries
        sampled_orig = A[np.ix_(row_indices, col_indices_flat)]
        error = float(
            np.linalg.norm(sampled_orig - recon_sub) /
            max(np.linalg.norm(sampled_orig), 1e-15)
        )

        weights = np.ones(len(row_indices)) / len(row_indices)
        n_queries = n_row_samples * n_col_samples

        return DequantizedSample(
            sampled_row_indices=row_indices,
            sampled_col_indices=col_indices_flat,
            weights=weights,
            reconstruction_error=error,
            n_queries=n_queries,
        )


# ---------------------------------------------------------------------------
# Tensor-Network Sampler
# ---------------------------------------------------------------------------

class TNSampler:
    """Tensor-network (MPS) based sampler.

    Represents a probability distribution as a Matrix Product State and
    generates samples via sequential conditional sampling (left to right
    through the MPS chain).
    """

    def __init__(
        self,
        bond_dimension: int = 8,
        seed: int = 42,
    ) -> None:
        self.bond_dimension = bond_dimension
        self.seed = seed

    def sample(
        self,
        tensors: List[np.ndarray],
        n_samples: int = 100,
    ) -> TNSampleResult:
        """Sample from the distribution defined by MPS tensors.

        Parameters
        ----------
        tensors : list of ndarray
            MPS tensors.  ``tensors[i]`` has shape
            ``(bond_left, phys_dim, bond_right)``.  For the first and
            last tensors the corresponding boundary bond dimension is 1.
        n_samples : int
            Number of independent samples to draw.

        Returns
        -------
        TNSampleResult
        """
        rng = np.random.default_rng(self.seed)
        n_sites = len(tensors)
        phys_dim = tensors[0].shape[1]

        samples = np.empty((n_samples, n_sites), dtype=np.int64)
        log_probs = np.zeros(n_samples, dtype=np.float64)

        for s in range(n_samples):
            # Sequential conditional sampling left -> right
            # Maintain a left boundary vector
            left = np.ones((1,), dtype=np.float64)

            for site in range(n_sites):
                T = tensors[site]  # (bl, d, br)
                bl, d, br = T.shape

                # Contract left boundary with tensor for each physical index
                probs = np.empty(d, dtype=np.float64)
                contracted = np.empty((d, br), dtype=np.float64)
                for p in range(d):
                    # left @ T[:, p, :] -> (br,)
                    vec = left @ T[:, p, :]
                    contracted[p] = vec
                    probs[p] = np.dot(vec, vec)  # ||vec||^2

                total = probs.sum()
                if total < 1e-30:
                    probs = np.ones(d) / d
                    total = 1.0
                else:
                    probs = probs / total

                # Sample physical index
                choice = rng.choice(d, p=probs)
                samples[s, site] = choice
                log_probs[s] += np.log(max(probs[choice], 1e-30))

                # Update left boundary
                new_left = contracted[choice]
                norm = np.linalg.norm(new_left)
                if norm > 1e-15:
                    new_left = new_left / norm
                left = new_left

        return TNSampleResult(
            samples=samples,
            log_probs=log_probs,
            acceptance_rate=1.0,  # direct sampling, no rejection
            bond_dimension=self.bond_dimension,
        )

    @staticmethod
    def random_mps(
        n_sites: int,
        phys_dim: int = 2,
        bond_dim: int = 4,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """Create a random MPS (useful for testing).

        Returns a list of tensors with shapes consistent with open
        boundary conditions.
        """
        rng = np.random.default_rng(seed)
        tensors = []
        for i in range(n_sites):
            bl = 1 if i == 0 else min(bond_dim, phys_dim ** i)
            br = 1 if i == n_sites - 1 else min(bond_dim, phys_dim ** (i + 1))
            bl = min(bl, bond_dim)
            br = min(br, bond_dim)
            T = rng.standard_normal((bl, phys_dim, br))
            # Normalise per physical index for numerical stability
            for p in range(phys_dim):
                norm = np.linalg.norm(T[:, p, :])
                if norm > 1e-15:
                    T[:, p, :] /= norm
            tensors.append(T)
        return tensors


# ---------------------------------------------------------------------------
# Quantum-Inspired Monte Carlo
# ---------------------------------------------------------------------------

class QIMonteCarlo:
    """Quantum-inspired lifted Markov chain Monte Carlo.

    Implements a lifted MCMC inspired by Szegedy's quantum walk framework.
    The classical version doubles the state space to (position, direction),
    achieving faster mixing analogous to the quantum quadratic speedup.
    """

    def __init__(
        self,
        n_steps: int = 1000,
        n_burnin: int = 200,
        seed: int = 42,
    ) -> None:
        self.n_steps = n_steps
        self.n_burnin = n_burnin
        self.seed = seed

    def sample(
        self,
        target_probs: np.ndarray,
        n_samples: int = 500,
    ) -> QIMCResult:
        """Sample from a discrete distribution using lifted MCMC.

        Parameters
        ----------
        target_probs : ndarray of shape (N,)
            Unnormalised target distribution over N states.
        n_samples : int
            Number of samples to collect after burn-in.

        Returns
        -------
        QIMCResult
        """
        rng = np.random.default_rng(self.seed)
        target = np.asarray(target_probs, dtype=np.float64)
        N = target.shape[0]

        # Normalise
        target = target / target.sum()

        # Lifted MCMC: state = (position, direction)
        # direction in {-1, +1} determines proposal bias
        pos = rng.integers(0, N)
        direction = 1

        samples = []
        accepted = 0
        total = 0

        total_steps = self.n_burnin + n_samples

        for step in range(total_steps):
            # Proposal: move in current direction with wrap-around
            proposed = (pos + direction) % N

            # Metropolis-Hastings acceptance
            alpha = min(1.0, target[proposed] / max(target[pos], 1e-30))
            total += 1

            if rng.random() < alpha:
                pos = proposed
                accepted += 1
            else:
                # On rejection: reverse direction (the "lift")
                direction = -direction

            if step >= self.n_burnin:
                samples.append(pos)

        samples_arr = np.array(samples, dtype=np.int64)

        # Compute effective sample size via autocorrelation
        ess = self._effective_sample_size(samples_arr, target)

        # Mixing diagnostics
        # Empirical distribution vs target (TV distance)
        empirical = np.bincount(samples_arr, minlength=N).astype(np.float64)
        empirical /= empirical.sum()
        tv_distance = 0.5 * np.sum(np.abs(empirical - target))

        diagnostics = {
            "total_variation_distance": float(tv_distance),
            "empirical_distribution": empirical,
            "target_distribution": target,
        }

        return QIMCResult(
            samples=samples_arr,
            effective_sample_size=float(ess),
            mixing_diagnostics=diagnostics,
            n_steps=total_steps,
            acceptance_rate=accepted / max(total, 1),
        )

    @staticmethod
    def _effective_sample_size(
        samples: np.ndarray, target: np.ndarray
    ) -> float:
        """Estimate ESS from autocorrelation of the chain."""
        n = len(samples)
        if n < 2:
            return float(n)

        # Use the indicator of each state for ESS estimation
        # Simplified: use the sample values directly
        x = samples.astype(np.float64)
        mean = x.mean()
        var = x.var()
        if var < 1e-15:
            return float(n)

        # Compute autocorrelation up to lag min(n//2, 100)
        max_lag = min(n // 2, 100)
        autocorr_sum = 0.0
        centered = x - mean
        for lag in range(1, max_lag + 1):
            c = np.mean(centered[:n - lag] * centered[lag:]) / var
            if c < 0.05:  # truncate at small autocorrelation
                break
            autocorr_sum += c

        ess = n / (1.0 + 2.0 * autocorr_sum)
        return max(ess, 1.0)
