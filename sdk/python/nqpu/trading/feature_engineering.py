"""
Quantum Feature Engineering.

Maps classical financial features into quantum Hilbert spaces using
several encoding strategies (angle, amplitude, ZZ-feature-map) and
provides quantum kernel functions for similarity computation between
financial time series.

The key insight is that quantum feature maps can capture non-linear
feature interactions that are exponentially hard to represent classically.
For ``n`` qubits, the Hilbert space dimension is ``2^n``, so even a small
number of qubits creates a rich feature space for financial pattern
recognition.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.feature_engineering import (
...     QuantumFeatureMap,
...     QuantumKernelSimilarity,
...     EntanglementFeatures,
...     compute_financial_features,
... )
>>>
>>> prices = np.cumsum(np.random.randn(500)) + 100
>>> features = compute_financial_features(prices, window=20)
>>> qfm = QuantumFeatureMap(n_qubits=4, encoding="angle")
>>> states = qfm.transform(features)
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Encoding types
# ---------------------------------------------------------------------------

class EncodingType(Enum):
    """Supported quantum encoding strategies."""
    ANGLE = "angle"
    AMPLITUDE = "amplitude"
    ZZ = "zz"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Quantum Feature Map
# ---------------------------------------------------------------------------

class QuantumFeatureMap:
    """Map classical feature vectors into a quantum Hilbert space.

    Supports three encoding strategies:

    - **angle**: Each feature becomes a rotation angle applied to one qubit.
      Efficient (``O(n)`` gates) but limited to ``n`` features.
    - **amplitude**: Feature vector is normalised and used directly as the
      state-vector amplitudes.  Encodes ``2^n`` features but requires
      exponential state-preparation depth in general.
    - **zz**: A ZZ-feature-map applies single-qubit rotations followed by
      entangling ``exp(i * x_i * x_j * ZZ)`` interactions, capturing
      pairwise feature correlations naturally.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    encoding : str or EncodingType
        Encoding strategy.
    n_layers : int
        Repetitions for ZZ encoding (ignored for others).

    Example
    -------
    >>> qfm = QuantumFeatureMap(n_qubits=4, encoding="angle")
    >>> x = np.array([0.1, 0.3, 0.5, 0.7])
    >>> state = qfm.encode(x)
    >>> assert state.shape == (16,)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        encoding: str = "angle",
        n_layers: int = 2,
    ) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.encoding = EncodingType(encoding) if isinstance(encoding, str) else encoding
        self.n_layers = n_layers

        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def fit(self, features: np.ndarray) -> "QuantumFeatureMap":
        """Learn feature scaling from training data.

        Parameters
        ----------
        features : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        self
        """
        features = np.atleast_2d(features)
        self._feature_min = features.min(axis=0)
        self._feature_max = features.max(axis=0)
        span = self._feature_max - self._feature_min
        span[span == 0.0] = 1.0
        self._feature_max = self._feature_min + span
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a single feature vector.

        Parameters
        ----------
        x : ndarray of shape ``(n_features,)``

        Returns
        -------
        state : ndarray of shape ``(2**n_qubits,)``
        """
        x = np.asarray(x, dtype=np.float64)
        x_normed = self._normalise(x)

        if self.encoding == EncodingType.ANGLE:
            return self._angle_encode(x_normed)
        elif self.encoding == EncodingType.AMPLITUDE:
            return self._amplitude_encode(x_normed)
        elif self.encoding == EncodingType.ZZ:
            return self._zz_encode(x_normed)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Encode a batch of feature vectors.

        Parameters
        ----------
        features : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        states : ndarray of shape ``(n_samples, 2**n_qubits)``
        """
        features = np.atleast_2d(features)
        return np.array([self.encode(row) for row in features])

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features)
        return self.transform(features)

    # -- encoding implementations -------------------------------------------

    def _angle_encode(self, x: np.ndarray) -> np.ndarray:
        """Ry angle encoding."""
        angles = x[: self.n_qubits] * np.pi
        state = np.array([1.0 + 0j], dtype=np.complex128)
        for i in range(self.n_qubits):
            theta = angles[i] if i < len(angles) else 0.0
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state

    def _amplitude_encode(self, x: np.ndarray) -> np.ndarray:
        """Amplitude encoding (normalised feature vector as state)."""
        amplitudes = np.zeros(self.dim, dtype=np.complex128)
        n = min(len(x), self.dim)
        amplitudes[:n] = x[:n]
        norm = np.linalg.norm(amplitudes)
        if norm < 1e-15:
            amplitudes[0] = 1.0
        else:
            amplitudes /= norm
        return amplitudes

    def _zz_encode(self, x: np.ndarray) -> np.ndarray:
        """ZZ-feature-map encoding with pairwise entanglement.

        Applies ``n_layers`` repetitions of:
        1. ``Ry(x_i)`` on each qubit.
        2. ``exp(i * x_i * x_j)`` phase on each pair ``(i, j)`` via
           ``CNOT - Rz(x_i * x_j) - CNOT``.
        """
        # Start in |0...0>.
        state = np.zeros(self.dim, dtype=np.complex128)
        state[0] = 1.0

        x_padded = np.zeros(self.n_qubits)
        x_padded[: min(len(x), self.n_qubits)] = x[: self.n_qubits]

        for _ in range(self.n_layers):
            # Single-qubit Ry rotations.
            for q in range(self.n_qubits):
                state = _apply_ry(state, self.n_qubits, q, x_padded[q] * np.pi)

            # ZZ entanglement: apply a phase based on pairwise feature products.
            for q1 in range(self.n_qubits):
                for q2 in range(q1 + 1, self.n_qubits):
                    phase_angle = x_padded[q1] * x_padded[q2] * np.pi
                    state = _apply_zz_phase(
                        state, self.n_qubits, q1, q2, phase_angle
                    )

        return state

    # -- helpers ------------------------------------------------------------

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Normalise features to [0, 1]."""
        if self._feature_min is not None:
            span = self._feature_max - self._feature_min
            n = min(len(x), len(span))
            normed = (x[:n] - self._feature_min[:n]) / span[:n]
            return np.clip(normed, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Quantum Kernel Similarity
# ---------------------------------------------------------------------------

class QuantumKernelSimilarity:
    """Compute quantum kernel matrices for financial time series.

    The quantum kernel between two data points ``x`` and ``y`` is defined as::

        K(x, y) = |<phi(x)|phi(y)>|^2

    where ``|phi(x)>`` is the quantum state produced by a feature map.
    This kernel implicitly maps data into an exponentially large Hilbert
    space, potentially capturing complex non-linear relationships.

    Parameters
    ----------
    feature_map : QuantumFeatureMap
        Feature map to use for encoding.

    Example
    -------
    >>> qfm = QuantumFeatureMap(n_qubits=4, encoding="zz")
    >>> qks = QuantumKernelSimilarity(qfm)
    >>> X = np.random.randn(50, 4)
    >>> K = qks.kernel_matrix(X)
    >>> assert K.shape == (50, 50)
    >>> assert np.allclose(K, K.T)
    """

    def __init__(self, feature_map: QuantumFeatureMap) -> None:
        self.feature_map = feature_map

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the quantum kernel between two feature vectors.

        Parameters
        ----------
        x, y : ndarray of shape ``(n_features,)``

        Returns
        -------
        k : float in ``[0, 1]``
        """
        state_x = self.feature_map.encode(x)
        state_y = self.feature_map.encode(y)
        return float(np.abs(np.vdot(state_x, state_y)) ** 2)

    def kernel_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the full kernel (Gram) matrix.

        Parameters
        ----------
        X : ndarray of shape ``(n_x, n_features)``
        Y : ndarray of shape ``(n_y, n_features)`` or None
            If ``None``, computes ``K(X, X)``.

        Returns
        -------
        K : ndarray of shape ``(n_x, n_y)``
        """
        X = np.atleast_2d(X)
        states_x = self.feature_map.transform(X)

        if Y is None:
            n = len(states_x)
            K = np.empty((n, n))
            for i in range(n):
                K[i, i] = 1.0
                for j in range(i + 1, n):
                    k_ij = float(np.abs(np.vdot(states_x[i], states_x[j])) ** 2)
                    K[i, j] = k_ij
                    K[j, i] = k_ij
            return K

        Y = np.atleast_2d(Y)
        states_y = self.feature_map.transform(Y)
        K = np.empty((len(states_x), len(states_y)))
        for i, sx in enumerate(states_x):
            for j, sy in enumerate(states_y):
                K[i, j] = float(np.abs(np.vdot(sx, sy)) ** 2)
        return K

    def target_alignment(
        self, X: np.ndarray, y: np.ndarray
    ) -> float:
        """Kernel-target alignment score.

        Measures how well the kernel matrix aligns with the ideal kernel
        derived from labels.  Values close to 1.0 indicate good alignment.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``
        y : ndarray of shape ``(n_samples,)``
            Binary or continuous labels.

        Returns
        -------
        alignment : float
        """
        K = self.kernel_matrix(X)
        y = np.asarray(y, dtype=np.float64)
        K_ideal = np.outer(y, y)

        num = np.sum(K * K_ideal)
        denom = np.sqrt(np.sum(K * K) * np.sum(K_ideal * K_ideal)) + 1e-15
        return float(num / denom)


# ---------------------------------------------------------------------------
# Entanglement Features
# ---------------------------------------------------------------------------

class EntanglementFeatures:
    """Extract correlation features using quantum entanglement measures.

    Given a set of asset return series, this class computes pairwise
    entanglement measures between assets by encoding joint distributions
    as quantum states and measuring entanglement entropy.

    Parameters
    ----------
    n_qubits_per_asset : int
        Qubits allocated to each asset (default 2).

    Example
    -------
    >>> ef = EntanglementFeatures(n_qubits_per_asset=2)
    >>> returns = np.random.randn(200, 4)  # 4 assets
    >>> ent_matrix = ef.pairwise_entanglement(returns)
    >>> assert ent_matrix.shape == (4, 4)
    """

    def __init__(self, n_qubits_per_asset: int = 2) -> None:
        self.n_qubits_per_asset = n_qubits_per_asset
        self.dim_per_asset = 2 ** n_qubits_per_asset

    def entanglement_entropy(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> float:
        """Von Neumann entanglement entropy between two asset return series.

        Joint distribution is encoded as a bipartite quantum state and the
        reduced density matrix of subsystem A is computed.  Its von Neumann
        entropy quantifies entanglement.

        Parameters
        ----------
        returns_a, returns_b : ndarray of shape ``(n_periods,)``

        Returns
        -------
        entropy : float >= 0
        """
        joint_state = self._encode_joint(returns_a, returns_b)
        rho_a = self._partial_trace_b(joint_state)
        return self._von_neumann_entropy(rho_a)

    def pairwise_entanglement(self, returns: np.ndarray) -> np.ndarray:
        """Compute pairwise entanglement matrix for multiple assets.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``

        Returns
        -------
        ent_matrix : ndarray of shape ``(n_assets, n_assets)``
            Symmetric matrix of entanglement entropy values.
        """
        n_assets = returns.shape[1]
        ent = np.zeros((n_assets, n_assets))
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                e = self.entanglement_entropy(returns[:, i], returns[:, j])
                ent[i, j] = e
                ent[j, i] = e
        return ent

    def entanglement_features(self, returns: np.ndarray) -> np.ndarray:
        """Flatten pairwise entanglement into a feature vector.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods, n_assets)``

        Returns
        -------
        features : ndarray of shape ``(n_assets * (n_assets - 1) / 2,)``
        """
        ent = self.pairwise_entanglement(returns)
        n = ent.shape[0]
        feats = []
        for i in range(n):
            for j in range(i + 1, n):
                feats.append(ent[i, j])
        return np.array(feats)

    # -- internals ----------------------------------------------------------

    def _encode_joint(
        self, returns_a: np.ndarray, returns_b: np.ndarray
    ) -> np.ndarray:
        """Encode joint distribution of two series as a bipartite state."""
        n_qubits = 2 * self.n_qubits_per_asset
        dim = 2 ** n_qubits

        # Build histogram of joint distribution.
        bins = self.dim_per_asset
        hist, _, _ = np.histogram2d(
            returns_a, returns_b, bins=bins, density=True
        )
        amplitudes = hist.ravel().astype(np.float64)
        amplitudes = np.sqrt(np.abs(amplitudes))
        norm = np.linalg.norm(amplitudes)
        if norm < 1e-15:
            amplitudes = np.ones(dim) / np.sqrt(dim)
        else:
            amplitudes /= norm

        # Pad or truncate to dim.
        state = np.zeros(dim, dtype=np.complex128)
        n = min(len(amplitudes), dim)
        state[:n] = amplitudes[:n]
        norm = np.linalg.norm(state)
        if norm > 1e-15:
            state /= norm
        else:
            state[0] = 1.0
        return state

    def _partial_trace_b(self, joint_state: np.ndarray) -> np.ndarray:
        """Trace out subsystem B to get reduced density matrix of A."""
        da = self.dim_per_asset
        db = self.dim_per_asset
        # Reshape joint state into (da, db).
        psi = joint_state[: da * db].reshape(da, db)
        # rho_A = Tr_B(|psi><psi|) = psi @ psi^dagger
        rho_a = psi @ psi.conj().T
        return rho_a

    @staticmethod
    def _von_neumann_entropy(rho: np.ndarray) -> float:
        """Von Neumann entropy S = -Tr(rho log rho)."""
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


# ---------------------------------------------------------------------------
# Common financial features
# ---------------------------------------------------------------------------

def compute_financial_features(
    prices: np.ndarray,
    window: int = 20,
    include_volume: bool = False,
    volume: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute standard financial features from a price series.

    Features per window: log-return mean, volatility, skewness, kurtosis,
    and optionally a normalised volume profile.

    Parameters
    ----------
    prices : ndarray of shape ``(n_periods,)``
        Price series (not returns).
    window : int
        Rolling window size.
    include_volume : bool
        Whether to append normalised volume.
    volume : ndarray of shape ``(n_periods,)`` or None
        Volume series (required if ``include_volume=True``).

    Returns
    -------
    features : ndarray of shape ``(n_windows, 4)`` or ``(n_windows, 5)``

    Example
    -------
    >>> prices = np.cumsum(np.random.randn(300)) + 100
    >>> feat = compute_financial_features(prices, window=20)
    >>> assert feat.shape[1] == 4
    """
    prices = np.asarray(prices, dtype=np.float64)
    log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))

    n_windows = len(log_returns) - window + 1
    n_feat = 5 if include_volume else 4
    features = np.empty((n_windows, n_feat))

    for i in range(n_windows):
        w = log_returns[i: i + window]
        mu = w.mean()
        sigma = w.std() + 1e-12
        skew = float(np.mean(((w - mu) / sigma) ** 3))
        kurt = float(np.mean(((w - mu) / sigma) ** 4)) - 3.0
        features[i, 0] = mu
        features[i, 1] = sigma
        features[i, 2] = skew
        features[i, 3] = kurt

        if include_volume and volume is not None:
            vol_window = volume[i + 1: i + 1 + window]
            vol_mean = vol_window.mean()
            vol_total_mean = volume.mean() + 1e-12
            features[i, 4] = vol_mean / vol_total_mean

    return features


# ---------------------------------------------------------------------------
# Internal gate helpers (shared with quantum_volatility)
# ---------------------------------------------------------------------------

def _apply_ry(
    state: np.ndarray, n_qubits: int, qubit: int, angle: float
) -> np.ndarray:
    """Apply Ry(angle) to a specific qubit in a state vector."""
    dim = len(state)
    c = np.cos(angle / 2)
    s = np.sin(angle / 2)
    new_state = state.copy()
    step = 1 << (n_qubits - 1 - qubit)
    for i in range(dim):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        partner = i ^ step
        if bit == 0 and partner > i:
            a0 = state[i]
            a1 = state[partner]
            new_state[i] = c * a0 - s * a1
            new_state[partner] = s * a0 + c * a1
    return new_state


def _apply_zz_phase(
    state: np.ndarray,
    n_qubits: int,
    q1: int,
    q2: int,
    angle: float,
) -> np.ndarray:
    """Apply exp(i * angle * Z_q1 Z_q2) phase to state.

    The ZZ interaction adds a phase ``exp(+i*angle)`` when both qubits are
    in the same state and ``exp(-i*angle)`` when they differ.
    """
    dim = len(state)
    new_state = state.copy()
    for i in range(dim):
        b1 = (i >> (n_qubits - 1 - q1)) & 1
        b2 = (i >> (n_qubits - 1 - q2)) & 1
        parity = 1 - 2 * (b1 ^ b2)  # +1 if same, -1 if different
        new_state[i] *= np.exp(1j * angle * parity)
    return new_state
