"""
Quantum Volatility Surface Tools.

Models implied-volatility (IV) surfaces using quantum state evolution.
Classical market observables -- underlying price, strike, time-to-expiry,
and current IV -- are encoded into a quantum state via angle encoding.
A parameterised Hamiltonian then evolves that state, and the Born rule
extracts measurement probabilities that are mapped back to IV predictions.

The pipeline is:

    market features --> QuantumStateEncoder (angle encoding)
                    --> HamiltonianEvolution (Rz, Ry, entanglement)
                    --> BornRuleMeasurement (Born rule probabilities)
                    --> IV surface prediction

This is a *generic tool*, not a proprietary trading strategy.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.quantum_volatility import QuantumVolatilitySurface
>>>
>>> strikes = np.linspace(90, 110, 11)
>>> expiries = np.array([0.08, 0.25, 0.5, 1.0])
>>> ivs = 0.20 + 0.05 * np.random.randn(len(strikes), len(expiries))
>>>
>>> surface = QuantumVolatilitySurface(n_qubits=4, n_layers=3)
>>> surface.fit(strikes, expiries, ivs, spot=100.0)
>>> predicted = surface.predict(strikes, expiries, spot=100.0)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Quantum State Encoder
# ---------------------------------------------------------------------------

class QuantumStateEncoder:
    """Encode classical market features into a quantum state vector.

    Uses *angle encoding*: each feature is mapped to a rotation angle in
    ``[0, pi]`` via min-max normalisation, then a single-qubit rotation
    ``Ry(theta)`` is applied to the corresponding qubit.  The resulting
    product state lives in a ``2**n``-dimensional Hilbert space, which gives
    the encoder exponential representational capacity for feature
    interactions through subsequent entanglement layers.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= number of features to encode).

    Example
    -------
    >>> enc = QuantumStateEncoder(n_qubits=4)
    >>> features = np.array([100.0, 105.0, 0.25, 0.22])
    >>> state = enc.encode(features)
    >>> assert state.shape == (16,)  # 2**4
    >>> assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def fit_scaling(self, features: np.ndarray) -> "QuantumStateEncoder":
        """Learn min/max scaling from a feature matrix.

        Parameters
        ----------
        features : ndarray of shape ``(n_samples, n_features)``
            Training features.

        Returns
        -------
        self
        """
        features = np.atleast_2d(features)
        self._feature_min = features.min(axis=0)
        self._feature_max = features.max(axis=0)
        # Avoid division by zero for constant features.
        span = self._feature_max - self._feature_min
        span[span == 0.0] = 1.0
        self._feature_max = self._feature_min + span
        return self

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode a single feature vector into a quantum state.

        Parameters
        ----------
        features : ndarray of shape ``(n_features,)``
            Classical feature vector.  ``n_features <= n_qubits``.

        Returns
        -------
        state : ndarray of shape ``(2**n_qubits,)``
            Complex-valued normalised state vector.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 1:
            raise ValueError("features must be a 1-D array")
        n_features = features.shape[0]
        if n_features > self.n_qubits:
            raise ValueError(
                f"Cannot encode {n_features} features into {self.n_qubits} qubits"
            )

        angles = self._features_to_angles(features)

        # Build product state via single-qubit Ry rotations.
        state = np.array([1.0 + 0j], dtype=np.complex128)
        for i in range(self.n_qubits):
            theta = angles[i] if i < len(angles) else 0.0
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)

        return state

    def encode_batch(self, features: np.ndarray) -> np.ndarray:
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

    # -- internals ----------------------------------------------------------

    def _features_to_angles(self, features: np.ndarray) -> np.ndarray:
        """Map features to rotation angles in ``[0, pi]``."""
        if self._feature_min is not None:
            span = self._feature_max - self._feature_min
            normed = (features - self._feature_min[: len(features)]) / span[: len(features)]
            normed = np.clip(normed, 0.0, 1.0)
        else:
            # Fallback: sigmoid-style squash to [0, 1].
            normed = 1.0 / (1.0 + np.exp(-features))
        return normed * np.pi


# ---------------------------------------------------------------------------
# Hamiltonian Evolution Layer
# ---------------------------------------------------------------------------

class HamiltonianEvolution:
    """Parameterised Hamiltonian evolution layer.

    Applies a trainable sequence of single-qubit rotations (``Rz``, ``Ry``)
    followed by nearest-neighbour ``CNOT`` entanglement.  Stacking multiple
    layers increases the expressivity of the quantum model.

    The unitary for one layer acting on state ``|psi>`` is::

        U(theta) = ENTANGLE . prod_i Ry(theta_i^y) Rz(theta_i^z)

    where ``ENTANGLE`` is a ladder of CNOTs coupling qubit *i* to *i+1*.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of sequential evolution layers.
    seed : int or None
        Random seed for parameter initialisation.

    Example
    -------
    >>> evo = HamiltonianEvolution(n_qubits=4, n_layers=2, seed=42)
    >>> state = np.zeros(16, dtype=complex); state[0] = 1.0
    >>> evolved = evo.evolve(state)
    >>> assert np.isclose(np.sum(np.abs(evolved) ** 2), 1.0)
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        rng = np.random.default_rng(seed)
        # Each layer has 2 angles (Rz, Ry) per qubit.
        self.params = rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 2))

    # -- public API ---------------------------------------------------------

    def evolve(self, state: np.ndarray) -> np.ndarray:
        """Apply the parameterised evolution to a state vector.

        Parameters
        ----------
        state : ndarray of shape ``(2**n_qubits,)``

        Returns
        -------
        evolved : ndarray of shape ``(2**n_qubits,)``
        """
        state = np.array(state, dtype=np.complex128)
        if state.shape != (self.dim,):
            raise ValueError(f"Expected state of dimension {self.dim}")

        for layer_idx in range(self.n_layers):
            # Single-qubit rotations.
            for q in range(self.n_qubits):
                rz_angle = self.params[layer_idx, q, 0]
                ry_angle = self.params[layer_idx, q, 1]
                state = self._apply_rz(state, q, rz_angle)
                state = self._apply_ry(state, q, ry_angle)

            # Entanglement: nearest-neighbour CNOTs.
            for q in range(self.n_qubits - 1):
                state = self._apply_cnot(state, q, q + 1)

        return state

    def set_params(self, params: np.ndarray) -> None:
        """Set all trainable parameters at once.

        Parameters
        ----------
        params : ndarray of shape ``(n_layers, n_qubits, 2)``
        """
        expected = (self.n_layers, self.n_qubits, 2)
        if params.shape != expected:
            raise ValueError(f"Expected shape {expected}, got {params.shape}")
        self.params = np.array(params, dtype=np.float64)

    @property
    def num_params(self) -> int:
        """Total number of trainable parameters."""
        return self.n_layers * self.n_qubits * 2

    # -- gate primitives (state-vector simulation) --------------------------

    def _apply_rz(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply Rz(angle) to *qubit* in the full state vector."""
        phase = np.exp(-1j * angle / 2)
        new_state = state.copy()
        for i in range(self.dim):
            if (i >> (self.n_qubits - 1 - qubit)) & 1:
                new_state[i] *= phase.conjugate() / phase  # e^{+i*angle/2} / e^{-i*angle/2}
            # |0> component unchanged up to global phase.
        # Rz = diag(e^{-i*a/2}, e^{+i*a/2})
        for i in range(self.dim):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            if bit == 0:
                new_state[i] = state[i] * np.exp(-1j * angle / 2)
            else:
                new_state[i] = state[i] * np.exp(1j * angle / 2)
        return new_state

    def _apply_ry(self, state: np.ndarray, qubit: int, angle: float) -> np.ndarray:
        """Apply Ry(angle) to *qubit*."""
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        new_state = state.copy()
        step = 1 << (self.n_qubits - 1 - qubit)
        for i in range(self.dim):
            bit = (i >> (self.n_qubits - 1 - qubit)) & 1
            partner = i ^ step
            if bit == 0 and partner > i:
                a0 = state[i]
                a1 = state[partner]
                new_state[i] = c * a0 - s * a1
                new_state[partner] = s * a0 + c * a1
        return new_state

    def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
        """Apply CNOT with given control and target qubits."""
        new_state = state.copy()
        t_step = 1 << (self.n_qubits - 1 - target)
        for i in range(self.dim):
            ctrl_bit = (i >> (self.n_qubits - 1 - control)) & 1
            tgt_bit = (i >> (self.n_qubits - 1 - target)) & 1
            if ctrl_bit == 1 and tgt_bit == 0:
                partner = i ^ t_step
                new_state[i], new_state[partner] = state[partner], state[i]
        return new_state


# ---------------------------------------------------------------------------
# Born Rule Measurement
# ---------------------------------------------------------------------------

class BornRuleMeasurement:
    """Extract probabilities and expectation values from quantum states.

    Given a state vector ``|psi>``, the Born rule yields measurement
    probabilities ``p_i = |<i|psi>|^2``.  This class converts those raw
    probabilities into useful financial outputs (e.g. predicted IV).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    output_range : tuple of (float, float)
        Target output range for the linear mapping from probabilities.
        Default ``(0.05, 1.0)`` covers typical IV values.

    Example
    -------
    >>> brm = BornRuleMeasurement(n_qubits=4)
    >>> state = np.zeros(16, dtype=complex); state[0] = 1.0
    >>> prob = brm.probabilities(state)
    >>> assert np.isclose(prob.sum(), 1.0)
    """

    def __init__(
        self,
        n_qubits: int,
        output_range: Tuple[float, float] = (0.05, 1.0),
    ) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.output_range = output_range
        # Trainable output weights.
        self.weights = np.ones(self.dim, dtype=np.float64) / self.dim

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Compute Born-rule probabilities.

        Parameters
        ----------
        state : ndarray of shape ``(2**n_qubits,)``

        Returns
        -------
        probs : ndarray of shape ``(2**n_qubits,)``
        """
        return np.abs(state) ** 2

    def expectation(self, state: np.ndarray) -> float:
        """Weighted expectation value mapped to *output_range*.

        Parameters
        ----------
        state : ndarray of shape ``(2**n_qubits,)``

        Returns
        -------
        value : float
            Scalar in ``[output_range[0], output_range[1]]``.
        """
        probs = self.probabilities(state)
        raw = float(np.dot(self.weights, probs))
        lo, hi = self.output_range
        return lo + (hi - lo) * np.clip(raw, 0.0, 1.0)

    def set_weights(self, weights: np.ndarray) -> None:
        """Set the output weights (length ``2**n_qubits``)."""
        if weights.shape != (self.dim,):
            raise ValueError(f"Expected weights of shape ({self.dim},)")
        self.weights = np.asarray(weights, dtype=np.float64)


# ---------------------------------------------------------------------------
# Quantum Volatility Surface (full pipeline)
# ---------------------------------------------------------------------------

@dataclass
class _FitState:
    """Internal state persisted after fitting."""
    encoder: QuantumStateEncoder = field(default=None)
    hamiltonian: HamiltonianEvolution = field(default=None)
    measurement: BornRuleMeasurement = field(default=None)
    iv_mean: float = 0.0
    iv_std: float = 1.0


class QuantumVolatilitySurface:
    """End-to-end quantum model for implied-volatility surfaces.

    Combines ``QuantumStateEncoder``, ``HamiltonianEvolution``, and
    ``BornRuleMeasurement`` into a single trainable pipeline that is fitted
    to observed IV data via simple gradient-free optimisation (Nelder-Mead).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (4 features are encoded: moneyness, expiry,
        current IV, and a volatility-of-volatility proxy).
    n_layers : int
        Hamiltonian evolution layers.
    seed : int or None
        Random seed.

    Example
    -------
    >>> surface = QuantumVolatilitySurface(n_qubits=4, n_layers=3)
    >>> strikes = np.linspace(90, 110, 11)
    >>> expiries = np.array([0.08, 0.25, 0.5, 1.0])
    >>> ivs = 0.20 + 0.02 * np.random.randn(len(strikes), len(expiries))
    >>> surface.fit(strikes, expiries, ivs, spot=100.0)
    >>> pred = surface.predict(strikes, expiries, spot=100.0)
    >>> assert pred.shape == ivs.shape
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = max(n_qubits, 4)
        self.n_layers = n_layers
        self.seed = seed
        self._state: Optional[_FitState] = None

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_matrix: np.ndarray,
        spot: float,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> "QuantumVolatilitySurface":
        """Fit the surface to observed implied volatilities.

        Parameters
        ----------
        strikes : ndarray of shape ``(n_strikes,)``
        expiries : ndarray of shape ``(n_expiries,)``
        iv_matrix : ndarray of shape ``(n_strikes, n_expiries)``
            Observed IV values.
        spot : float
            Current underlying price.
        max_iter : int
            Maximum optimisation iterations.
        tol : float
            Convergence tolerance on MSE.

        Returns
        -------
        self
        """
        strikes = np.asarray(strikes, dtype=np.float64)
        expiries = np.asarray(expiries, dtype=np.float64)
        iv_matrix = np.asarray(iv_matrix, dtype=np.float64)

        if iv_matrix.shape != (len(strikes), len(expiries)):
            raise ValueError(
                f"iv_matrix shape {iv_matrix.shape} does not match "
                f"({len(strikes)}, {len(expiries)})"
            )

        # Build feature matrix: [moneyness, expiry, iv, vol-of-vol proxy].
        features, iv_flat = self._build_features(strikes, expiries, iv_matrix, spot)

        iv_mean = iv_flat.mean()
        iv_std = iv_flat.std() + 1e-12

        encoder = QuantumStateEncoder(self.n_qubits)
        encoder.fit_scaling(features)

        hamiltonian = HamiltonianEvolution(self.n_qubits, self.n_layers, seed=self.seed)
        measurement = BornRuleMeasurement(
            self.n_qubits, output_range=(iv_flat.min() * 0.8, iv_flat.max() * 1.2)
        )

        # Gradient-free optimisation (Nelder-Mead via numpy).
        flat_params = hamiltonian.params.ravel().copy()
        output_weights = measurement.weights.copy()
        all_params = np.concatenate([flat_params, output_weights])

        best_params = all_params.copy()
        best_loss = np.inf

        # Simple coordinate-descent + random perturbation optimiser.
        rng = np.random.default_rng(self.seed)
        for iteration in range(max_iter):
            # Perturb parameters.
            step_size = 0.5 * (1.0 - iteration / max_iter)
            candidate = best_params + rng.normal(0, step_size, size=best_params.shape)

            n_ham = flat_params.shape[0]
            hamiltonian.set_params(candidate[:n_ham].reshape(hamiltonian.params.shape))
            measurement.set_weights(np.abs(candidate[n_ham:]))

            loss = self._compute_loss(encoder, hamiltonian, measurement, features, iv_flat)

            if loss < best_loss:
                best_loss = loss
                best_params = candidate.copy()

            if best_loss < tol:
                break

        # Restore best parameters.
        n_ham = flat_params.shape[0]
        hamiltonian.set_params(best_params[:n_ham].reshape(hamiltonian.params.shape))
        measurement.set_weights(np.abs(best_params[n_ham:]))

        self._state = _FitState(
            encoder=encoder,
            hamiltonian=hamiltonian,
            measurement=measurement,
            iv_mean=iv_mean,
            iv_std=iv_std,
        )
        return self

    def predict(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,
        spot: float,
    ) -> np.ndarray:
        """Predict IV surface for given strikes and expiries.

        Parameters
        ----------
        strikes : ndarray of shape ``(n_strikes,)``
        expiries : ndarray of shape ``(n_expiries,)``
        spot : float

        Returns
        -------
        iv_pred : ndarray of shape ``(n_strikes, n_expiries)``
        """
        if self._state is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        s = self._state
        predictions = np.empty((len(strikes), len(expiries)))
        for i, k in enumerate(strikes):
            for j, t in enumerate(expiries):
                moneyness = np.log(k / spot)
                vol_proxy = np.abs(moneyness) / np.sqrt(t + 1e-12)
                feat = np.array([moneyness, t, 0.20, vol_proxy])
                state = s.encoder.encode(feat)
                evolved = s.hamiltonian.evolve(state)
                predictions[i, j] = s.measurement.expectation(evolved)

        return predictions

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _build_features(
        strikes: np.ndarray,
        expiries: np.ndarray,
        iv_matrix: np.ndarray,
        spot: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (n_samples, 4) feature matrix and flat IV target."""
        rows = []
        iv_flat = []
        for i, k in enumerate(strikes):
            for j, t in enumerate(expiries):
                moneyness = np.log(k / spot)
                vol_proxy = np.abs(moneyness) / np.sqrt(t + 1e-12)
                rows.append([moneyness, t, iv_matrix[i, j], vol_proxy])
                iv_flat.append(iv_matrix[i, j])
        return np.array(rows), np.array(iv_flat)

    @staticmethod
    def _compute_loss(
        encoder: QuantumStateEncoder,
        hamiltonian: HamiltonianEvolution,
        measurement: BornRuleMeasurement,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        """Mean squared error between predictions and targets."""
        errors = []
        for feat, target in zip(features, targets):
            state = encoder.encode(feat)
            evolved = hamiltonian.evolve(state)
            pred = measurement.expectation(evolved)
            errors.append((pred - target) ** 2)
        return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def interpolate_iv_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    iv_matrix: np.ndarray,
    target_strikes: np.ndarray,
    target_expiries: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of an IV surface onto a new grid.

    Parameters
    ----------
    strikes, expiries : ndarray
        Original grid axes.
    iv_matrix : ndarray of shape ``(len(strikes), len(expiries))``
    target_strikes, target_expiries : ndarray
        New grid axes.

    Returns
    -------
    interpolated : ndarray of shape ``(len(target_strikes), len(target_expiries))``

    Example
    -------
    >>> s = np.array([90, 100, 110], dtype=float)
    >>> t = np.array([0.25, 0.5], dtype=float)
    >>> iv = np.array([[0.30, 0.28], [0.20, 0.19], [0.25, 0.24]])
    >>> new_s = np.array([95.0, 105.0])
    >>> new_t = np.array([0.375])
    >>> result = interpolate_iv_surface(s, t, iv, new_s, new_t)
    >>> assert result.shape == (2, 1)
    """
    result = np.empty((len(target_strikes), len(target_expiries)))
    for i, tk in enumerate(target_strikes):
        for j, te in enumerate(target_expiries):
            # Find bounding indices.
            ki = np.searchsorted(strikes, tk, side="right") - 1
            ki = np.clip(ki, 0, len(strikes) - 2)
            ei = np.searchsorted(expiries, te, side="right") - 1
            ei = np.clip(ei, 0, len(expiries) - 2)

            # Bilinear weights.
            dk = (tk - strikes[ki]) / (strikes[ki + 1] - strikes[ki] + 1e-12)
            de = (te - expiries[ei]) / (expiries[ei + 1] - expiries[ei] + 1e-12)
            dk = np.clip(dk, 0.0, 1.0)
            de = np.clip(de, 0.0, 1.0)

            v00 = iv_matrix[ki, ei]
            v10 = iv_matrix[ki + 1, ei]
            v01 = iv_matrix[ki, ei + 1]
            v11 = iv_matrix[ki + 1, ei + 1]

            result[i, j] = (
                v00 * (1 - dk) * (1 - de)
                + v10 * dk * (1 - de)
                + v01 * (1 - dk) * de
                + v11 * dk * de
            )
    return result


def extrapolate_iv_surface(
    strikes: np.ndarray,
    expiries: np.ndarray,
    iv_matrix: np.ndarray,
    target_strikes: np.ndarray,
    target_expiries: np.ndarray,
    method: str = "flat",
) -> np.ndarray:
    """Extrapolate an IV surface beyond its observed range.

    Parameters
    ----------
    strikes, expiries, iv_matrix
        Observed surface.
    target_strikes, target_expiries
        Target grid (may extend beyond observed range).
    method : ``"flat"`` | ``"linear"``
        ``"flat"`` clamps to boundary values; ``"linear"`` extends the
        boundary slope.

    Returns
    -------
    extrapolated : ndarray of shape ``(len(target_strikes), len(target_expiries))``
    """
    result = np.empty((len(target_strikes), len(target_expiries)))

    for i, tk in enumerate(target_strikes):
        # Clamp or extrapolate along strike axis.
        if tk <= strikes[0]:
            ki, wk = 0, 0.0
            if method == "linear" and len(strikes) >= 2:
                slope_k = (iv_matrix[1, :] - iv_matrix[0, :]) / (strikes[1] - strikes[0] + 1e-12)
            else:
                slope_k = np.zeros(len(expiries))
            dist_k = tk - strikes[0]
        elif tk >= strikes[-1]:
            ki, wk = len(strikes) - 1, 0.0
            if method == "linear" and len(strikes) >= 2:
                slope_k = (iv_matrix[-1, :] - iv_matrix[-2, :]) / (strikes[-1] - strikes[-2] + 1e-12)
            else:
                slope_k = np.zeros(len(expiries))
            dist_k = tk - strikes[-1]
        else:
            ki = int(np.searchsorted(strikes, tk, side="right")) - 1
            ki = np.clip(ki, 0, len(strikes) - 2)
            wk = (tk - strikes[ki]) / (strikes[ki + 1] - strikes[ki] + 1e-12)
            slope_k = None
            dist_k = 0.0

        for j, te in enumerate(target_expiries):
            if slope_k is not None:
                # Extrapolating along strike.
                base = iv_matrix[ki, np.clip(np.searchsorted(expiries, te) - 1, 0, len(expiries) - 1)]
                result[i, j] = base + slope_k[
                    np.clip(np.searchsorted(expiries, te) - 1, 0, len(expiries) - 1)
                ] * dist_k
            else:
                # Interpolate along strike, handle expiry bounds.
                if te <= expiries[0]:
                    v = iv_matrix[ki, 0] * (1 - wk) + iv_matrix[ki + 1, 0] * wk
                elif te >= expiries[-1]:
                    v = iv_matrix[ki, -1] * (1 - wk) + iv_matrix[ki + 1, -1] * wk
                else:
                    ei = int(np.searchsorted(expiries, te, side="right")) - 1
                    ei = np.clip(ei, 0, len(expiries) - 2)
                    we = (te - expiries[ei]) / (expiries[ei + 1] - expiries[ei] + 1e-12)
                    v00 = iv_matrix[ki, ei]
                    v10 = iv_matrix[ki + 1, ei]
                    v01 = iv_matrix[ki, ei + 1]
                    v11 = iv_matrix[ki + 1, ei + 1]
                    v = v00 * (1 - wk) * (1 - we) + v10 * wk * (1 - we) + v01 * (1 - wk) * we + v11 * wk * we
                result[i, j] = v

    return result
