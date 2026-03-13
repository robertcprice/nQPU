"""
Quantum Regime Detection.

Classifies market regimes (bull, bear, sideways, volatile) by mapping
observable market statistics onto quantum state vectors and measuring
overlap (fidelity) with learned prototype states for each regime.

Regime transitions are modelled as quantum channels -- completely
positive trace-preserving (CPTP) maps -- which naturally enforce
probability conservation and allow for coherent superpositions of
regimes during transitional periods.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.regime_detection import (
...     QuantumRegimeDetector,
...     RegimeTransitionMatrix,
...     VolatilityRegimeClassifier,
... )
>>>
>>> returns = np.random.randn(500) * 0.02
>>> detector = QuantumRegimeDetector(n_qubits=3)
>>> detector.fit(returns, window=60)
>>> regime = detector.detect(returns[-60:])
>>> print(regime)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Market Regime enum
# ---------------------------------------------------------------------------

class MarketRegime(Enum):
    """Canonical market regime labels."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Quantum Regime Detector
# ---------------------------------------------------------------------------

@dataclass
class _RegimePrototype:
    """Learned quantum state prototype for a single regime."""
    regime: MarketRegime
    state: np.ndarray  # (dim,) complex state vector
    threshold: float = 0.5  # minimum fidelity for assignment


class QuantumRegimeDetector:
    """Classify market regimes via quantum state overlap.

    The detector works in three steps:

    1. **Feature extraction** -- rolling statistics (mean return, volatility,
       skewness, kurtosis) are computed from a return series.
    2. **State encoding** -- each feature window is angle-encoded into a
       quantum state in a ``2**n_qubits``-dimensional Hilbert space.
    3. **Regime assignment** -- the encoded state is compared against learned
       *prototype states* for each regime using quantum state fidelity
       ``F = |<proto|psi>|^2``.  The regime with the highest fidelity wins.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (>= 4 recommended for four features).
    regimes : list of MarketRegime or None
        Regimes to detect.  Defaults to all four canonical regimes.

    Example
    -------
    >>> det = QuantumRegimeDetector(n_qubits=4)
    >>> returns = np.random.randn(500) * 0.015
    >>> det.fit(returns, window=60)
    >>> label = det.detect(returns[-60:])
    >>> assert isinstance(label, MarketRegime)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        regimes: Optional[List[MarketRegime]] = None,
    ) -> None:
        self.n_qubits = max(n_qubits, 4)
        self.dim = 2 ** self.n_qubits
        self.regimes = regimes or list(MarketRegime)
        self._prototypes: List[_RegimePrototype] = []
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        returns: np.ndarray,
        window: int = 60,
        regime_labels: Optional[np.ndarray] = None,
    ) -> "QuantumRegimeDetector":
        """Fit prototype states from historical returns.

        If *regime_labels* is ``None``, an unsupervised heuristic assigns
        regimes based on rolling statistics thresholds.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``
            Daily (or intraday) return series.
        window : int
            Rolling window length for feature extraction.
        regime_labels : ndarray of shape ``(n_windows,)`` or None
            Optional ground-truth labels.

        Returns
        -------
        self
        """
        returns = np.asarray(returns, dtype=np.float64)
        features = self._extract_rolling_features(returns, window)

        if regime_labels is None:
            regime_labels = self._heuristic_labels(features)

        # Store scaling parameters.
        self._feature_min = features.min(axis=0)
        self._feature_max = features.max(axis=0)
        span = self._feature_max - self._feature_min
        span[span == 0.0] = 1.0
        self._feature_max = self._feature_min + span

        # Build prototype state for each regime as the mean encoded state.
        self._prototypes = []
        for regime in self.regimes:
            mask = regime_labels == regime.value
            if not np.any(mask):
                # Fallback: uniform superposition.
                state = np.ones(self.dim, dtype=np.complex128) / np.sqrt(self.dim)
            else:
                states = np.array(
                    [self._encode(features[i]) for i in range(len(features)) if mask[i]]
                )
                state = states.mean(axis=0)
                norm = np.linalg.norm(state)
                state = state / (norm + 1e-15)

            self._prototypes.append(_RegimePrototype(regime=regime, state=state))

        return self

    def detect(self, returns_window: np.ndarray) -> MarketRegime:
        """Classify a single window of returns.

        Parameters
        ----------
        returns_window : ndarray of shape ``(window,)``

        Returns
        -------
        regime : MarketRegime
        """
        if not self._prototypes:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        features = self._window_features(returns_window)
        state = self._encode(features)
        return self._classify(state)

    def detect_series(
        self, returns: np.ndarray, window: int = 60
    ) -> List[MarketRegime]:
        """Classify rolling windows across a full series.

        Parameters
        ----------
        returns : ndarray of shape ``(n_periods,)``
        window : int

        Returns
        -------
        regimes : list of MarketRegime, length ``n_periods - window + 1``
        """
        results = []
        for start in range(len(returns) - window + 1):
            results.append(self.detect(returns[start: start + window]))
        return results

    def fidelities(self, returns_window: np.ndarray) -> Dict[MarketRegime, float]:
        """Return fidelity to every prototype for a single window.

        Parameters
        ----------
        returns_window : ndarray of shape ``(window,)``

        Returns
        -------
        fidelity_map : dict mapping MarketRegime to float in ``[0, 1]``.
        """
        features = self._window_features(returns_window)
        state = self._encode(features)
        return {
            p.regime: float(np.abs(np.vdot(p.state, state)) ** 2)
            for p in self._prototypes
        }

    # -- internals ----------------------------------------------------------

    def _extract_rolling_features(
        self, returns: np.ndarray, window: int
    ) -> np.ndarray:
        """Compute rolling [mean, std, skew, kurtosis] over *returns*."""
        n = len(returns) - window + 1
        features = np.empty((n, 4))
        for i in range(n):
            w = returns[i: i + window]
            mu = w.mean()
            sigma = w.std() + 1e-12
            skew = float(np.mean(((w - mu) / sigma) ** 3))
            kurt = float(np.mean(((w - mu) / sigma) ** 4)) - 3.0
            features[i] = [mu, sigma, skew, kurt]
        return features

    @staticmethod
    def _window_features(returns_window: np.ndarray) -> np.ndarray:
        """Features for a single window."""
        w = np.asarray(returns_window, dtype=np.float64)
        mu = w.mean()
        sigma = w.std() + 1e-12
        skew = float(np.mean(((w - mu) / sigma) ** 3))
        kurt = float(np.mean(((w - mu) / sigma) ** 4)) - 3.0
        return np.array([mu, sigma, skew, kurt])

    @staticmethod
    def _heuristic_labels(features: np.ndarray) -> np.ndarray:
        """Assign regime labels heuristically from features."""
        labels = np.empty(len(features), dtype=object)
        for i, (mu, sigma, skew, kurt) in enumerate(features):
            if sigma > np.percentile(features[:, 1], 80):
                labels[i] = MarketRegime.VOLATILE.value
            elif mu > np.percentile(features[:, 0], 60):
                labels[i] = MarketRegime.BULL.value
            elif mu < np.percentile(features[:, 0], 40):
                labels[i] = MarketRegime.BEAR.value
            else:
                labels[i] = MarketRegime.SIDEWAYS.value
        return labels

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """Angle-encode a 4-feature vector into a quantum state."""
        if self._feature_min is not None:
            span = self._feature_max - self._feature_min
            normed = (features - self._feature_min[: len(features)]) / span[: len(features)]
            normed = np.clip(normed, 0.0, 1.0)
        else:
            normed = 1.0 / (1.0 + np.exp(-features))
        angles = normed * np.pi

        # Ry product state.
        state = np.array([1.0 + 0j], dtype=np.complex128)
        for i in range(self.n_qubits):
            theta = angles[i] if i < len(angles) else 0.0
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state

    def _classify(self, state: np.ndarray) -> MarketRegime:
        """Return the regime with highest fidelity."""
        best_regime = self.regimes[0]
        best_fidelity = -1.0
        for proto in self._prototypes:
            fidelity = float(np.abs(np.vdot(proto.state, state)) ** 2)
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_regime = proto.regime
        return best_regime


# ---------------------------------------------------------------------------
# Regime Transition Matrix (quantum channel model)
# ---------------------------------------------------------------------------

class RegimeTransitionMatrix:
    """Model regime transitions as a quantum channel.

    Transition probabilities between ``n`` regimes are stored in an
    ``n x n`` row-stochastic matrix.  This is isomorphic to a classical
    Markov chain, but the class additionally provides *Kraus operator*
    representations, enabling the matrix to be embedded into a full
    quantum channel for use in quantum circuit simulations.

    Parameters
    ----------
    regimes : list of MarketRegime or None
        Defaults to all four canonical regimes.

    Example
    -------
    >>> tm = RegimeTransitionMatrix()
    >>> labels = [MarketRegime.BULL, MarketRegime.BULL, MarketRegime.BEAR]
    >>> tm.fit(labels)
    >>> probs = tm.transition_probs(MarketRegime.BULL)
    >>> assert np.isclose(sum(probs.values()), 1.0)
    """

    def __init__(self, regimes: Optional[List[MarketRegime]] = None) -> None:
        self.regimes = regimes or list(MarketRegime)
        self.n = len(self.regimes)
        self._index = {r: i for i, r in enumerate(self.regimes)}
        self.matrix = np.ones((self.n, self.n)) / self.n  # uniform prior

    def fit(self, regime_sequence: List[MarketRegime]) -> "RegimeTransitionMatrix":
        """Estimate transition probabilities from a sequence of regime labels.

        Parameters
        ----------
        regime_sequence : list of MarketRegime

        Returns
        -------
        self
        """
        counts = np.zeros((self.n, self.n))
        for prev, curr in zip(regime_sequence[:-1], regime_sequence[1:]):
            pi = self._index.get(prev)
            ci = self._index.get(curr)
            if pi is not None and ci is not None:
                counts[pi, ci] += 1

        # Row-normalise with Laplace smoothing.
        row_sums = counts.sum(axis=1, keepdims=True) + self.n * 1e-6
        self.matrix = (counts + 1e-6) / row_sums
        return self

    def transition_probs(self, from_regime: MarketRegime) -> Dict[MarketRegime, float]:
        """Return transition probabilities from a given regime.

        Parameters
        ----------
        from_regime : MarketRegime

        Returns
        -------
        probs : dict mapping MarketRegime to float
        """
        idx = self._index[from_regime]
        return {r: float(self.matrix[idx, j]) for j, r in enumerate(self.regimes)}

    def steady_state(self) -> Dict[MarketRegime, float]:
        """Compute the stationary distribution of the Markov chain.

        Returns
        -------
        distribution : dict mapping MarketRegime to float
        """
        # The stationary distribution is the left eigenvector with eigenvalue 1.
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix.T)
        # Find the eigenvalue closest to 1.
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return {r: float(stationary[i]) for i, r in enumerate(self.regimes)}

    def kraus_operators(self) -> List[np.ndarray]:
        """Return Kraus operators for the quantum channel representation.

        Each transition ``|j> -> |i>`` with probability ``T_{ji}`` yields a
        Kraus operator ``K_ij = sqrt(T_{ji}) |i><j|``.

        Returns
        -------
        kraus : list of ndarray, each of shape ``(n, n)``
        """
        operators = []
        for i in range(self.n):
            for j in range(self.n):
                p = self.matrix[j, i]
                if p > 1e-12:
                    K = np.zeros((self.n, self.n))
                    K[i, j] = np.sqrt(p)
                    operators.append(K)
        return operators

    def n_step_forecast(
        self, current: MarketRegime, steps: int = 5
    ) -> List[Dict[MarketRegime, float]]:
        """Forecast regime probabilities *steps* periods ahead.

        Parameters
        ----------
        current : MarketRegime
        steps : int

        Returns
        -------
        forecasts : list of dicts (one per step)
        """
        state = np.zeros(self.n)
        state[self._index[current]] = 1.0
        forecasts = []
        for _ in range(steps):
            state = state @ self.matrix
            forecasts.append({r: float(state[i]) for i, r in enumerate(self.regimes)})
        return forecasts


# ---------------------------------------------------------------------------
# Volatility Regime Classifier
# ---------------------------------------------------------------------------

class VolatilityRegimeClassifier:
    """Map a volatility term-structure to quantum states for regime ID.

    Rather than using price returns, this classifier works directly on the
    *term structure of implied volatility* (e.g. 1-week, 1-month, 3-month,
    6-month IV).  The term structure shape reveals market expectations:

    - **Contango** (upward-sloping): calm market, expectation of future events.
    - **Backwardation** (downward-sloping): near-term stress, fear.
    - **Flat**: regime uncertainty.
    - **Humped**: event-driven (e.g. earnings, election).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (should match term structure length).
    vol_tenors : list of float or None
        Tenor labels in years (e.g. ``[7/252, 21/252, 63/252, 126/252]``).

    Example
    -------
    >>> clf = VolatilityRegimeClassifier(n_qubits=4)
    >>> term_structure = np.array([0.25, 0.22, 0.20, 0.19])
    >>> regime = clf.classify(term_structure)
    """

    SHAPE_REGIMES = {
        "contango": MarketRegime.SIDEWAYS,
        "backwardation": MarketRegime.VOLATILE,
        "flat": MarketRegime.BULL,
        "humped": MarketRegime.BEAR,
    }

    def __init__(
        self,
        n_qubits: int = 4,
        vol_tenors: Optional[List[float]] = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.vol_tenors = vol_tenors or [7 / 252, 21 / 252, 63 / 252, 126 / 252]
        self._shape_prototypes: Dict[str, np.ndarray] = {}
        self._build_prototypes()

    def classify(self, term_structure: np.ndarray) -> MarketRegime:
        """Classify a single term-structure observation.

        Parameters
        ----------
        term_structure : ndarray of shape ``(n_qubits,)``
            IV values at each tenor.

        Returns
        -------
        regime : MarketRegime
        """
        state = self._encode_term_structure(term_structure)
        best_shape = "flat"
        best_fidelity = -1.0
        for shape, proto in self._shape_prototypes.items():
            fidelity = float(np.abs(np.vdot(proto, state)) ** 2)
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_shape = shape
        return self.SHAPE_REGIMES[best_shape]

    def classify_series(
        self, term_structures: np.ndarray
    ) -> List[MarketRegime]:
        """Classify a time series of term structures.

        Parameters
        ----------
        term_structures : ndarray of shape ``(n_observations, n_qubits)``

        Returns
        -------
        regimes : list of MarketRegime
        """
        return [self.classify(ts) for ts in term_structures]

    def shape_fidelities(
        self, term_structure: np.ndarray
    ) -> Dict[str, float]:
        """Return fidelity to every shape prototype.

        Parameters
        ----------
        term_structure : ndarray of shape ``(n_qubits,)``

        Returns
        -------
        fidelities : dict mapping shape name to float in ``[0, 1]``
        """
        state = self._encode_term_structure(term_structure)
        return {
            shape: float(np.abs(np.vdot(proto, state)) ** 2)
            for shape, proto in self._shape_prototypes.items()
        }

    # -- internals ----------------------------------------------------------

    def _build_prototypes(self) -> None:
        """Build canonical quantum state prototypes for each shape."""
        n = self.n_qubits
        # Contango: monotonically increasing angles.
        contango_angles = np.linspace(0.1, 0.9, n) * np.pi
        # Backwardation: monotonically decreasing.
        backwardation_angles = np.linspace(0.9, 0.1, n) * np.pi
        # Flat: equal angles.
        flat_angles = np.full(n, 0.5) * np.pi
        # Humped: peak in the middle.
        humped_angles = np.array(
            [0.3 + 0.4 * np.exp(-((i - n / 2) ** 2) / max(n / 4, 1)) for i in range(n)]
        ) * np.pi

        for name, angles in [
            ("contango", contango_angles),
            ("backwardation", backwardation_angles),
            ("flat", flat_angles),
            ("humped", humped_angles),
        ]:
            self._shape_prototypes[name] = self._angles_to_state(angles)

    def _angles_to_state(self, angles: np.ndarray) -> np.ndarray:
        """Build a product state from rotation angles."""
        state = np.array([1.0 + 0j], dtype=np.complex128)
        for theta in angles:
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state

    def _encode_term_structure(self, ts: np.ndarray) -> np.ndarray:
        """Angle-encode a term-structure into a quantum state."""
        ts = np.asarray(ts, dtype=np.float64)
        # Normalise to [0, 1] then scale to [0, pi].
        vmin, vmax = ts.min(), ts.max()
        span = vmax - vmin
        if span < 1e-12:
            normed = np.full_like(ts, 0.5)
        else:
            normed = (ts - vmin) / span
        angles = normed * np.pi
        return self._angles_to_state(angles)
