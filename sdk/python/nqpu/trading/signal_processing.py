"""
Quantum Signal Processing.

Generates trading signals by encoding market data into quantum states and
extracting actionable buy/sell/hold decisions from quantum measurements.
Includes quantum-inspired filtering, momentum via quantum walks, and
mean-reversion via state fidelity with equilibrium references.

The core idea: classical indicators treat price as a scalar trajectory,
while quantum signal processing encodes price *and* volume into a state
vector whose interference patterns can reveal hidden structure that
single-observable analysis misses.

Example
-------
>>> import numpy as np
>>> from nqpu.trading.signal_processing import (
...     QuantumSignalGenerator,
...     QuantumFilter,
...     QuantumMomentum,
...     QuantumMeanReversion,
...     combine_signals,
... )
>>>
>>> prices = np.cumsum(np.random.randn(200)) + 100
>>> volume = np.abs(np.random.randn(200)) * 1e6
>>> gen = QuantumSignalGenerator(n_qubits=4)
>>> signals = gen.generate(prices, volume, window=20)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A single trading signal with metadata.

    Attributes
    ----------
    direction : float
        Signed strength in ``[-1, 1]``.  Positive = bullish, negative =
        bearish, near-zero = hold.
    confidence : float
        Confidence in ``[0, 1]`` derived from quantum measurement entropy.
    label : str
        Human-readable label: ``"buy"``, ``"sell"``, or ``"hold"``.
    """

    direction: float
    confidence: float
    label: str

    @staticmethod
    def from_direction(direction: float, confidence: float) -> "Signal":
        """Create a Signal with automatic label assignment.

        Parameters
        ----------
        direction : float
            Signed strength in ``[-1, 1]``.
        confidence : float
            Confidence in ``[0, 1]``.

        Returns
        -------
        Signal
        """
        if direction > 0.15:
            label = "buy"
        elif direction < -0.15:
            label = "sell"
        else:
            label = "hold"
        return Signal(
            direction=float(np.clip(direction, -1.0, 1.0)),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            label=label,
        )


# ---------------------------------------------------------------------------
# Quantum Signal Generator
# ---------------------------------------------------------------------------

class QuantumSignalGenerator:
    """Generate trading signals from quantum state measurements.

    Pipeline:

    1. Extract rolling features (log-return, volatility, skew, volume ratio)
       from a price/volume window.
    2. Angle-encode features into a ``2**n_qubits``-dimensional state.
    3. Evolve the state through a parameterised Hamiltonian layer.
    4. Measure observables: the *Z-expectation* on each qubit yields a
       signed direction, and the *measurement entropy* yields a confidence
       score.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (>= 4 recommended).
    n_layers : int
        Hamiltonian evolution layers.
    seed : int or None
        Random seed for reproducibility.

    Example
    -------
    >>> gen = QuantumSignalGenerator(n_qubits=4, seed=42)
    >>> prices = np.cumsum(np.random.randn(200)) + 100
    >>> volume = np.abs(np.random.randn(200)) * 1e6
    >>> signals = gen.generate(prices, volume, window=20)
    >>> assert all(isinstance(s, Signal) for s in signals)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        self.n_qubits = max(n_qubits, 4)
        self.dim = 2 ** self.n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self._params = rng.uniform(0, 2 * np.pi, size=(n_layers, self.n_qubits, 2))
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        prices: np.ndarray,
        volume: Optional[np.ndarray] = None,
        window: int = 20,
    ) -> List[Signal]:
        """Generate signals for every rolling window in the series.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        volume : ndarray of shape ``(n_periods,)`` or None
        window : int
            Rolling window size.

        Returns
        -------
        signals : list of Signal, length ``n_periods - window``
        """
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))

        if volume is not None:
            volume = np.asarray(volume, dtype=np.float64)
            vol_mean = volume.mean() + 1e-12
        else:
            vol_mean = 1.0

        # Extract rolling features for scaling calibration.
        all_features = self._extract_all_features(log_returns, volume, vol_mean, window)
        self._feature_min = all_features.min(axis=0)
        self._feature_max = all_features.max(axis=0)
        span = self._feature_max - self._feature_min
        span[span == 0.0] = 1.0
        self._feature_max = self._feature_min + span

        signals: List[Signal] = []
        for feat in all_features:
            state = self._encode(feat)
            evolved = self._evolve(state)
            direction = self._z_expectation(evolved)
            confidence = self._measurement_confidence(evolved)
            signals.append(Signal.from_direction(direction, confidence))

        return signals

    def generate_single(self, features: np.ndarray) -> Signal:
        """Generate a signal from a pre-computed feature vector.

        Parameters
        ----------
        features : ndarray of shape ``(n_features,)``
            At most ``n_qubits`` features.

        Returns
        -------
        Signal
        """
        state = self._encode(features)
        evolved = self._evolve(state)
        direction = self._z_expectation(evolved)
        confidence = self._measurement_confidence(evolved)
        return Signal.from_direction(direction, confidence)

    # -- internals ----------------------------------------------------------

    def _extract_all_features(
        self,
        log_returns: np.ndarray,
        volume: Optional[np.ndarray],
        vol_mean: float,
        window: int,
    ) -> np.ndarray:
        """Extract rolling features over the entire series."""
        n_windows = len(log_returns) - window + 1
        features = np.empty((n_windows, 4))
        for i in range(n_windows):
            w = log_returns[i : i + window]
            mu = w.mean()
            sigma = w.std() + 1e-12
            skew = float(np.mean(((w - mu) / sigma) ** 3))

            if volume is not None:
                # volume index offset: returns start at index 1 relative to prices
                vol_window = volume[i + 1 : i + 1 + window]
                vol_ratio = vol_window.mean() / vol_mean
            else:
                vol_ratio = 1.0

            features[i] = [mu, sigma, skew, vol_ratio]
        return features

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """Angle-encode features into a quantum state."""
        if self._feature_min is not None:
            span = self._feature_max - self._feature_min
            n = min(len(features), len(span))
            normed = (features[:n] - self._feature_min[:n]) / span[:n]
            normed = np.clip(normed, 0.0, 1.0)
        else:
            normed = 1.0 / (1.0 + np.exp(-features))
        angles = normed * np.pi

        state = np.array([1.0 + 0j], dtype=np.complex128)
        for i in range(self.n_qubits):
            theta = angles[i] if i < len(angles) else 0.0
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state

    def _evolve(self, state: np.ndarray) -> np.ndarray:
        """Apply parameterised Ry-Rz rotations with nearest-neighbour CNOTs."""
        for layer_idx in range(self.n_layers):
            for q in range(self.n_qubits):
                rz_angle = self._params[layer_idx, q, 0]
                ry_angle = self._params[layer_idx, q, 1]
                state = _apply_rz(state, self.n_qubits, q, rz_angle)
                state = _apply_ry(state, self.n_qubits, q, ry_angle)
            for q in range(self.n_qubits - 1):
                state = _apply_cnot(state, self.n_qubits, q, q + 1)
        return state

    def _z_expectation(self, state: np.ndarray) -> float:
        """Weighted average Z-expectation across all qubits.

        Returns a value in ``[-1, 1]`` where positive indicates bullish and
        negative indicates bearish.
        """
        probs = np.abs(state) ** 2
        expectation = 0.0
        for q in range(self.n_qubits):
            for i in range(self.dim):
                bit = (i >> (self.n_qubits - 1 - q)) & 1
                z_val = 1.0 - 2.0 * bit  # |0> -> +1, |1> -> -1
                expectation += z_val * probs[i]
        return expectation / self.n_qubits

    def _measurement_confidence(self, state: np.ndarray) -> float:
        """Confidence derived from measurement entropy.

        Low entropy (concentrated probability) means high confidence; high
        entropy (uniform) means low confidence.
        """
        probs = np.abs(state) ** 2
        probs = probs[probs > 1e-15]
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(self.dim)
        return float(1.0 - entropy / max_entropy) if max_entropy > 0 else 1.0


# ---------------------------------------------------------------------------
# Quantum Filter (quantum-inspired Kalman variant)
# ---------------------------------------------------------------------------

class QuantumFilter:
    """Quantum-inspired Kalman filter for financial time series.

    Maintains an internal quantum state that evolves with each new
    observation.  The filter merges the *prediction* (Hamiltonian evolution
    of the prior state) with the *measurement* (encoding of the new
    observation) using a fidelity-weighted superposition, producing a
    de-noised estimate of the underlying signal.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for state representation.
    process_noise : float
        Controls how much the state is perturbed per step (analogous to
        ``Q`` in a Kalman filter).
    measurement_weight : float
        Relative weight of new observations vs. prior prediction in
        ``[0, 1]``.  Higher values make the filter more responsive.

    Example
    -------
    >>> qf = QuantumFilter(n_qubits=3, measurement_weight=0.3)
    >>> noisy = np.sin(np.linspace(0, 4 * np.pi, 200)) + 0.3 * np.random.randn(200)
    >>> filtered = qf.filter(noisy)
    >>> assert len(filtered) == len(noisy)
    """

    def __init__(
        self,
        n_qubits: int = 3,
        process_noise: float = 0.1,
        measurement_weight: float = 0.3,
    ) -> None:
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.process_noise = process_noise
        self.measurement_weight = np.clip(measurement_weight, 0.0, 1.0)
        self._state: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def filter(self, series: np.ndarray) -> np.ndarray:
        """Filter an entire time series.

        Parameters
        ----------
        series : ndarray of shape ``(n_periods,)``

        Returns
        -------
        filtered : ndarray of shape ``(n_periods,)``
        """
        series = np.asarray(series, dtype=np.float64)
        self._state = None
        filtered = np.empty_like(series)
        for t, observation in enumerate(series):
            filtered[t] = self.step(observation)
        return filtered

    def step(self, observation: float) -> float:
        """Process a single observation and return the filtered value.

        Parameters
        ----------
        observation : float

        Returns
        -------
        estimate : float
        """
        obs_state = self._encode_scalar(observation)

        if self._state is None:
            self._state = obs_state.copy()
            return observation

        # Prediction: evolve prior state with small random rotation
        # (simulates process noise / Hamiltonian drift).
        predicted = self._evolve_prior(self._state)

        # Update: superpose prediction and measurement.
        alpha = self.measurement_weight
        combined = (1.0 - alpha) * predicted + alpha * obs_state
        norm = np.linalg.norm(combined)
        if norm > 1e-15:
            combined /= norm
        else:
            combined = obs_state.copy()

        self._state = combined

        # Decode: extract scalar estimate from state.
        return self._decode_scalar(combined)

    def reset(self) -> None:
        """Reset the filter state."""
        self._state = None

    # -- internals ----------------------------------------------------------

    def _encode_scalar(self, value: float) -> np.ndarray:
        """Encode a scalar into a quantum state via sigmoid + angle encoding."""
        normed = 1.0 / (1.0 + np.exp(-value))
        angle = normed * np.pi
        state = np.array([1.0 + 0j], dtype=np.complex128)
        for _ in range(self.n_qubits):
            qubit = np.array(
                [np.cos(angle / 2), np.sin(angle / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state

    def _evolve_prior(self, state: np.ndarray) -> np.ndarray:
        """Apply a small random rotation to simulate process noise."""
        # Apply Ry rotation on each qubit proportional to process noise.
        evolved = state.copy()
        for q in range(self.n_qubits):
            angle = self.process_noise * np.sin(q + 1)
            evolved = _apply_ry(evolved, self.n_qubits, q, angle)
        return evolved

    def _decode_scalar(self, state: np.ndarray) -> float:
        """Extract a scalar from a quantum state.

        Uses the Z-expectation of the first qubit, mapped back through
        the inverse sigmoid.
        """
        probs = np.abs(state) ** 2
        # Z-expectation of qubit 0: sum p_i * (+1 if qubit0=0, -1 if qubit0=1)
        z_exp = 0.0
        for i in range(self.dim):
            bit = (i >> (self.n_qubits - 1)) & 1
            z_exp += (1.0 - 2.0 * bit) * probs[i]
        # Map from [-1, 1] back to original scale via inverse sigmoid.
        p = (z_exp + 1.0) / 2.0
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(-np.log(1.0 / p - 1.0))


# ---------------------------------------------------------------------------
# Quantum Momentum (quantum walk on price graph)
# ---------------------------------------------------------------------------

class QuantumMomentum:
    """Momentum indicator based on quantum walk dynamics.

    Simulates a discrete-time quantum walk on a graph where nodes represent
    discretised price levels and edges represent allowed transitions.
    Unlike a classical random walk (which diffuses as ``sqrt(t)``), a
    quantum walk spreads as ``t``, making it more sensitive to directional
    momentum.

    The momentum signal is extracted from the asymmetry of the walker's
    probability distribution: rightward (upward price) bias indicates
    bullish momentum and vice versa.

    Parameters
    ----------
    n_levels : int
        Number of discretised price levels (must be a power of 2).
    n_steps : int
        Number of quantum walk steps per measurement.

    Example
    -------
    >>> qm = QuantumMomentum(n_levels=16, n_steps=5)
    >>> prices = np.cumsum(np.random.randn(100)) + 100
    >>> momentum = qm.compute(prices, window=20)
    >>> assert len(momentum) == len(prices) - 20 + 1
    """

    def __init__(self, n_levels: int = 16, n_steps: int = 5) -> None:
        # Round up to nearest power of 2.
        n_bits = max(int(np.ceil(np.log2(max(n_levels, 2)))), 1)
        self.n_levels = 2 ** n_bits
        self.n_steps = n_steps
        # Coin operator: Hadamard on the coin qubit.
        self._coin = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)

    # -- public API ---------------------------------------------------------

    def compute(
        self, prices: np.ndarray, window: int = 20
    ) -> np.ndarray:
        """Compute quantum momentum for rolling windows.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        window : int

        Returns
        -------
        momentum : ndarray of shape ``(n_periods - window + 1,)``
            Values in ``[-1, 1]``; positive = bullish momentum.
        """
        prices = np.asarray(prices, dtype=np.float64)
        n_windows = len(prices) - window + 1
        momentum = np.empty(n_windows)
        for i in range(n_windows):
            w = prices[i : i + window]
            momentum[i] = self._walk_momentum(w)
        return momentum

    def compute_single(self, price_window: np.ndarray) -> float:
        """Compute momentum for a single price window.

        Parameters
        ----------
        price_window : ndarray of shape ``(window,)``

        Returns
        -------
        momentum : float in ``[-1, 1]``
        """
        return self._walk_momentum(np.asarray(price_window, dtype=np.float64))

    # -- internals ----------------------------------------------------------

    def _walk_momentum(self, prices: np.ndarray) -> float:
        """Run a quantum walk initialised at the mid-price level."""
        # Discretise price trajectory into levels.
        p_min, p_max = prices.min(), prices.max()
        span = p_max - p_min
        if span < 1e-12:
            return 0.0

        # Map the last price to a starting node.
        start_node = int(
            np.clip(
                (prices[-1] - p_min) / span * (self.n_levels - 1),
                0,
                self.n_levels - 1,
            )
        )

        # State: (position, coin) -> amplitude.  Total dim = n_levels * 2.
        dim = self.n_levels * 2
        state = np.zeros(dim, dtype=np.complex128)
        # Initialise at start_node with coin in |+> state.
        state[start_node * 2 + 0] = 1.0 / np.sqrt(2)
        state[start_node * 2 + 1] = 1.0 / np.sqrt(2)

        # Encode price direction bias into the coin.
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))
        bias = np.tanh(log_returns.mean() / (log_returns.std() + 1e-12))
        biased_coin = self._biased_coin(bias)

        for _ in range(self.n_steps):
            state = self._apply_coin(state, biased_coin)
            state = self._shift(state)

        # Extract position probabilities.
        pos_probs = np.zeros(self.n_levels)
        for node in range(self.n_levels):
            pos_probs[node] = (
                np.abs(state[node * 2]) ** 2 + np.abs(state[node * 2 + 1]) ** 2
            )

        # Momentum = asymmetry around start node.
        mid = start_node
        right_mass = pos_probs[mid + 1 :].sum() if mid < self.n_levels - 1 else 0.0
        left_mass = pos_probs[:mid].sum() if mid > 0 else 0.0
        total = right_mass + left_mass + 1e-15
        return float((right_mass - left_mass) / total)

    def _biased_coin(self, bias: float) -> np.ndarray:
        """Construct a bias-aware coin operator.

        When ``bias > 0``, the coin favours rightward (bullish) transitions;
        when ``bias < 0``, leftward (bearish).
        """
        theta = (np.pi / 4) * (1.0 + np.clip(bias, -1.0, 1.0))
        return np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [np.sin(theta), -np.cos(theta)],
            ],
            dtype=np.complex128,
        )

    def _apply_coin(self, state: np.ndarray, coin: np.ndarray) -> np.ndarray:
        """Apply coin operator to the coin register at every position."""
        new_state = np.zeros_like(state)
        for node in range(self.n_levels):
            idx0 = node * 2
            idx1 = node * 2 + 1
            c = state[idx0:idx1 + 1]  # noqa: this slice is [idx0, idx1]
            amp = np.array([state[idx0], state[idx1]], dtype=np.complex128)
            rotated = coin @ amp
            new_state[idx0] = rotated[0]
            new_state[idx1] = rotated[1]
        return new_state

    def _shift(self, state: np.ndarray) -> np.ndarray:
        """Conditional shift: coin=0 -> move right, coin=1 -> move left."""
        new_state = np.zeros_like(state)
        for node in range(self.n_levels):
            # Coin = 0 component shifts right.
            right = min(node + 1, self.n_levels - 1)
            new_state[right * 2] += state[node * 2]
            # Coin = 1 component shifts left.
            left = max(node - 1, 0)
            new_state[left * 2 + 1] += state[node * 2 + 1]
        return new_state


# ---------------------------------------------------------------------------
# Quantum Mean Reversion
# ---------------------------------------------------------------------------

class QuantumMeanReversion:
    """Mean-reversion signals via quantum state fidelity.

    Computes the fidelity (overlap squared) between the current market
    state and a reference *equilibrium* state learned from historical data.
    High fidelity indicates the market is near equilibrium (no trade);
    low fidelity indicates displacement from equilibrium, generating a
    reversion signal whose direction depends on whether the current state
    is "above" or "below" the equilibrium.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    lookback : int
        Number of observations used to define the equilibrium state.

    Example
    -------
    >>> qmr = QuantumMeanReversion(n_qubits=4, lookback=100)
    >>> prices = np.cumsum(np.random.randn(300)) + 100
    >>> signals = qmr.compute(prices, window=20)
    >>> assert len(signals) > 0
    """

    def __init__(self, n_qubits: int = 4, lookback: int = 100) -> None:
        self.n_qubits = max(n_qubits, 4)
        self.dim = 2 ** self.n_qubits
        self.lookback = lookback
        self._equilibrium_state: Optional[np.ndarray] = None
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def fit(self, prices: np.ndarray, window: int = 20) -> "QuantumMeanReversion":
        """Learn the equilibrium state from historical prices.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        window : int

        Returns
        -------
        self
        """
        features = self._price_features(prices, window)
        n_eq = min(self.lookback, len(features))
        eq_features = features[:n_eq]

        self._feature_min = eq_features.min(axis=0)
        self._feature_max = eq_features.max(axis=0)
        span = self._feature_max - self._feature_min
        span[span == 0.0] = 1.0
        self._feature_max = self._feature_min + span

        # Equilibrium = mean encoded state (normalised).
        states = np.array([self._encode(f) for f in eq_features])
        mean_state = states.mean(axis=0)
        norm = np.linalg.norm(mean_state)
        self._equilibrium_state = mean_state / (norm + 1e-15)
        return self

    def compute(
        self, prices: np.ndarray, window: int = 20
    ) -> List[Signal]:
        """Compute mean-reversion signals over rolling windows.

        Parameters
        ----------
        prices : ndarray of shape ``(n_periods,)``
        window : int

        Returns
        -------
        signals : list of Signal
        """
        if self._equilibrium_state is None:
            self.fit(prices, window)

        features = self._price_features(prices, window)
        signals: List[Signal] = []
        for feat in features:
            state = self._encode(feat)
            fidelity = float(np.abs(np.vdot(self._equilibrium_state, state)) ** 2)

            # Direction: compare mean return to zero.
            direction = -np.sign(feat[0]) * (1.0 - fidelity)
            confidence = 1.0 - fidelity
            signals.append(Signal.from_direction(direction, confidence))
        return signals

    def fidelity(self, features: np.ndarray) -> float:
        """Compute fidelity between a feature vector and the equilibrium.

        Parameters
        ----------
        features : ndarray of shape ``(n_features,)``

        Returns
        -------
        fidelity : float in ``[0, 1]``
        """
        if self._equilibrium_state is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        state = self._encode(features)
        return float(np.abs(np.vdot(self._equilibrium_state, state)) ** 2)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _price_features(prices: np.ndarray, window: int) -> np.ndarray:
        """Extract rolling features from a price series."""
        prices = np.asarray(prices, dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(prices, 1e-12)))
        n_windows = len(log_returns) - window + 1
        features = np.empty((n_windows, 4))
        for i in range(n_windows):
            w = log_returns[i : i + window]
            mu = w.mean()
            sigma = w.std() + 1e-12
            skew = float(np.mean(((w - mu) / sigma) ** 3))
            kurt = float(np.mean(((w - mu) / sigma) ** 4)) - 3.0
            features[i] = [mu, sigma, skew, kurt]
        return features

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """Angle-encode features into a quantum state."""
        if self._feature_min is not None:
            span = self._feature_max - self._feature_min
            n = min(len(features), len(span))
            normed = (features[:n] - self._feature_min[:n]) / span[:n]
            normed = np.clip(normed, 0.0, 1.0)
        else:
            normed = 1.0 / (1.0 + np.exp(-features))
        angles = normed * np.pi

        state = np.array([1.0 + 0j], dtype=np.complex128)
        for i in range(self.n_qubits):
            theta = angles[i] if i < len(angles) else 0.0
            qubit = np.array(
                [np.cos(theta / 2), np.sin(theta / 2)], dtype=np.complex128
            )
            state = np.kron(state, qubit)
        return state


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------

def combine_signals(
    signal_sources: Dict[str, List[Signal]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Signal]:
    """Combine multiple signal sources with configurable weights.

    Each source contributes a direction and confidence; the combined signal
    is a weighted average of directions with a joint confidence derived
    from the individual confidences.

    Parameters
    ----------
    signal_sources : dict mapping source name to list of Signal
        All lists must have the same length.
    weights : dict mapping source name to float or None
        Relative importance weights.  If ``None``, all sources are equally
        weighted.

    Returns
    -------
    combined : list of Signal

    Example
    -------
    >>> from nqpu.trading.signal_processing import Signal, combine_signals
    >>> s1 = [Signal.from_direction(0.5, 0.8), Signal.from_direction(-0.3, 0.6)]
    >>> s2 = [Signal.from_direction(0.2, 0.9), Signal.from_direction(-0.5, 0.7)]
    >>> combined = combine_signals({"momentum": s1, "reversion": s2})
    >>> assert len(combined) == 2
    """
    if not signal_sources:
        return []

    names = list(signal_sources.keys())
    lengths = [len(signal_sources[n]) for n in names]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All signal lists must have the same length, got {lengths}"
        )
    n = lengths[0]

    if weights is None:
        weights = {name: 1.0 for name in names}

    total_weight = sum(weights.get(name, 1.0) for name in names)
    if total_weight < 1e-15:
        total_weight = 1.0

    combined: List[Signal] = []
    for i in range(n):
        direction = 0.0
        confidence = 0.0
        for name in names:
            w = weights.get(name, 1.0) / total_weight
            sig = signal_sources[name][i]
            direction += w * sig.direction
            confidence += w * sig.confidence
        combined.append(Signal.from_direction(direction, confidence))

    return combined


# ---------------------------------------------------------------------------
# Internal gate helpers
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


def _apply_rz(
    state: np.ndarray, n_qubits: int, qubit: int, angle: float
) -> np.ndarray:
    """Apply Rz(angle) to a specific qubit in a state vector."""
    dim = len(state)
    new_state = state.copy()
    for i in range(dim):
        bit = (i >> (n_qubits - 1 - qubit)) & 1
        if bit == 0:
            new_state[i] = state[i] * np.exp(-1j * angle / 2)
        else:
            new_state[i] = state[i] * np.exp(1j * angle / 2)
    return new_state


def _apply_cnot(
    state: np.ndarray, n_qubits: int, control: int, target: int
) -> np.ndarray:
    """Apply CNOT with given control and target qubits."""
    new_state = state.copy()
    t_step = 1 << (n_qubits - 1 - target)
    for i in range(len(state)):
        ctrl_bit = (i >> (n_qubits - 1 - control)) & 1
        tgt_bit = (i >> (n_qubits - 1 - target)) & 1
        if ctrl_bit == 1 and tgt_bit == 0:
            partner = i ^ t_step
            new_state[i], new_state[partner] = state[partner], state[i]
    return new_state
