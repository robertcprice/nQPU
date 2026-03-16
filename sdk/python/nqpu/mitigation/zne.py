"""Zero Noise Extrapolation (ZNE) for quantum error mitigation.

Implements noise amplification via gate folding followed by extrapolation
to the zero-noise limit.  The core idea is that if you can controllably
*increase* the noise in a circuit, you can measure the observable at
several noise levels and fit a model to extrapolate what the result would
be at zero noise.

Noise amplification strategies:
  - **Global gate folding**: Replace the full circuit U with U (U^dag U)^n,
    amplifying noise uniformly across all gates.
  - **Local gate folding**: Replace each gate g with g (g^dag g)^n,
    amplifying noise per-gate for finer control.

Extrapolation models:
  - **Linear**: Least-squares fit of y = a*x + b, extrapolate b.
  - **Polynomial**: Fit y = sum(c_k * x^k), extrapolate c_0.
  - **Exponential**: Fit y = a * exp(b*x), extrapolate a.
  - **Richardson**: Polynomial interpolation at x=0 using Lagrange weights.

References:
    - Li & Benjamin, PRX 7, 021050 (2017)
    - Temme, Bravyi, Gambetta, PRL 119, 180509 (2017)
    - Giurgica-Tiron et al., IEEE ISIT (2020)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

# =====================================================================
# Circuit representation (lightweight, framework-agnostic)
# =====================================================================

# A gate is a tuple: (name: str, qubits: tuple[int, ...], params: tuple[float, ...])
Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]


def _gate_inverse(gate: Gate) -> Gate:
    """Return the inverse (dagger) of a gate.

    For rotation gates Rx, Ry, Rz the inverse negates the angle.
    For self-inverse gates (H, X, Y, Z, CNOT, CZ) the inverse is
    the gate itself.  For S and T we return Sdg / Tdg.
    """
    name, qubits, params = gate
    upper = name.upper()

    # Rotation gates: negate angle
    if upper in ("RX", "RY", "RZ", "P", "PHASE", "U1"):
        return (name, qubits, tuple(-p for p in params))

    # Self-inverse gates
    if upper in ("H", "X", "Y", "Z", "CNOT", "CX", "CZ", "SWAP"):
        return gate

    # S / T adjoint
    if upper == "S":
        return ("Sdg", qubits, params)
    if upper == "SDG":
        return ("S", qubits, params)
    if upper == "T":
        return ("Tdg", qubits, params)
    if upper == "TDG":
        return ("T", qubits, params)

    # Generic: assume self-inverse as fallback
    return gate


# =====================================================================
# Noise scaling
# =====================================================================


class FoldingStrategy(Enum):
    """Gate folding strategy for noise amplification."""

    GLOBAL = auto()
    LOCAL = auto()


class NoiseScaler:
    """Amplify circuit noise via gate folding.

    Parameters
    ----------
    strategy : FoldingStrategy
        Which folding approach to use.
    """

    def __init__(self, strategy: FoldingStrategy = FoldingStrategy.LOCAL) -> None:
        self.strategy = strategy

    def fold_gates(
        self, circuit: List[Gate], scale_factor: int
    ) -> List[Gate]:
        """Return a noise-amplified version of *circuit*.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit as a list of (name, qubits, params) tuples.
        scale_factor : int
            Noise amplification factor.  Must be an odd positive integer
            (1 = no amplification, 3 = 3x noise, etc.).  Even values are
            rounded up to the next odd integer.

        Returns
        -------
        list of Gate
            Folded circuit with amplified noise but identical ideal unitary.
        """
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        # Round up to nearest odd
        if scale_factor % 2 == 0:
            scale_factor += 1

        if scale_factor == 1:
            return list(circuit)

        if self.strategy == FoldingStrategy.GLOBAL:
            return self._fold_global(circuit, scale_factor)
        return self._fold_local(circuit, scale_factor)

    def pulse_stretch(
        self, circuit: List[Gate], factor: float
    ) -> List[Gate]:
        """Simulate pulse stretching by tagging gates with a stretch factor.

        This is a higher-level abstraction: real pulse stretching requires
        hardware-level access.  Here we attach the stretch factor as
        metadata so that a noise model can interpret it.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit.
        factor : float
            Pulse duration multiplier (>= 1.0).

        Returns
        -------
        list of Gate
            Circuit with stretch metadata appended to each gate's params.
        """
        if factor < 1.0:
            raise ValueError("Pulse stretch factor must be >= 1.0")
        stretched: List[Gate] = []
        for name, qubits, params in circuit:
            # Append the stretch factor as extra metadata
            new_params = params + (factor,)
            stretched.append((name, qubits, new_params))
        return stretched

    # ----- internal helpers -----

    @staticmethod
    def _fold_global(circuit: List[Gate], scale: int) -> List[Gate]:
        """Global folding: U -> U (U^dag U)^((scale-1)/2)."""
        n_pairs = (scale - 1) // 2
        forward = list(circuit)
        backward = [_gate_inverse(g) for g in reversed(circuit)]
        result = list(forward)
        for _ in range(n_pairs):
            result.extend(backward)
            result.extend(forward)
        return result

    @staticmethod
    def _fold_local(circuit: List[Gate], scale: int) -> List[Gate]:
        """Local folding: each g -> g (g^dag g)^((scale-1)/2)."""
        n_pairs = (scale - 1) // 2
        result: List[Gate] = []
        for g in circuit:
            g_dag = _gate_inverse(g)
            result.append(g)
            for _ in range(n_pairs):
                result.append(g_dag)
                result.append(g)
        return result


# =====================================================================
# Extrapolation models
# =====================================================================


class ExtrapolationMethod(Enum):
    """Available extrapolation models."""

    LINEAR = auto()
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()
    RICHARDSON = auto()


def _linear_extrapolate(
    scales: np.ndarray, values: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Least-squares linear fit y = a*x + b, return (b, [b, a])."""
    if len(scales) == 1:
        return float(values[0]), np.array([float(values[0]), 0.0])
    coeffs = np.polyfit(scales, values, 1)  # [a, b]  (highest power first)
    return float(coeffs[1]), coeffs


def _polynomial_extrapolate(
    scales: np.ndarray, values: np.ndarray, degree: int = 2
) -> Tuple[float, np.ndarray]:
    """Polynomial fit of given degree, return (value at x=0, coefficients)."""
    degree = min(degree, len(scales) - 1)
    coeffs = np.polyfit(scales, values, degree)  # highest power first
    # Value at x=0 is the last coefficient
    return float(coeffs[-1]), coeffs


def _exponential_extrapolate(
    scales: np.ndarray, values: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Fit y = a * exp(b * x), return (a, [a, b]).

    If any values are non-positive, a shift is applied before fitting.
    """
    min_val = np.min(values)
    shift = 0.0
    if min_val <= 0:
        shift = 1.0 - min_val

    log_values = np.log(values + shift)
    # Linear fit in log-space: ln(y) = ln(a) + b*x
    if len(scales) < 2:
        a = float(np.exp(log_values[0])) - shift
        return a, np.array([a, 0.0])

    coeffs = np.polyfit(scales, log_values, 1)  # [b, ln(a)]
    a = float(np.exp(coeffs[1])) - shift
    b = float(coeffs[0])
    return a, np.array([a, b])


def _richardson_extrapolate(
    scales: np.ndarray, values: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Richardson extrapolation using Lagrange weights at x=0.

    For n points at distinct scales x_i, the Lagrange interpolation
    weight for x=0 is:

        w_i = prod_{j != i}  (-x_j) / (x_i - x_j)

    The extrapolated value is sum(w_i * y_i).
    """
    n = len(scales)
    if n == 1:
        return float(values[0]), np.array([float(values[0])])

    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                denom = scales[i] - scales[j]
                if abs(denom) < 1e-12:
                    # Degenerate scales -- fall back to linear
                    return _linear_extrapolate(scales, values)
                weights[i] *= (-scales[j]) / denom

    estimate = float(np.dot(weights, values))
    return estimate, weights


# =====================================================================
# ZNE result container
# =====================================================================


@dataclass
class ZNEResult:
    """Result of a zero-noise extrapolation procedure.

    Attributes
    ----------
    estimated_value : float
        Extrapolated expectation value at zero noise.
    fit_parameters : np.ndarray
        Coefficients of the fitted model.
    noise_factors : list of float
        Scale factors at which circuits were executed.
    raw_values : list of float
        Measured values at each noise factor.
    method : ExtrapolationMethod
        Extrapolation method used.
    confidence : float
        Estimated confidence in the extrapolation (R-squared for
        polynomial fits, or a heuristic for others).
    """

    estimated_value: float
    fit_parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_factors: List[float] = field(default_factory=list)
    raw_values: List[float] = field(default_factory=list)
    method: ExtrapolationMethod = ExtrapolationMethod.LINEAR
    confidence: float = 0.0


# =====================================================================
# ZNE estimator
# =====================================================================


class ZNEEstimator:
    """Zero Noise Extrapolation estimator.

    Runs a circuit at multiple noise amplification levels, fits a model,
    and extrapolates to zero noise.

    Parameters
    ----------
    noise_factors : list of int
        Odd scale factors for noise amplification.  Default: [1, 3, 5].
    method : ExtrapolationMethod
        Model to fit through the data points.
    poly_degree : int
        Polynomial degree (only used when method is POLYNOMIAL).
    folding : FoldingStrategy
        Folding approach for noise amplification.
    """

    def __init__(
        self,
        noise_factors: Optional[Sequence[int]] = None,
        method: ExtrapolationMethod = ExtrapolationMethod.LINEAR,
        poly_degree: int = 2,
        folding: FoldingStrategy = FoldingStrategy.LOCAL,
    ) -> None:
        self.noise_factors = list(noise_factors if noise_factors is not None else [1, 3, 5])
        self.method = method
        self.poly_degree = poly_degree
        self.scaler = NoiseScaler(strategy=folding)

        # Validate
        if not self.noise_factors:
            raise ValueError("noise_factors must not be empty")
        for nf in self.noise_factors:
            if nf < 1:
                raise ValueError(f"noise factor must be >= 1, got {nf}")

    def estimate(
        self,
        circuit: List[Gate],
        executor: Callable[[List[Gate]], float],
    ) -> ZNEResult:
        """Run ZNE and return the extrapolated result.

        Parameters
        ----------
        circuit : list of Gate
            The circuit to mitigate.
        executor : callable
            Function that takes a circuit and returns an expectation value.
            This should include any noise the circuit experiences.

        Returns
        -------
        ZNEResult
            Extrapolation result with estimated value and diagnostics.
        """
        raw_values: List[float] = []
        actual_factors: List[float] = []

        for nf in self.noise_factors:
            folded = self.scaler.fold_gates(circuit, nf)
            val = executor(folded)
            raw_values.append(val)
            actual_factors.append(float(nf))

        scales = np.array(actual_factors)
        vals = np.array(raw_values)

        estimated, params = self._extrapolate(scales, vals)
        confidence = self._compute_confidence(scales, vals, estimated, params)

        return ZNEResult(
            estimated_value=estimated,
            fit_parameters=params,
            noise_factors=actual_factors,
            raw_values=raw_values,
            method=self.method,
            confidence=confidence,
        )

    def _extrapolate(
        self, scales: np.ndarray, values: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Dispatch to the chosen extrapolation method."""
        if self.method == ExtrapolationMethod.LINEAR:
            return _linear_extrapolate(scales, values)
        elif self.method == ExtrapolationMethod.POLYNOMIAL:
            return _polynomial_extrapolate(scales, values, self.poly_degree)
        elif self.method == ExtrapolationMethod.EXPONENTIAL:
            return _exponential_extrapolate(scales, values)
        elif self.method == ExtrapolationMethod.RICHARDSON:
            return _richardson_extrapolate(scales, values)
        else:
            raise ValueError(f"Unknown extrapolation method: {self.method}")

    def _compute_confidence(
        self,
        scales: np.ndarray,
        values: np.ndarray,
        estimated: float,
        params: np.ndarray,
    ) -> float:
        """Compute a confidence metric (R-squared for fitted models)."""
        if len(values) <= 1:
            return 1.0

        # For Richardson, confidence is based on leave-one-out stability
        if self.method == ExtrapolationMethod.RICHARDSON:
            if len(scales) < 3:
                return 0.5
            loo_estimates = []
            for i in range(len(scales)):
                s_loo = np.delete(scales, i)
                v_loo = np.delete(values, i)
                est, _ = _richardson_extrapolate(s_loo, v_loo)
                loo_estimates.append(est)
            spread = np.std(loo_estimates)
            # Heuristic: narrower spread -> higher confidence
            return float(max(0.0, 1.0 - spread / (abs(estimated) + 1e-10)))

        # For fitted models, compute R-squared of the fit
        if self.method == ExtrapolationMethod.LINEAR:
            predicted = np.polyval(params, scales)
        elif self.method == ExtrapolationMethod.POLYNOMIAL:
            predicted = np.polyval(params, scales)
        elif self.method == ExtrapolationMethod.EXPONENTIAL:
            a, b = params[0], params[1]
            predicted = a * np.exp(b * scales)
        else:
            return 0.5

        ss_res = float(np.sum((values - predicted) ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        if ss_tot < 1e-15:
            return 1.0
        r_squared = 1.0 - ss_res / ss_tot
        return float(max(0.0, min(1.0, r_squared)))


# =====================================================================
# Convenience function
# =====================================================================


def run_zne(
    circuit: List[Gate],
    executor: Callable[[List[Gate]], float],
    noise_factors: Optional[Sequence[int]] = None,
    method: ExtrapolationMethod = ExtrapolationMethod.LINEAR,
    poly_degree: int = 2,
    folding: FoldingStrategy = FoldingStrategy.LOCAL,
) -> ZNEResult:
    """One-shot ZNE convenience function.

    Parameters
    ----------
    circuit : list of Gate
        Circuit to mitigate.
    executor : callable
        Executes a circuit and returns an expectation value.
    noise_factors : list of int, optional
        Scale factors.  Default [1, 3, 5].
    method : ExtrapolationMethod
        Extrapolation model.
    poly_degree : int
        Degree for polynomial extrapolation.
    folding : FoldingStrategy
        Gate folding strategy.

    Returns
    -------
    ZNEResult
    """
    estimator = ZNEEstimator(
        noise_factors=noise_factors,
        method=method,
        poly_degree=poly_degree,
        folding=folding,
    )
    return estimator.estimate(circuit, executor)
