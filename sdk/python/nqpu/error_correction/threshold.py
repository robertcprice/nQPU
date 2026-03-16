"""Threshold estimation for quantum error correcting codes.

Implements Monte Carlo sampling of logical error rates across multiple
code distances and physical error rates, with threshold crossing-point
detection via curve fitting.

The **threshold** is the physical error rate below which increasing the
code distance suppresses logical errors exponentially -- the fundamental
figure of merit for a code + decoder combination.

Known thresholds (for reference / validation):
  - Surface code + MWPM, code-capacity depolarizing: ~10.3%
  - Surface code + MWPM, phenomenological: ~2.9%
  - Surface code + MWPM, circuit-level: ~0.57%
  - Repetition code + majority vote, bit-flip: ~50% (trivial)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np

from .codes import QuantumCode, RepetitionCode, SurfaceCode
from .decoders import Decoder, DecoderBenchmark, benchmark_decoder


# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

@dataclass
class ThresholdDataPoint:
    """Single data point in a threshold sweep.

    Attributes
    ----------
    distance : int
        Code distance.
    physical_error_rate : float
        Physical error probability.
    logical_error_rate : float
        Measured logical error rate.
    num_trials : int
        Number of Monte Carlo samples.
    std_error : float
        Standard error of the logical error rate estimate.
    """

    distance: int
    physical_error_rate: float
    logical_error_rate: float
    num_trials: int
    std_error: float = 0.0


@dataclass
class ThresholdResult:
    """Result of a threshold estimation.

    Attributes
    ----------
    threshold : float
        Estimated threshold error rate (crossing point).
    threshold_error : float
        Estimated uncertainty in the threshold.
    data_points : list of ThresholdDataPoint
        All measured data points.
    fit_params : dict
        Parameters of the fitting model.
    code_name : str
        Name of the code family.
    decoder_name : str
        Name of the decoder.
    """

    threshold: float
    threshold_error: float
    data_points: List[ThresholdDataPoint]
    fit_params: Dict[str, Any] = field(default_factory=dict)
    code_name: str = ""
    decoder_name: str = ""


# ------------------------------------------------------------------ #
# Threshold estimator
# ------------------------------------------------------------------ #

class ThresholdEstimator:
    """Monte Carlo threshold estimator.

    Sweeps physical error rates for multiple code distances, measures
    logical error rates, and estimates the threshold crossing point.

    Parameters
    ----------
    distances : list of int
        Code distances to test (e.g. [3, 5, 7]).
    decoder_cls : type
        Decoder class (must accept a code as first argument).
    code_cls : type or None
        Code class. If None, uses :class:`SurfaceCode`.
    code_kwargs : dict or None
        Extra keyword arguments for the code constructor.
    decoder_kwargs : dict or None
        Extra keyword arguments for the decoder constructor.
    num_trials : int
        Monte Carlo trials per data point (default 1000).
    seed : int or None
        Base random seed (each data point gets a derived seed).
    """

    def __init__(
        self,
        distances: List[int],
        decoder_cls: Type[Decoder],
        code_cls: Optional[Type[QuantumCode]] = None,
        code_kwargs: Optional[Dict[str, Any]] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        num_trials: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.distances = sorted(distances)
        self.decoder_cls = decoder_cls
        self.code_cls = code_cls or SurfaceCode
        self.code_kwargs = code_kwargs or {}
        self.decoder_kwargs = decoder_kwargs or {}
        self.num_trials = num_trials
        self.seed = seed

    def run(
        self,
        noise_type: str = "depolarizing",
        p_range: Tuple[float, float] = (0.01, 0.20),
        num_points: int = 10,
        error_type: str = "depolarizing",
    ) -> ThresholdResult:
        """Run the threshold sweep.

        Parameters
        ----------
        noise_type : str
            Noise model type (currently only affects error_type).
        p_range : tuple of float
            (min_p, max_p) range of physical error rates to sweep.
        num_points : int
            Number of error rate points in the sweep.
        error_type : str
            Error type for benchmarking (``"depolarizing"``, ``"x_only"``,
            ``"z_only"``).

        Returns
        -------
        ThresholdResult
            Threshold estimate with all data points.
        """
        p_values = np.linspace(p_range[0], p_range[1], num_points)
        data_points: List[ThresholdDataPoint] = []

        for d in self.distances:
            code = self._make_code(d)
            decoder = self._make_decoder(code)

            for p_idx, p in enumerate(p_values):
                seed = None
                if self.seed is not None:
                    seed = self.seed + d * 10000 + p_idx

                result = benchmark_decoder(
                    code=code,
                    decoder=decoder,
                    physical_error_rate=p,
                    num_trials=self.num_trials,
                    error_type=error_type,
                    seed=seed,
                )

                # Standard error estimate (binomial)
                ler = result.logical_error_rate
                se = np.sqrt(ler * (1 - ler) / max(self.num_trials, 1))

                dp = ThresholdDataPoint(
                    distance=d,
                    physical_error_rate=p,
                    logical_error_rate=ler,
                    num_trials=self.num_trials,
                    std_error=se,
                )
                data_points.append(dp)

        # Estimate threshold from crossing point
        threshold, threshold_error = self._find_threshold(data_points)

        return ThresholdResult(
            threshold=threshold,
            threshold_error=threshold_error,
            data_points=data_points,
            code_name=self.code_cls.__name__,
            decoder_name=self.decoder_cls.__name__,
        )

    def _make_code(self, distance: int) -> QuantumCode:
        """Construct a code instance for the given distance."""
        if self.code_cls == RepetitionCode:
            return self.code_cls(distance=distance, **self.code_kwargs)
        elif self.code_cls == SurfaceCode:
            return self.code_cls(distance=distance, **self.code_kwargs)
        else:
            # Generic: try distance parameter
            try:
                return self.code_cls(distance=distance, **self.code_kwargs)
            except TypeError:
                return self.code_cls(**self.code_kwargs)

    def _make_decoder(self, code: QuantumCode) -> Decoder:
        """Construct a decoder instance for the given code."""
        return self.decoder_cls(code, **self.decoder_kwargs)

    def _find_threshold(
        self, data_points: List[ThresholdDataPoint]
    ) -> Tuple[float, float]:
        """Find threshold by detecting where curves for different distances cross.

        Uses pairwise interpolation of logical error rate curves to find
        crossing points, then averages.
        """
        # Group by distance
        by_distance: Dict[int, List[ThresholdDataPoint]] = {}
        for dp in data_points:
            by_distance.setdefault(dp.distance, []).append(dp)

        # Sort each distance's data by physical error rate
        for d in by_distance:
            by_distance[d].sort(key=lambda x: x.physical_error_rate)

        distances = sorted(by_distance.keys())
        if len(distances) < 2:
            # Cannot find crossing with only one distance
            return 0.0, 1.0

        crossings = []
        for i in range(len(distances) - 1):
            d1 = distances[i]
            d2 = distances[i + 1]
            pts1 = by_distance[d1]
            pts2 = by_distance[d2]

            # Find where the curves cross by looking at sign changes
            # in (ler_small_d - ler_large_d)
            for j in range(min(len(pts1), len(pts2)) - 1):
                if j >= len(pts1) - 1 or j >= len(pts2) - 1:
                    break
                diff_j = pts1[j].logical_error_rate - pts2[j].logical_error_rate
                diff_j1 = (
                    pts1[j + 1].logical_error_rate
                    - pts2[j + 1].logical_error_rate
                )
                if diff_j * diff_j1 < 0:
                    # Linear interpolation for crossing point
                    p_j = pts1[j].physical_error_rate
                    p_j1 = pts1[j + 1].physical_error_rate
                    # x where diff_j + (diff_j1 - diff_j) * t = 0
                    t = -diff_j / (diff_j1 - diff_j + 1e-30)
                    p_cross = p_j + t * (p_j1 - p_j)
                    crossings.append(p_cross)

        if crossings:
            threshold = float(np.mean(crossings))
            threshold_error = float(np.std(crossings)) if len(crossings) > 1 else 0.01
            return threshold, threshold_error

        # Fallback: no crossing found -- estimate from where curves are closest
        return self._fallback_threshold(by_distance, distances)

    def _fallback_threshold(
        self,
        by_distance: Dict[int, List[ThresholdDataPoint]],
        distances: List[int],
    ) -> Tuple[float, float]:
        """Fallback threshold estimation when no clear crossing is found."""
        d1 = distances[0]
        d2 = distances[-1]
        pts1 = by_distance[d1]
        pts2 = by_distance[d2]

        min_diff = float("inf")
        best_p = 0.0

        for j in range(min(len(pts1), len(pts2))):
            diff = abs(pts1[j].logical_error_rate - pts2[j].logical_error_rate)
            if diff < min_diff:
                min_diff = diff
                best_p = pts1[j].physical_error_rate

        return best_p, 0.05  # large uncertainty for fallback


# ------------------------------------------------------------------ #
# Convenience functions
# ------------------------------------------------------------------ #

def estimate_threshold(
    code_cls: Type[QuantumCode],
    decoder_cls: Type[Decoder],
    distances: Optional[List[int]] = None,
    p_range: Tuple[float, float] = (0.01, 0.20),
    num_points: int = 10,
    num_trials: int = 1000,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> ThresholdResult:
    """One-shot threshold estimation convenience function.

    Parameters
    ----------
    code_cls : type
        Code class (e.g. SurfaceCode).
    decoder_cls : type
        Decoder class (e.g. MWPMDecoder).
    distances : list of int or None
        Code distances to test (default [3, 5, 7]).
    p_range : tuple
        Physical error rate range.
    num_points : int
        Number of sweep points.
    num_trials : int
        Monte Carlo trials per point.
    seed : int or None
        Random seed.
    **kwargs
        Additional keyword arguments passed to ThresholdEstimator.

    Returns
    -------
    ThresholdResult
    """
    if distances is None:
        distances = [3, 5, 7]

    estimator = ThresholdEstimator(
        distances=distances,
        decoder_cls=decoder_cls,
        code_cls=code_cls,
        num_trials=num_trials,
        seed=seed,
        **kwargs,
    )
    return estimator.run(p_range=p_range, num_points=num_points)


def compare_codes(
    code_decoder_pairs: List[Tuple[Type[QuantumCode], Type[Decoder]]],
    distance: int = 5,
    p_values: Optional[List[float]] = None,
    num_trials: int = 1000,
    seed: Optional[int] = None,
) -> List[Tuple[str, str, List[ThresholdDataPoint]]]:
    """Compare multiple code + decoder combinations at a fixed distance.

    Parameters
    ----------
    code_decoder_pairs : list of (code_cls, decoder_cls) tuples
        Code and decoder pairs to compare.
    distance : int
        Code distance to use.
    p_values : list of float or None
        Physical error rates (default: 10 points from 0.01 to 0.20).
    num_trials : int
        Monte Carlo trials per point.
    seed : int or None
        Random seed.

    Returns
    -------
    list of (code_name, decoder_name, data_points) tuples
    """
    if p_values is None:
        p_values = list(np.linspace(0.01, 0.20, 10))

    results = []
    for code_cls, decoder_cls in code_decoder_pairs:
        try:
            code = code_cls(distance=distance)
        except TypeError:
            code = code_cls()

        decoder = decoder_cls(code)
        points = []

        for p_idx, p in enumerate(p_values):
            s = None
            if seed is not None:
                s = seed + p_idx

            bench = benchmark_decoder(
                code=code,
                decoder=decoder,
                physical_error_rate=p,
                num_trials=num_trials,
                seed=s,
            )
            dp = ThresholdDataPoint(
                distance=distance,
                physical_error_rate=p,
                logical_error_rate=bench.logical_error_rate,
                num_trials=num_trials,
            )
            points.append(dp)

        results.append((code_cls.__name__, decoder_cls.__name__, points))

    return results


def plot_threshold_data(result: ThresholdResult) -> Dict[str, Any]:
    """Extract threshold data in a format suitable for plotting.

    Returns a dictionary with keys:
      - ``"distances"``: list of distance values
      - ``"p_values"``: dict mapping distance -> list of p values
      - ``"ler_values"``: dict mapping distance -> list of logical error rates
      - ``"threshold"``: estimated threshold
      - ``"threshold_error"``: uncertainty

    This data can be fed directly into matplotlib or any other plotting library.
    """
    by_distance: Dict[int, Tuple[List[float], List[float]]] = {}
    for dp in result.data_points:
        if dp.distance not in by_distance:
            by_distance[dp.distance] = ([], [])
        by_distance[dp.distance][0].append(dp.physical_error_rate)
        by_distance[dp.distance][1].append(dp.logical_error_rate)

    return {
        "distances": sorted(by_distance.keys()),
        "p_values": {d: ps for d, (ps, _) in by_distance.items()},
        "ler_values": {d: ls for d, (_, ls) in by_distance.items()},
        "threshold": result.threshold,
        "threshold_error": result.threshold_error,
    }
