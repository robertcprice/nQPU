"""nQPU Quantum Error Mitigation -- noise-aware expectation value correction.

Implements five complementary error mitigation techniques that require no
additional qubits (unlike full quantum error correction) and work with
near-term noisy quantum hardware:

  - **zne**: Zero Noise Extrapolation -- amplify noise via gate folding,
    fit a model, extrapolate to zero noise.
  - **pec**: Probabilistic Error Cancellation -- decompose the inverse noise
    channel into a quasi-probability distribution, sample corrected circuits.
  - **twirling**: Pauli Twirling / Randomized Compiling -- convert coherent
    noise into stochastic Pauli noise for easier mitigation.
  - **readout**: Measurement Error Mitigation -- characterize and invert
    readout confusion matrices.
  - **cdr**: Clifford Data Regression -- learn a noise correction model from
    classically-simulable Clifford circuits.

All modules are pure numpy with no external dependencies.  Circuits are
represented as lists of (gate_name, qubits, params) tuples for framework
independence.

Example:
    from nqpu.mitigation import mitigate, ZNEEstimator, ExtrapolationMethod

    # Quick one-shot mitigation
    result = mitigate(circuit, executor, method="zne")

    # Fine-grained control
    estimator = ZNEEstimator(
        noise_factors=[1, 3, 5, 7],
        method=ExtrapolationMethod.POLYNOMIAL,
        poly_degree=3,
    )
    result = estimator.estimate(circuit, executor)
    print(f"Mitigated: {result.estimated_value:.6f}")
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ----- ZNE -----
from .zne import (
    ExtrapolationMethod,
    FoldingStrategy,
    NoiseScaler,
    ZNEEstimator,
    ZNEResult,
    run_zne,
)

# ----- PEC -----
from .pec import (
    ChannelType,
    NoiseChannel,
    PECDecomposition,
    PECEstimator,
    PECOperation,
    PECResult,
    TwoQubitNoiseChannel,
    run_pec,
)

# ----- Twirling -----
from .twirling import (
    PauliFrame,
    PauliTwirler,
    RandomizedCompiling,
    TwirledCircuit,
    twirl_and_average,
)

# ----- Readout -----
from .readout import (
    CorrectionMethod,
    ReadoutCalibration,
    ReadoutCorrector,
    correct_counts,
)

# ----- CDR -----
from .cdr import (
    CDREstimator,
    CDRModel,
    CDRResult,
    CDRTrainingPoint,
    cdr_correct,
    replace_non_clifford,
)

# Gate type alias
Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]


def mitigate(
    circuit: List[Gate],
    executor: Callable[[List[Gate]], float],
    method: str = "zne",
    **kwargs: Any,
) -> Any:
    """Convenience function for one-shot error mitigation.

    Parameters
    ----------
    circuit : list of Gate
        Circuit as list of (gate_name, qubits, params) tuples.
    executor : callable
        Function that takes a circuit and returns an expectation value.
    method : str
        Mitigation method: "zne", "pec", "twirling", "cdr".
        (Readout mitigation requires calibration data and should be
        used directly via :class:`ReadoutCorrector`.)
    **kwargs
        Additional arguments passed to the underlying estimator.

    Returns
    -------
    Depends on method:
        - "zne": :class:`ZNEResult`
        - "pec": :class:`PECResult`
        - "twirling": tuple of (mean, std_error)
        - "cdr": float (corrected value, requires ideal_executor kwarg)
    """
    method_lower = method.lower()

    if method_lower == "zne":
        noise_factors = kwargs.pop("noise_factors", None)
        extrapolation = kwargs.pop(
            "extrapolation_method", ExtrapolationMethod.LINEAR
        )
        poly_degree = kwargs.pop("poly_degree", 2)
        folding = kwargs.pop("folding", FoldingStrategy.LOCAL)
        return run_zne(
            circuit,
            executor,
            noise_factors=noise_factors,
            method=extrapolation,
            poly_degree=poly_degree,
            folding=folding,
        )

    elif method_lower == "pec":
        error_rate = kwargs.pop("error_rate", 0.01)
        num_samples = kwargs.pop("num_samples", 1000)
        seed = kwargs.pop("seed", None)
        return run_pec(
            circuit,
            executor,
            error_rate=error_rate,
            num_samples=num_samples,
            seed=seed,
        )

    elif method_lower == "twirling":
        num_samples = kwargs.pop("num_samples", 100)
        seed = kwargs.pop("seed", None)
        return twirl_and_average(
            circuit, executor, num_samples=num_samples, seed=seed
        )

    elif method_lower == "cdr":
        ideal_executor = kwargs.pop("ideal_executor", None)
        if ideal_executor is None:
            raise ValueError(
                "CDR requires an ideal_executor kwarg for Clifford simulation"
            )
        num_training = kwargs.pop("num_training_circuits", 20)
        degree = kwargs.pop("degree", 1)
        seed = kwargs.pop("seed", None)
        estimator = CDREstimator(
            num_training_circuits=num_training,
            degree=degree,
            seed=seed,
        )
        result = estimator.estimate(circuit, executor, ideal_executor)
        return result

    else:
        raise ValueError(
            f"Unknown mitigation method '{method}'. "
            "Choose from: zne, pec, twirling, cdr"
        )


__all__ = [
    # Top-level convenience
    "mitigate",
    # ZNE
    "ZNEEstimator",
    "ZNEResult",
    "NoiseScaler",
    "FoldingStrategy",
    "ExtrapolationMethod",
    "run_zne",
    # PEC
    "PECEstimator",
    "PECResult",
    "PECDecomposition",
    "PECOperation",
    "NoiseChannel",
    "TwoQubitNoiseChannel",
    "ChannelType",
    "run_pec",
    # Twirling
    "PauliTwirler",
    "PauliFrame",
    "TwirledCircuit",
    "RandomizedCompiling",
    "twirl_and_average",
    # Readout
    "ReadoutCalibration",
    "ReadoutCorrector",
    "CorrectionMethod",
    "correct_counts",
    # CDR
    "CDREstimator",
    "CDRModel",
    "CDRResult",
    "CDRTrainingPoint",
    "cdr_correct",
    "replace_non_clifford",
]
