"""nQPU Quantum Error Correction -- stabilizer codes, decoders, and fault tolerance.

Full-stack quantum error correction toolkit implementing the core abstractions
needed for fault-tolerant quantum computation research:

  - **codes**: Stabilizer code families -- repetition, Steane [[7,1,3]],
    Shor [[9,1,3]], surface codes (rotated/unrotated), and color codes.
  - **decoders**: Syndrome decoding algorithms -- lookup table, minimum weight
    perfect matching (MWPM), union-find, and belief propagation (BP).
  - **noise_models**: QEC-specific noise channels -- depolarizing, phenomenological,
    circuit-level, and biased noise for threshold studies.
  - **threshold**: Monte Carlo threshold estimation with curve fitting and code
    comparison utilities.
  - **lattice_surgery**: Logical qubit operations via lattice surgery merge/split,
    magic state distillation, and resource estimation.

All modules are pure numpy with no external dependencies.

Example:
    from nqpu.error_correction import (
        SurfaceCode, MWPMDecoder, DepolarizingNoise, ThresholdEstimator,
    )

    # Create a distance-3 rotated surface code
    code = SurfaceCode(distance=3, rotated=True)
    decoder = MWPMDecoder(code)
    noise = DepolarizingNoise(p=0.05)

    # Single-shot decode
    error = noise.sample_error(code.n)
    syndrome = code.syndrome(error)
    correction = decoder.decode(syndrome)
    success = code.check_correction(error, correction)

    # Threshold estimation
    estimator = ThresholdEstimator(distances=[3, 5, 7], decoder_cls=MWPMDecoder)
    results = estimator.run(noise_type="depolarizing", p_range=(0.01, 0.20), num_points=10)
    print(f"Threshold: {results.threshold:.4f}")
"""

from __future__ import annotations

# ----- Codes -----
from .codes import (
    QuantumCode,
    RepetitionCode,
    SurfaceCode,
    SteaneCode,
    ShorCode,
    ColorCode,
    PauliType,
)

# ----- Decoders -----
from .decoders import (
    Decoder,
    LookupTableDecoder,
    MWPMDecoder,
    UnionFindDecoder,
    BPDecoder,
    DecoderBenchmark,
    benchmark_decoder,
)

# ----- Noise Models -----
from .noise_models import (
    NoiseModel,
    DepolarizingNoise,
    PhenomenologicalNoise,
    CircuitLevelNoise,
    BiasedNoise,
)

# ----- Threshold Estimation -----
from .threshold import (
    ThresholdEstimator,
    ThresholdResult,
    ThresholdDataPoint,
    estimate_threshold,
    compare_codes,
)

# ----- Lattice Surgery -----
from .lattice_surgery import (
    LogicalQubit,
    LatticeSurgery,
    MagicStateDistillation,
    PauliFrame,
    ResourceEstimate,
    estimate_resources,
)

# ----- XZZX Surface Code -----
from .xzzx import (
    XZZXCode,
    BiasedNoiseChannel,
    XZZXDecoder,
    XZZXThresholdStudy,
    XZZXThresholdResult,
)

# ----- Quantum LDPC Codes -----
from .qldpc import (
    ClassicalCode,
    HypergraphProductCode,
    BicycleCode,
    LiftedProductCode,
    BPDecoderQLDPC,
)

# ----- Correlated Decoding -----
from .correlated_decoding import (
    SyndromeHistory,
    SlidingWindowDecoder,
    SpaceTimeMWPM,
    CorrelatedNoiseModel,
    DecodingBenchmark,
    BenchmarkResult,
)

__all__ = [
    # Codes
    "QuantumCode",
    "RepetitionCode",
    "SurfaceCode",
    "SteaneCode",
    "ShorCode",
    "ColorCode",
    "PauliType",
    # Decoders
    "Decoder",
    "LookupTableDecoder",
    "MWPMDecoder",
    "UnionFindDecoder",
    "BPDecoder",
    "DecoderBenchmark",
    "benchmark_decoder",
    # Noise models
    "NoiseModel",
    "DepolarizingNoise",
    "PhenomenologicalNoise",
    "CircuitLevelNoise",
    "BiasedNoise",
    # Threshold
    "ThresholdEstimator",
    "ThresholdResult",
    "ThresholdDataPoint",
    "estimate_threshold",
    "compare_codes",
    # Lattice surgery
    "LogicalQubit",
    "LatticeSurgery",
    "MagicStateDistillation",
    "PauliFrame",
    "ResourceEstimate",
    "estimate_resources",
    # XZZX surface code
    "XZZXCode",
    "BiasedNoiseChannel",
    "XZZXDecoder",
    "XZZXThresholdStudy",
    "XZZXThresholdResult",
    # Quantum LDPC codes
    "ClassicalCode",
    "HypergraphProductCode",
    "BicycleCode",
    "LiftedProductCode",
    "BPDecoderQLDPC",
    # Correlated decoding
    "SyndromeHistory",
    "SlidingWindowDecoder",
    "SpaceTimeMWPM",
    "CorrelatedNoiseModel",
    "DecodingBenchmark",
    "BenchmarkResult",
]
