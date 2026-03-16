"""nQPU Quantum Random Number Generation (QRNG) Package.

Provides simulation-based quantum random number generators with full
device-independent certification, NIST SP 800-22 statistical testing,
and provable randomness extraction.

Core concepts:
  - Generators: Produce raw random bits from simulated quantum processes
  - Extractors: Post-process raw bits to remove bias and correlations
  - Statistical tests: Validate output quality against NIST SP 800-22
  - Certification: Device-independent randomness proofs via Bell violations
  - Protocols: High-level pipelines combining the above

Example:
    from nqpu.qrng import random_bits, random_uniform, RandomnessReport

    # Quick generation
    bits = random_bits(1024, protocol='measurement')
    floats = random_uniform(100, low=0.0, high=1.0)

    # Full NIST validation
    report = RandomnessReport(bits)
    report.run_all()
    print(report.summary())
"""

from .generators import (
    MeasurementQRNG,
    VacuumFluctuationQRNG,
    EntanglementQRNG,
    QuantumDiceRoll,
)
from .extractors import (
    VonNeumannExtractor,
    ToeplitzExtractor,
    TrevisanExtractor,
    XORExtractor,
    MinEntropyEstimator,
)
from .statistical_tests import (
    frequency_test,
    block_frequency_test,
    runs_test,
    longest_run_test,
    serial_test,
    approximate_entropy_test,
    cumulative_sums_test,
    dft_spectral_test,
    RandomnessReport,
    StatisticalTestResult,
)
from .certification import (
    CHSHCertifier,
    EntropyAccumulation,
    RandomnessExpansion,
)
from .protocols import (
    DIQRNG,
    SemiDIQRNG,
    RandomBeacon,
    random_bits,
    random_uniform,
    random_integers,
)

__all__ = [
    # Generators
    "MeasurementQRNG",
    "VacuumFluctuationQRNG",
    "EntanglementQRNG",
    "QuantumDiceRoll",
    # Extractors
    "VonNeumannExtractor",
    "ToeplitzExtractor",
    "TrevisanExtractor",
    "XORExtractor",
    "MinEntropyEstimator",
    # Statistical tests
    "frequency_test",
    "block_frequency_test",
    "runs_test",
    "longest_run_test",
    "serial_test",
    "approximate_entropy_test",
    "cumulative_sums_test",
    "dft_spectral_test",
    "RandomnessReport",
    "StatisticalTestResult",
    # Certification
    "CHSHCertifier",
    "EntropyAccumulation",
    "RandomnessExpansion",
    # Protocols
    "DIQRNG",
    "SemiDIQRNG",
    "RandomBeacon",
    "random_bits",
    "random_uniform",
    "random_integers",
]
