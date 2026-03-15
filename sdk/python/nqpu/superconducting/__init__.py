"""nQPU Superconducting Transmon Backend -- Physics-based transmon QPU simulation.

Models superconducting transmon quantum processors with realistic noise from
T1/T2 decoherence, ZZ crosstalk, leakage, and calibration drift.

Device presets for IBM Eagle/Heron, Google Sycamore/Willow, Rigetti Ankaa.

Example:
    from nqpu.superconducting import TransmonSimulator, ChipConfig, DevicePresets

    config = DevicePresets.IBM_HERON.build(num_qubits=5)
    sim = TransmonSimulator(config, execution_mode="noisy")
    sim.h(0)
    sim.cnot(0, 1)
    result = sim.measure_all()
"""

from .qubit import TransmonQubit
from .chip import (
    ChipConfig,
    ChipTopology,
    DevicePresets,
    NativeGateFamily,
    TopologyType,
)
from .gates import (
    TransmonGateSet,
    GateInstruction,
    NativeGateType,
)
from .noise import TransmonNoiseModel
from .simulator import TransmonSimulator, CircuitStats, LeakageReductionUnit
from .qcvv import TransmonQCVV, BenchmarkResult, compare_backends
from .digital_twin import (
    CalibrationData,
    CalibrationDrift,
    DigitalTwin,
    StabilityReport,
    ValidationReport,
)
from .compiler_bench import CompilerBenchmark, CompilationResult, NativeGateAnalyzer
from .pulse import (
    PulseShape,
    Pulse,
    PulseSchedule,
    PulseSimulator,
    build_lindblad_operators,
    build_two_qubit_lindblad_operators,
    evolve_density_matrix,
    ReadoutSimulator,
    ReadoutResult,
    DiscriminationResult,
    CRCalibrator,
    CRCalibrationResult,
    thermal_state,
    EchoedCRCalibrator,
    EchoedCRCalibrationResult,
)
from .grape import GrapeOptimizer, GrapeResult
from .unified_sim import (
    UnifiedSimulator,
    FidelityTable,
    ScalingData,
    GateOverheadTable,
    PublicationFigures,
    BackendResult,
)
from .pulse_library import (
    PulseLibrary,
    HardwarePreset,
    HARDWARE_PRESETS,
    SUPPORTED_GATES,
    DURATION_PRESETS,
)
from .noise_fingerprint import NoiseFingerprint, FingerprintComparison
from .qec_drift_study import QECDriftStudy, DriftImpactReport
from .calibration_forecast import CalibrationForecaster, ForecastReport

__all__ = [
    # Qubit
    "TransmonQubit",
    # Chip
    "ChipConfig",
    "ChipTopology",
    "DevicePresets",
    "NativeGateFamily",
    "TopologyType",
    # Gates
    "TransmonGateSet",
    "GateInstruction",
    "NativeGateType",
    # Noise
    "TransmonNoiseModel",
    # Simulator
    "TransmonSimulator",
    "CircuitStats",
    # QCVV
    "TransmonQCVV",
    "BenchmarkResult",
    "compare_backends",
    # Digital Twin
    "CalibrationData",
    "CalibrationDrift",
    "DigitalTwin",
    "StabilityReport",
    "ValidationReport",
    # Compiler Benchmark
    "CompilerBenchmark",
    "CompilationResult",
    "NativeGateAnalyzer",
    # Pulse-level simulation
    "PulseShape",
    "Pulse",
    "PulseSchedule",
    "PulseSimulator",
    # Lindblad master equation
    "build_lindblad_operators",
    "build_two_qubit_lindblad_operators",
    "evolve_density_matrix",
    # Dispersive readout
    "ReadoutSimulator",
    "ReadoutResult",
    "DiscriminationResult",
    # CR gate auto-calibration
    "CRCalibrator",
    "CRCalibrationResult",
    # Echoed CR calibration
    "EchoedCRCalibrator",
    "EchoedCRCalibrationResult",
    # Thermal initial state
    "thermal_state",
    # Leakage reduction
    "LeakageReductionUnit",
    # GRAPE optimal control
    "GrapeOptimizer",
    "GrapeResult",
    # Unified publication simulator
    "UnifiedSimulator",
    "FidelityTable",
    "ScalingData",
    "GateOverheadTable",
    "PublicationFigures",
    "BackendResult",
    # Pulse library (pre-optimized GRAPE cache)
    "PulseLibrary",
    "HardwarePreset",
    "HARDWARE_PRESETS",
    "SUPPORTED_GATES",
    "DURATION_PRESETS",
    # Noise fingerprinting
    "NoiseFingerprint",
    "FingerprintComparison",
    # QEC drift study
    "QECDriftStudy",
    "DriftImpactReport",
    # Calibration forecasting
    "CalibrationForecaster",
    "ForecastReport",
]
