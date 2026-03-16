"""nQPU Quantum Circuit Learning (QCL) -- practical ML with parameterized quantum circuits.

Provides a batteries-included quantum machine learning package covering the full
pipeline from data encoding through training and prediction.  Five modules:

  - **circuits**: Parameterized quantum circuits with data encoding and ansatz layers.
  - **gradients**: Exact and approximate gradient methods (parameter-shift, natural gradient).
  - **training**: Training loops, objectives, and optimizers for classification/regression.
  - **kernels**: Quantum kernel methods including QSVM and kernel PCA.
  - **expressibility**: Circuit analysis tools (expressibility, barren plateaus, capacity).

Example::

    from nqpu.qcl import CircuitTemplate, AngleEncoding, HardwareEfficientAnsatz
    from nqpu.qcl import QCLTrainer, ClassificationObjective

    circuit = CircuitTemplate(
        encoding=AngleEncoding(n_qubits=4),
        ansatz=HardwareEfficientAnsatz(n_qubits=4, n_layers=3),
    )
    trainer = QCLTrainer(circuit, ClassificationObjective(n_classes=2))
    history = trainer.fit(X_train, y_train, epochs=50, lr=0.1)
    predictions = trainer.predict(X_test)
"""

from .circuits import (
    AmplitudeEncoding,
    AngleEncoding,
    AnsatzCircuit,
    CircuitTemplate,
    DataEncodingCircuit,
    HardwareEfficientAnsatz,
    IQPEncoding,
    ParameterizedCircuit,
    SimplifiedTwoDesign,
    StronglyEntanglingLayers,
    StatevectorSimulator,
)
from .gradients import (
    BarrenPlateauScanner,
    FiniteDifferenceGradient,
    GradientResult,
    NaturalGradient,
    ParameterShiftRule,
    StochasticParameterShift,
)
from .training import (
    ClassificationObjective,
    KernelObjective,
    QCLTrainer,
    RegressionObjective,
    TrainingHistory,
)
from .kernels import (
    ProjectedQuantumKernel,
    QKernelPCA,
    QSVM,
    QuantumKernel,
    TrainableKernel,
    kernel_target_alignment,
)
from .expressibility import (
    BarrenPlateauDetector,
    EffectiveDimension,
    ExpressibilityAnalyzer,
)

# ----- Data Re-uploading -----
from .reuploading import (
    ReuploadingLayer,
    ReuploadingClassifier,
    ReuploadingHistory,
    MultiQubitReuploading,
)

# ----- Architecture Search -----
from .architecture_search import (
    GateSpec,
    CircuitArchitecture,
    FitnessEvaluator,
    FitnessResult,
    EvolutionarySearch,
    BayesianSearch,
    SearchResult,
)

# ----- Barren Plateau Mitigation -----
from .barren_plateau import (
    VarianceMonitor,
    VarianceStatus,
    IdentityBlockInit,
    LayerwiseTraining,
    LayerwiseResult,
    BarrenPlateauAnalyzer,
    AnalysisResult,
    ScalingResult,
)

__all__ = [
    # circuits
    "ParameterizedCircuit",
    "DataEncodingCircuit",
    "AngleEncoding",
    "AmplitudeEncoding",
    "IQPEncoding",
    "AnsatzCircuit",
    "HardwareEfficientAnsatz",
    "StronglyEntanglingLayers",
    "SimplifiedTwoDesign",
    "CircuitTemplate",
    "StatevectorSimulator",
    # gradients
    "ParameterShiftRule",
    "FiniteDifferenceGradient",
    "StochasticParameterShift",
    "NaturalGradient",
    "GradientResult",
    "BarrenPlateauScanner",
    # training
    "QCLTrainer",
    "ClassificationObjective",
    "RegressionObjective",
    "KernelObjective",
    "TrainingHistory",
    # kernels
    "QuantumKernel",
    "ProjectedQuantumKernel",
    "TrainableKernel",
    "QSVM",
    "QKernelPCA",
    "kernel_target_alignment",
    # expressibility
    "ExpressibilityAnalyzer",
    "BarrenPlateauDetector",
    "EffectiveDimension",
    # reuploading
    "ReuploadingLayer",
    "ReuploadingClassifier",
    "ReuploadingHistory",
    "MultiQubitReuploading",
    # architecture search
    "GateSpec",
    "CircuitArchitecture",
    "FitnessEvaluator",
    "FitnessResult",
    "EvolutionarySearch",
    "BayesianSearch",
    "SearchResult",
    # barren plateau mitigation
    "VarianceMonitor",
    "VarianceStatus",
    "IdentityBlockInit",
    "LayerwiseTraining",
    "LayerwiseResult",
    "BarrenPlateauAnalyzer",
    "AnalysisResult",
    "ScalingResult",
]
