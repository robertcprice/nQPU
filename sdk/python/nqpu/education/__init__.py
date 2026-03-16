"""Quantum computing education platform.

Interactive tools for learning quantum computing, quantum error correction,
quantum biology, and quantum algorithms through guided exploration and
self-paced exercises.

All modules are pure numpy with no external dependencies.

Modules
-------
circuit_tutorial
    Gate-by-gate quantum circuit builder with explanations.
qec_playground
    Interactive QEC walkthrough and syndrome visualization.
bio_explorer
    Guided quantum biology demonstrations.
exercises
    Self-paced exercises with solution checkers.

Example:
    from nqpu.education import GateTutor, EntanglementTutor

    tutor = GateTutor(n_qubits=2)
    info = tutor.apply("H", [0])
    print(info["description"])

    ent = EntanglementTutor()
    lesson = ent.bell_state_lesson()
    result = lesson.run()
    print(result.summary())
"""

from __future__ import annotations

from .circuit_tutorial import (
    GateTutor,
    EntanglementTutor,
    AlgorithmWalkthrough,
    CircuitLesson,
    LessonStep,
    TutorialResult,
)
from .qec_playground import (
    QECPlayground,
    SyndromeVisualizer,
    DecoderRace,
    BitFlipLesson,
    PhaseFlipLesson,
    ShorCodeLesson,
    SteaneCodeLesson,
    SurfaceCodeLesson,
)
from .bio_explorer import (
    FMOExplorer,
    TunnelingExplorer,
    DNAMutationExplorer,
    NavigationExplorer,
    BioDemo,
    BioDemoResult,
)
from .exercises import (
    Exercise,
    ExerciseSet,
    ExerciseResult,
    gate_exercises,
    entanglement_exercises,
    algorithm_exercises,
    qec_exercises,
    run_exercise,
    check_solution,
)

__all__ = [
    # circuit_tutorial
    "GateTutor",
    "EntanglementTutor",
    "AlgorithmWalkthrough",
    "CircuitLesson",
    "LessonStep",
    "TutorialResult",
    # qec_playground
    "QECPlayground",
    "SyndromeVisualizer",
    "DecoderRace",
    "BitFlipLesson",
    "PhaseFlipLesson",
    "ShorCodeLesson",
    "SteaneCodeLesson",
    "SurfaceCodeLesson",
    # bio_explorer
    "FMOExplorer",
    "TunnelingExplorer",
    "DNAMutationExplorer",
    "NavigationExplorer",
    "BioDemo",
    "BioDemoResult",
    # exercises
    "Exercise",
    "ExerciseSet",
    "ExerciseResult",
    "gate_exercises",
    "entanglement_exercises",
    "algorithm_exercises",
    "qec_exercises",
    "run_exercise",
    "check_solution",
]
