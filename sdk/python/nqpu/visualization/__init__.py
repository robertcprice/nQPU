"""nQPU Quantum Visualization Toolkit.

Pure-numpy visualization tools for quantum circuits, states, and results.
All ASCII renderers work without matplotlib; optional matplotlib wrappers
degrade gracefully when the library is not installed.

Submodules:
  - circuit_drawing: ASCII quantum circuit diagrams
  - bloch_sphere: Bloch vector extraction and ASCII sphere rendering
  - state_viz: Probability bar charts, Hinton diagrams, entanglement maps
  - plots: Optional matplotlib wrappers with graceful degradation
  - formatters: ASCII tables, progress bars, result formatting

Example:
    from nqpu.visualization import (
        draw_circuit,
        bloch_from_state,
        probability_bar_chart,
        table,
        format_statevector,
    )
"""

from .bloch_sphere import (
    ASCIIBlochSphere,
    BlochVector,
    bloch_from_angles,
    bloch_from_state,
    bloch_trajectory,
)
from .circuit_drawing import (
    CircuitDrawer,
    CircuitGlyph,
    draw_circuit,
    gate_to_ascii,
)
from .formatters import (
    ASCIITable,
    ProgressBar,
    ResultFormatter,
    format_complex,
    format_statevector,
    progress_bar,
    table,
)
from .plots import (
    HAS_MATPLOTLIB,
    QuantumPlotter,
    plot_density_matrix,
    plot_state,
)
from .state_viz import (
    EntanglementMap,
    HintonDiagram,
    ProbabilityDisplay,
    density_matrix_display,
    probability_bar_chart,
    state_table,
)

__all__ = [
    # circuit_drawing
    "CircuitDrawer",
    "CircuitGlyph",
    "draw_circuit",
    "gate_to_ascii",
    # bloch_sphere
    "ASCIIBlochSphere",
    "BlochVector",
    "bloch_from_angles",
    "bloch_from_state",
    "bloch_trajectory",
    # state_viz
    "EntanglementMap",
    "HintonDiagram",
    "ProbabilityDisplay",
    "density_matrix_display",
    "probability_bar_chart",
    "state_table",
    # plots
    "HAS_MATPLOTLIB",
    "QuantumPlotter",
    "plot_density_matrix",
    "plot_state",
    # formatters
    "ASCIITable",
    "ProgressBar",
    "ResultFormatter",
    "format_complex",
    "format_statevector",
    "progress_bar",
    "table",
]
