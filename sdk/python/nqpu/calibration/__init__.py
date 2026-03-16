"""nQPU Hardware Calibration -- vendor-specific and generic device configs.

Parse, compare, and export calibration data from the major quantum hardware
vendors so that nQPU's transpiler, noise-aware compiler, and error mitigation
pipeline can operate with realistic device parameters.

Supported backends:

1. **IBM Quantum** (``ibm``): Transmon superconducting processors.
   Parse ``backend.properties()`` JSON or use Eagle/Heron presets.

2. **Quantinuum** (``quantinuum``): Trapped-ion QCCD H-Series.
   Parse spec dictionaries or use H1-1 / H2-1 presets.

3. **QuEra** (``quera``): Neutral-atom Rydberg-blockade processors.
   Parse capabilities dicts, generate lattice geometries, or use
   the Aquila preset.

4. **Generic** (``generic``): Hardware-agnostic representation with
   auto-detection, ideal calibrations for testing, and multi-device
   comparison.

5. **Exporters** (``exporters``): Reports, diffs, JSON / CSV export.

Example
-------
>>> from nqpu.calibration import ibm_eagle_r3, h1_1, aquila, compare_devices
>>> from nqpu.calibration import GenericCalibration
>>> eagle = ibm_eagle_r3()
>>> cal = GenericCalibration.from_transmon(eagle)
>>> print(cal.summary())
"""

from __future__ import annotations

# -- IBM Quantum (superconducting transmon) ---------------------------------
from .ibm import (
    TransmonGate,
    TransmonProcessor,
    TransmonQubit,
    ibm_eagle_r3,
    ibm_heron_r2,
    parse_ibm_properties,
    parse_ibm_v2,
)

# -- Quantinuum (trapped-ion QCCD) -----------------------------------------
from .quantinuum import (
    TrapConfig,
    TrapZone,
    h1_1,
    h2_1,
    parse_quantinuum_specs,
)

# -- QuEra (neutral atom) --------------------------------------------------
from .quera import (
    AtomSite,
    NeutralAtomConfig,
    aquila,
    generate_grid,
    generate_kagome,
    generate_triangular,
    parse_quera_capabilities,
)

# -- Generic / hardware-agnostic -------------------------------------------
from .generic import (
    GenericCalibration,
    auto_detect_format,
    compare_devices,
    ideal_calibration,
    load_calibration,
)

# -- Exporters --------------------------------------------------------------
from .exporters import (
    CalibrationDiff,
    CalibrationReport,
    from_json,
    to_csv,
    to_json,
)

__all__ = [
    # IBM
    "TransmonQubit",
    "TransmonGate",
    "TransmonProcessor",
    "parse_ibm_properties",
    "parse_ibm_v2",
    "ibm_eagle_r3",
    "ibm_heron_r2",
    # Quantinuum
    "TrapZone",
    "TrapConfig",
    "parse_quantinuum_specs",
    "h1_1",
    "h2_1",
    # QuEra
    "AtomSite",
    "NeutralAtomConfig",
    "parse_quera_capabilities",
    "aquila",
    "generate_grid",
    "generate_kagome",
    "generate_triangular",
    # Generic
    "GenericCalibration",
    "auto_detect_format",
    "load_calibration",
    "ideal_calibration",
    "compare_devices",
    # Exporters
    "CalibrationReport",
    "CalibrationDiff",
    "to_json",
    "from_json",
    "to_csv",
]
