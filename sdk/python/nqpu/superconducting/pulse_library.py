"""Pre-optimized GRAPE pulse library with caching and hardware presets.

Provides a persistent cache of GRAPE-optimized microwave pulse shapes for
standard quantum gates at common hardware configurations.  This eliminates
the need to re-run expensive GRAPE optimization each time a gate is needed,
enabling instant pulse lookup for calibrated hardware presets.

Supported hardware presets:
    - ibm_heron:     5.0 GHz, -340 MHz anharmonicity (IBM tunable-coupler)
    - google_willow: 6.0 GHz, -210 MHz anharmonicity (Google surface code)
    - quantinuum_h1: 12.6 GHz, -200 MHz anharmonicity (ion-equivalent params)

Supported gates:  X, H, SX (sqrt-X), T (pi/8 phase)

Supported durations: fast (15 ns), standard (25 ns), slow (40 ns)

Cache format: JSON with pulse amplitudes, fidelity metrics, and metadata.

Example:
    >>> from nqpu.superconducting.pulse_library import PulseLibrary
    >>> lib = PulseLibrary()
    >>> result = lib.optimize_and_cache("X", "ibm_heron", 25.0)
    >>> print(f"Fidelity: {result.fidelity:.6f}")
    >>> lib.save("/tmp/pulse_cache.json")

References:
    - Khaneja et al., J. Magn. Reson. 172, 296 (2005) [GRAPE]
    - Motzoi et al., PRL 103, 110501 (2009) [DRAG / transmon pulses]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .grape import GrapeOptimizer, GrapeResult
from .qubit import TransmonQubit


# ---------------------------------------------------------------------------
# Hardware presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwarePreset:
    """Physical parameters defining a hardware configuration for GRAPE.

    Attributes
    ----------
    name : str
        Human-readable preset identifier.
    frequency_ghz : float
        Qubit frequency in GHz.
    anharmonicity_mhz : float
        Transmon anharmonicity in MHz (negative).
    max_amplitude_ghz : float
        Maximum drive amplitude in GHz.
    """

    name: str
    frequency_ghz: float
    anharmonicity_mhz: float
    max_amplitude_ghz: float


HARDWARE_PRESETS: dict[str, HardwarePreset] = {
    "ibm_heron": HardwarePreset(
        name="ibm_heron",
        frequency_ghz=5.0,
        anharmonicity_mhz=-340.0,
        max_amplitude_ghz=0.15,
    ),
    "google_willow": HardwarePreset(
        name="google_willow",
        frequency_ghz=6.0,
        anharmonicity_mhz=-210.0,
        max_amplitude_ghz=0.15,
    ),
    "quantinuum_h1": HardwarePreset(
        name="quantinuum_h1",
        frequency_ghz=12.6,
        anharmonicity_mhz=-200.0,
        max_amplitude_ghz=0.20,
    ),
}


# ---------------------------------------------------------------------------
# Standard gate target unitaries
# ---------------------------------------------------------------------------

_GATE_UNITARIES: dict[str, np.ndarray] = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "H": np.array(
        [[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128
    ) / math.sqrt(2.0),
    "SX": np.array(
        [[1.0 + 1j, 1.0 - 1j], [1.0 - 1j, 1.0 + 1j]], dtype=np.complex128
    ) / 2.0,
    "T": np.array(
        [[1.0, 0.0], [0.0, np.exp(1j * math.pi / 4.0)]], dtype=np.complex128
    ),
}

SUPPORTED_GATES = list(_GATE_UNITARIES.keys())

DURATION_PRESETS: dict[str, float] = {
    "fast": 15.0,
    "standard": 25.0,
    "slow": 40.0,
}


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(gate_name: str, hardware_preset: str, duration_ns: float) -> str:
    """Generate a deterministic cache key from gate, hardware, and duration."""
    return f"{gate_name}|{hardware_preset}|{duration_ns:.1f}"


# ---------------------------------------------------------------------------
# PulseLibrary
# ---------------------------------------------------------------------------

class PulseLibrary:
    """Pre-optimized GRAPE pulse cache with hardware-aware lookup.

    Maintains an in-memory dictionary of GRAPE optimization results,
    indexed by (gate, hardware_preset, duration).  Supports JSON
    serialization for persistent storage.

    Parameters
    ----------
    seed : int
        Random seed for reproducible GRAPE optimizations.
    num_slices : int
        Number of piecewise-constant time slices per pulse.
    max_iterations : int
        Maximum GRAPE iterations per optimization.
    convergence_threshold : float
        Fidelity threshold for early stopping.

    Examples
    --------
    >>> lib = PulseLibrary()
    >>> result = lib.optimize_and_cache("X", "ibm_heron", 25.0)
    >>> cached = lib.get_cached("X", "ibm_heron")
    >>> assert cached is not None
    """

    def __init__(
        self,
        seed: int = 42,
        num_slices: int = 40,
        max_iterations: int = 300,
        convergence_threshold: float = 0.9999,
    ) -> None:
        self.seed = seed
        self.num_slices = num_slices
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def optimize_and_cache(
        self,
        gate_name: str,
        hardware_preset: str,
        duration_ns: float,
        verbose: bool = False,
    ) -> GrapeResult:
        """Run GRAPE optimization for a gate/hardware/duration and cache the result.

        If the result is already cached, returns it immediately without
        re-optimizing.

        Parameters
        ----------
        gate_name : str
            Gate name: ``"X"``, ``"H"``, ``"SX"``, or ``"T"``.
        hardware_preset : str
            Hardware preset: ``"ibm_heron"``, ``"google_willow"``, or
            ``"quantinuum_h1"``.
        duration_ns : float
            Pulse duration in nanoseconds.
        verbose : bool
            Print GRAPE progress.

        Returns
        -------
        GrapeResult
            The optimized pulse with fidelity metrics.

        Raises
        ------
        ValueError
            If gate_name or hardware_preset is not recognized.
        """
        gate_name = gate_name.upper()
        if gate_name not in _GATE_UNITARIES:
            raise ValueError(
                f"Unknown gate '{gate_name}'. Supported: {SUPPORTED_GATES}"
            )
        if hardware_preset not in HARDWARE_PRESETS:
            raise ValueError(
                f"Unknown hardware preset '{hardware_preset}'. "
                f"Supported: {list(HARDWARE_PRESETS.keys())}"
            )

        key = _cache_key(gate_name, hardware_preset, duration_ns)

        # Return cached if available
        if key in self._cache:
            return self._result_from_cache_entry(self._cache[key])

        # Run GRAPE optimization
        hw = HARDWARE_PRESETS[hardware_preset]
        qubit = TransmonQubit(
            frequency_ghz=hw.frequency_ghz,
            anharmonicity_mhz=hw.anharmonicity_mhz,
        )
        optimizer = GrapeOptimizer(
            qubit,
            max_amplitude_ghz=hw.max_amplitude_ghz,
            lambda_leakage=5.0,
            lambda_smoothness=0.01,
        )

        target_unitary = _GATE_UNITARIES[gate_name]
        result = optimizer.optimize(
            target_unitary,
            duration_ns=duration_ns,
            num_slices=self.num_slices,
            max_iterations=self.max_iterations,
            convergence_threshold=self.convergence_threshold,
            seed=self.seed,
            step_size=0.005,
            momentum=0.9,
            verbose=verbose,
        )

        # Store in cache
        entry = {
            "gate": gate_name,
            "hardware": hardware_preset,
            "duration_ns": duration_ns,
            "fidelity": result.fidelity,
            "leakage": result.leakage,
            "iterations": result.num_iterations,
            "converged": result.converged,
            "amplitudes_i": result.optimized_amplitudes_I.tolist(),
            "amplitudes_q": result.optimized_amplitudes_Q.tolist(),
            "anharmonicity_mhz": hw.anharmonicity_mhz,
            "frequency_ghz": hw.frequency_ghz,
            "dt_ns": result.dt_ns,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._cache[key] = entry

        return result

    def get_cached(
        self,
        gate_name: str,
        hardware_preset: str,
        duration_ns: float = 25.0,
    ) -> Optional[GrapeResult]:
        """Retrieve a cached GRAPE result without re-optimizing.

        Parameters
        ----------
        gate_name : str
            Gate name.
        hardware_preset : str
            Hardware preset name.
        duration_ns : float
            Pulse duration in nanoseconds.

        Returns
        -------
        GrapeResult or None
            Cached result, or ``None`` if not in cache.
        """
        key = _cache_key(gate_name.upper(), hardware_preset, duration_ns)
        entry = self._cache.get(key)
        if entry is None:
            return None
        return self._result_from_cache_entry(entry)

    def list_cached(self) -> list[dict]:
        """List all cached entries with metadata.

        Returns
        -------
        list[dict]
            Each dict contains: gate, hardware, duration_ns, fidelity,
            leakage, iterations, converged, timestamp.
        """
        entries = []
        for entry in self._cache.values():
            entries.append({
                "gate": entry["gate"],
                "hardware": entry["hardware"],
                "duration_ns": entry["duration_ns"],
                "fidelity": entry["fidelity"],
                "leakage": entry["leakage"],
                "iterations": entry["iterations"],
                "converged": entry["converged"],
                "timestamp": entry.get("timestamp", ""),
            })
        return entries

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the pulse cache to a JSON file.

        Parameters
        ----------
        path : str
            File path for the JSON output.
        """
        with open(path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def load(self, path: str) -> None:
        """Load a pulse cache from a JSON file.

        Merges loaded entries with any existing cache entries.  If a key
        exists in both, the loaded version overwrites.

        Parameters
        ----------
        path : str
            File path to a JSON pulse cache.
        """
        with open(path, "r") as f:
            loaded = json.load(f)
        self._cache.update(loaded)

    # ------------------------------------------------------------------
    # Batch optimization
    # ------------------------------------------------------------------

    def populate_standard_library(self, verbose: bool = False) -> list[dict]:
        """Optimize and cache all standard gate/hardware/duration combinations.

        Iterates over all supported gates, hardware presets, and duration
        presets, running GRAPE for any combination not already cached.

        Parameters
        ----------
        verbose : bool
            Print per-optimization progress.

        Returns
        -------
        list[dict]
            Summary of all cached entries after population.
        """
        for gate in SUPPORTED_GATES:
            for hw_name in HARDWARE_PRESETS:
                for dur_name, dur_ns in DURATION_PRESETS.items():
                    if verbose:
                        print(
                            f"  Optimizing {gate} on {hw_name} "
                            f"({dur_name}, {dur_ns:.0f} ns) ...",
                            end=" ",
                        )
                    result = self.optimize_and_cache(
                        gate, hw_name, dur_ns, verbose=False
                    )
                    if verbose:
                        print(
                            f"F={result.fidelity:.6f} "
                            f"leak={result.leakage:.6f} "
                            f"iter={result.num_iterations}"
                        )
        return self.list_cached()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_from_cache_entry(entry: dict) -> GrapeResult:
        """Reconstruct a GrapeResult from a cache dictionary."""
        return GrapeResult(
            optimized_amplitudes_I=np.array(entry["amplitudes_i"]),
            optimized_amplitudes_Q=np.array(entry["amplitudes_q"]),
            fidelity=entry["fidelity"],
            leakage=entry["leakage"],
            num_iterations=entry["iterations"],
            converged=entry.get("converged", True),
            fidelity_history=[],
            cost_history=[],
            duration_ns=entry["duration_ns"],
            dt_ns=entry.get("dt_ns", entry["duration_ns"] / len(entry["amplitudes_i"])),
        )


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 72)
    print("PULSE LIBRARY -- SELF-TEST SUITE")
    print("=" * 72)

    passed = 0
    failed = 0

    def _check(label: str, condition: bool, detail: str = "") -> None:
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS: {label}")
        else:
            failed += 1
            msg = f"  FAIL: {label}"
            if detail:
                msg += f" -- {detail}"
            print(msg)

    # ---- Test 1: Single gate optimization and caching ----
    print("\n--- Test 1: Optimize and cache ---")
    lib = PulseLibrary(seed=42, num_slices=30, max_iterations=150)

    result = lib.optimize_and_cache("X", "ibm_heron", 25.0)
    _check("X gate optimized", result.fidelity > 0.95, f"F={result.fidelity:.6f}")
    _check("Leakage bounded", result.leakage < 0.05, f"leak={result.leakage:.6f}")
    _check(
        "Amplitudes have correct length",
        len(result.optimized_amplitudes_I) == 30,
    )

    # ---- Test 2: Cache lookup ----
    print("\n--- Test 2: Cache lookup ---")
    cached = lib.get_cached("X", "ibm_heron", 25.0)
    _check("Cache hit returns result", cached is not None)
    if cached is not None:
        _check(
            "Cache fidelity matches",
            abs(cached.fidelity - result.fidelity) < 1e-10,
        )

    miss = lib.get_cached("X", "google_willow", 25.0)
    _check("Cache miss returns None", miss is None)

    # ---- Test 3: List cached ----
    print("\n--- Test 3: List cached entries ---")
    entries = lib.list_cached()
    _check("One cached entry", len(entries) == 1)
    if entries:
        _check("Entry has gate field", entries[0]["gate"] == "X")
        _check("Entry has hardware field", entries[0]["hardware"] == "ibm_heron")

    # ---- Test 4: JSON save/load round-trip ----
    print("\n--- Test 4: JSON save/load ---")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        lib.save(tmp_path)
        _check("Save succeeded", os.path.exists(tmp_path))

        lib2 = PulseLibrary()
        lib2.load(tmp_path)
        loaded = lib2.get_cached("X", "ibm_heron", 25.0)
        _check("Load recovers cached result", loaded is not None)
        if loaded is not None:
            _check(
                "Loaded fidelity matches",
                abs(loaded.fidelity - result.fidelity) < 1e-10,
            )
            _check(
                "Loaded amplitudes match",
                np.allclose(
                    loaded.optimized_amplitudes_I,
                    result.optimized_amplitudes_I,
                ),
            )
    finally:
        os.unlink(tmp_path)

    # ---- Test 5: Multiple gates ----
    print("\n--- Test 5: Multiple gates ---")
    for gate in ["H", "SX", "T"]:
        r = lib.optimize_and_cache(gate, "ibm_heron", 25.0)
        _check(
            f"{gate} gate optimized (F={r.fidelity:.4f})",
            r.fidelity > 0.90,
        )

    _check("4 entries cached", len(lib.list_cached()) == 4)

    # ---- Test 6: Different hardware presets ----
    print("\n--- Test 6: Hardware presets ---")
    for hw in ["google_willow", "quantinuum_h1"]:
        r = lib.optimize_and_cache("X", hw, 25.0)
        _check(
            f"X on {hw} (F={r.fidelity:.4f})",
            r.fidelity > 0.90,
        )

    # ---- Test 7: Input validation ----
    print("\n--- Test 7: Input validation ---")
    try:
        lib.optimize_and_cache("INVALID_GATE", "ibm_heron", 25.0)
        _check("Rejects unknown gate", False)
    except ValueError:
        _check("Rejects unknown gate", True)

    try:
        lib.optimize_and_cache("X", "unknown_hardware", 25.0)
        _check("Rejects unknown hardware", False)
    except ValueError:
        _check("Rejects unknown hardware", True)

    # ---- Test 8: Different durations ----
    print("\n--- Test 8: Duration presets ---")
    for dur_label, dur_ns in DURATION_PRESETS.items():
        r = lib.optimize_and_cache("X", "ibm_heron", dur_ns)
        _check(
            f"X {dur_label} ({dur_ns:.0f}ns) F={r.fidelity:.4f}",
            r.fidelity > 0.85,
        )

    # ---- Summary ----
    print("\n" + "=" * 72)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All pulse library tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 72)
