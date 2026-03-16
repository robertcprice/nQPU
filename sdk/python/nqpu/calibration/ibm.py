"""Parse IBM Quantum backend properties into nQPU device configs.

Supports two IBM calibration data formats:

1. **Legacy (v1)**: ``backend.properties().to_dict()`` JSON with per-qubit
   and per-gate lists of ``{name, value, unit}`` parameter dictionaries.
2. **Runtime (v2)**: Qiskit Runtime v2 ``backend.configuration()`` style
   with nested ``qubits`` and ``gates`` objects.

Also ships two preset device configs built from IBM's published
specifications for Eagle r3 (127-qubit) and Heron r2 (156-qubit)
processors so that users can run routing / noise-aware compilation
without needing a live IBM Quantum account.

Example
-------
>>> from nqpu.calibration.ibm import ibm_eagle_r3, ibm_heron_r2
>>> eagle = ibm_eagle_r3()
>>> print(eagle.summary())
>>> best = eagle.best_subgraph(20)

References
----------
- IBM Quantum Systems documentation (2024)
- Qiskit backend properties JSON schema
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TransmonQubit:
    """Single transmon qubit calibration data."""

    index: int
    frequency: float  # GHz
    t1: float  # microseconds
    t2: float  # microseconds
    readout_error: float
    readout_length: float  # ns
    anharmonicity: float = -0.34  # GHz (typical)

    @property
    def t1_ns(self) -> float:
        """T1 in nanoseconds."""
        return self.t1 * 1000.0

    @property
    def coherence_ratio(self) -> float:
        """T2 / T1 ratio (ideally <= 2 for Markovian noise)."""
        return self.t2 / self.t1 if self.t1 > 0 else 0.0


@dataclass
class TransmonGate:
    """Gate calibration for a specific qubit or qubit pair."""

    gate_type: str
    qubits: tuple
    error_rate: float
    gate_length: float  # ns


@dataclass
class TransmonProcessor:
    """Full transmon processor calibration.

    Holds qubit parameters, gate error rates, and the device coupling map.
    Provides utility methods for quality scoring and best-subgraph selection
    so that transpiler / noise-aware routing can pick the most reliable
    qubits for a given circuit width.
    """

    name: str
    n_qubits: int
    qubits: Dict[int, TransmonQubit] = field(default_factory=dict)
    gates: List[TransmonGate] = field(default_factory=list)
    coupling_map: List[Tuple[int, int]] = field(default_factory=list)
    basis_gates: List[str] = field(
        default_factory=lambda: ["cx", "id", "rz", "sx", "x"]
    )
    timestamp: str = ""

    # -- Aggregate properties -----------------------------------------------

    @property
    def median_t1(self) -> float:
        """Median T1 across all qubits in microseconds."""
        vals = [q.t1 for q in self.qubits.values()]
        return float(np.median(vals)) if vals else 0.0

    @property
    def median_t2(self) -> float:
        """Median T2 across all qubits in microseconds."""
        vals = [q.t2 for q in self.qubits.values()]
        return float(np.median(vals)) if vals else 0.0

    @property
    def median_cx_error(self) -> float:
        """Median CX (two-qubit) gate error rate."""
        cx_errors = [
            g.error_rate for g in self.gates
            if g.gate_type in ("cx", "ecr", "cz") and len(g.qubits) == 2
        ]
        return float(np.median(cx_errors)) if cx_errors else 0.0

    @property
    def best_qubits(self) -> List[int]:
        """Qubit indices sorted by quality score (best first)."""
        scores = self.qubit_quality_scores()
        return sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]

    # -- Quality scoring ----------------------------------------------------

    def qubit_quality_scores(self) -> Dict[int, float]:
        """Score each qubit on [0, 1] based on T1, T2, and readout error.

        Scoring formula (equal weight):
            score = (1/3) * t1_norm + (1/3) * t2_norm + (1/3) * (1 - readout_error)

        where ``t1_norm`` and ``t2_norm`` are normalised relative to the
        per-device maximum T1 / T2 respectively.
        """
        if not self.qubits:
            return {}

        max_t1 = max(q.t1 for q in self.qubits.values()) or 1.0
        max_t2 = max(q.t2 for q in self.qubits.values()) or 1.0

        scores: Dict[int, float] = {}
        for idx, q in self.qubits.items():
            t1_norm = q.t1 / max_t1
            t2_norm = q.t2 / max_t2
            ro_score = 1.0 - q.readout_error
            scores[idx] = (t1_norm + t2_norm + ro_score) / 3.0
        return scores

    def pair_quality_scores(self) -> Dict[Tuple[int, int], float]:
        """Score each qubit pair based on CX error and qubit quality.

        score(i, j) = (1 - cx_error) * sqrt(q_score[i] * q_score[j])
        """
        q_scores = self.qubit_quality_scores()
        pair_scores: Dict[Tuple[int, int], float] = {}
        for gate in self.gates:
            if len(gate.qubits) == 2:
                i, j = gate.qubits
                qi = q_scores.get(i, 0.5)
                qj = q_scores.get(j, 0.5)
                pair_scores[(i, j)] = (1.0 - gate.error_rate) * math.sqrt(qi * qj)
        return pair_scores

    def best_subgraph(self, n_qubits: int) -> List[int]:
        """Find the best connected subgraph of *n_qubits* qubits.

        Uses a greedy expansion strategy: start from the highest-quality
        qubit, then iteratively add the neighbour with the best pair score
        until the requested size is reached.

        Returns a list of qubit indices (may be shorter than *n_qubits*
        if the device has fewer qubits or insufficient connectivity).
        """
        if n_qubits <= 0:
            return []
        if n_qubits >= self.n_qubits:
            return list(self.qubits.keys())

        q_scores = self.qubit_quality_scores()
        if not q_scores:
            return []

        # Build adjacency with pair scores
        adj: Dict[int, Dict[int, float]] = {i: {} for i in self.qubits}
        pair_scores = self.pair_quality_scores()
        for (i, j), score in pair_scores.items():
            adj.setdefault(i, {})[j] = score
            adj.setdefault(j, {})[i] = score

        # Greedy expansion from best qubit
        start = max(q_scores, key=q_scores.get)  # type: ignore[arg-type]
        selected = {start}

        while len(selected) < n_qubits:
            best_candidate = None
            best_score = -1.0
            for node in selected:
                for neighbor, pscore in adj.get(node, {}).items():
                    if neighbor not in selected:
                        combined = pscore + q_scores.get(neighbor, 0.0)
                        if combined > best_score:
                            best_score = combined
                            best_candidate = neighbor
            if best_candidate is None:
                break
            selected.add(best_candidate)

        return sorted(selected)

    def summary(self) -> str:
        """Human-readable summary of processor calibration."""
        lines = [
            f"TransmonProcessor: {self.name}",
            f"  Qubits:          {self.n_qubits}",
            f"  Basis gates:     {', '.join(self.basis_gates)}",
            f"  Coupling edges:  {len(self.coupling_map)}",
            f"  Median T1:       {self.median_t1:.1f} us",
            f"  Median T2:       {self.median_t2:.1f} us",
            f"  Median CX error: {self.median_cx_error:.5f}",
        ]
        if self.timestamp:
            lines.append(f"  Timestamp:       {self.timestamp}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _extract_param(params: list, name: str, default: float = 0.0) -> float:
    """Extract a named parameter from an IBM properties parameter list."""
    for p in params:
        if p.get("name") == name:
            val = p.get("value")
            if val is not None:
                return float(val)
    return default


def parse_ibm_properties(properties_json: dict) -> TransmonProcessor:
    """Parse IBM backend ``properties().to_dict()`` JSON format.

    Parameters
    ----------
    properties_json : dict
        The dictionary produced by ``backend.properties().to_dict()``.

    Returns
    -------
    TransmonProcessor
        Populated device configuration.
    """
    backend_name = properties_json.get("backend_name", "unknown")
    last_update = properties_json.get("last_update_date", "")

    qubits_data = properties_json.get("qubits", [])
    gates_data = properties_json.get("gates", [])
    general = properties_json.get("general", [])

    processor = TransmonProcessor(
        name=backend_name,
        n_qubits=len(qubits_data),
        timestamp=last_update,
    )

    # Parse qubits
    for idx, qubit_params in enumerate(qubits_data):
        processor.qubits[idx] = TransmonQubit(
            index=idx,
            frequency=_extract_param(qubit_params, "frequency"),
            t1=_extract_param(qubit_params, "T1"),
            t2=_extract_param(qubit_params, "T2"),
            readout_error=_extract_param(qubit_params, "readout_error"),
            readout_length=_extract_param(qubit_params, "readout_length"),
            anharmonicity=_extract_param(qubit_params, "anharmonicity", -0.34),
        )

    # Parse gates
    for gate_entry in gates_data:
        gate_type = gate_entry.get("gate", gate_entry.get("name", "unknown"))
        qubit_list = gate_entry.get("qubits", [])
        params = gate_entry.get("parameters", [])
        error_rate = _extract_param(params, "gate_error")
        gate_length = _extract_param(params, "gate_length")
        processor.gates.append(
            TransmonGate(
                gate_type=gate_type,
                qubits=tuple(qubit_list),
                error_rate=error_rate,
                gate_length=gate_length,
            )
        )
        if len(qubit_list) == 2:
            edge = (qubit_list[0], qubit_list[1])
            if edge not in processor.coupling_map:
                processor.coupling_map.append(edge)

    # Basis gates from general section
    for item in general:
        if isinstance(item, dict) and item.get("name") == "basis_gates":
            bg = item.get("value")
            if isinstance(bg, list):
                processor.basis_gates = bg

    return processor


def parse_ibm_v2(backend_config: dict) -> TransmonProcessor:
    """Parse IBM Qiskit Runtime v2 backend configuration.

    Parameters
    ----------
    backend_config : dict
        The v2 configuration dictionary with ``"qubits"`` and ``"gates"``
        top-level keys and optional ``"coupling_map"`` list.

    Returns
    -------
    TransmonProcessor
        Populated device configuration.
    """
    name = backend_config.get("backend_name", backend_config.get("name", "unknown"))
    n_qubits = backend_config.get("n_qubits", 0)
    timestamp = backend_config.get("timestamp", "")

    processor = TransmonProcessor(
        name=name,
        n_qubits=n_qubits,
        timestamp=timestamp,
    )

    # v2 qubit data is a list-of-lists of property dicts
    for idx, qubit_props in enumerate(backend_config.get("qubits", [])):
        if isinstance(qubit_props, dict):
            processor.qubits[idx] = TransmonQubit(
                index=idx,
                frequency=qubit_props.get("frequency", 5.0),
                t1=qubit_props.get("t1", 100.0),
                t2=qubit_props.get("t2", 80.0),
                readout_error=qubit_props.get("readout_error", 0.01),
                readout_length=qubit_props.get("readout_length", 800.0),
                anharmonicity=qubit_props.get("anharmonicity", -0.34),
            )
        elif isinstance(qubit_props, list):
            # Same format as v1 parameter list
            processor.qubits[idx] = TransmonQubit(
                index=idx,
                frequency=_extract_param(qubit_props, "frequency"),
                t1=_extract_param(qubit_props, "T1"),
                t2=_extract_param(qubit_props, "T2"),
                readout_error=_extract_param(qubit_props, "readout_error"),
                readout_length=_extract_param(qubit_props, "readout_length"),
                anharmonicity=_extract_param(qubit_props, "anharmonicity", -0.34),
            )

    # v2 gates
    for gate_entry in backend_config.get("gates", []):
        gate_type = gate_entry.get("gate", gate_entry.get("name", "unknown"))
        qubit_list = gate_entry.get("qubits", [])
        error_rate = gate_entry.get("error_rate", gate_entry.get("gate_error", 0.0))
        gate_length = gate_entry.get("gate_length", 0.0)
        # Handle nested parameters dict
        params = gate_entry.get("parameters", {})
        if isinstance(params, dict):
            error_rate = params.get("gate_error", error_rate)
            gate_length = params.get("gate_length", gate_length)
        elif isinstance(params, list):
            error_rate = _extract_param(params, "gate_error", error_rate)
            gate_length = _extract_param(params, "gate_length", gate_length)
        processor.gates.append(
            TransmonGate(
                gate_type=gate_type,
                qubits=tuple(qubit_list),
                error_rate=error_rate,
                gate_length=gate_length,
            )
        )

    # Coupling map
    cmap = backend_config.get("coupling_map", [])
    for edge in cmap:
        if len(edge) == 2:
            processor.coupling_map.append((edge[0], edge[1]))

    # Basis gates
    bg = backend_config.get("basis_gates", None)
    if bg is not None:
        processor.basis_gates = list(bg)

    if n_qubits == 0 and processor.qubits:
        processor.n_qubits = len(processor.qubits)

    return processor


# ---------------------------------------------------------------------------
# Preset devices
# ---------------------------------------------------------------------------

def _heavy_hex_coupling(n_qubits: int) -> List[Tuple[int, int]]:
    """Generate a simplified heavy-hex coupling map.

    IBM Eagle/Heron processors use a heavy-hex lattice. This generates
    a representative subset of edges for a device of the given size.
    """
    edges: List[Tuple[int, int]] = []
    # Build a chain backbone with periodic cross-links every 4 qubits
    for i in range(n_qubits - 1):
        edges.append((i, i + 1))
    # Cross-links characteristic of heavy-hex
    for i in range(0, n_qubits - 4, 8):
        bridge = i + 4
        if bridge < n_qubits:
            edges.append((i, bridge))
    return edges


def _build_preset_processor(
    name: str,
    n_qubits: int,
    median_t1: float,
    median_t2: float,
    median_readout_error: float,
    median_cx_error: float,
    median_freq: float,
    basis_gates: List[str],
    cx_gate_name: str = "cx",
) -> TransmonProcessor:
    """Build a synthetic TransmonProcessor from aggregate published specs."""
    rng = np.random.RandomState(42)

    qubits: Dict[int, TransmonQubit] = {}
    for i in range(n_qubits):
        # Add realistic spread: +/- 15% around median
        t1 = median_t1 * (1.0 + 0.15 * (rng.random() * 2 - 1))
        t2 = min(median_t2 * (1.0 + 0.15 * (rng.random() * 2 - 1)), 2.0 * t1)
        ro = median_readout_error * (1.0 + 0.3 * (rng.random() * 2 - 1))
        ro = max(0.0, min(1.0, ro))
        freq = median_freq + 0.1 * (rng.random() * 2 - 1)
        qubits[i] = TransmonQubit(
            index=i,
            frequency=freq,
            t1=t1,
            t2=t2,
            readout_error=ro,
            readout_length=800.0,
        )

    coupling = _heavy_hex_coupling(n_qubits)
    gates: List[TransmonGate] = []

    # Single-qubit gates
    for i in range(n_qubits):
        for gname in basis_gates:
            if gname in ("cx", "ecr", "cz"):
                continue
            gates.append(
                TransmonGate(
                    gate_type=gname,
                    qubits=(i,),
                    error_rate=3e-4 * (1.0 + 0.2 * (rng.random() * 2 - 1)),
                    gate_length=25.0 if gname == "rz" else 35.0,
                )
            )

    # Two-qubit gates
    for edge in coupling:
        err = median_cx_error * (1.0 + 0.3 * (rng.random() * 2 - 1))
        gates.append(
            TransmonGate(
                gate_type=cx_gate_name,
                qubits=edge,
                error_rate=max(0.0, err),
                gate_length=300.0,
            )
        )

    return TransmonProcessor(
        name=name,
        n_qubits=n_qubits,
        qubits=qubits,
        gates=gates,
        coupling_map=coupling,
        basis_gates=basis_gates,
    )


def ibm_eagle_r3() -> TransmonProcessor:
    """IBM Eagle r3 (127 qubits) preset based on published specs.

    Returns a ``TransmonProcessor`` with synthetic per-qubit calibration
    data drawn around IBM's published median values for the Eagle r3
    processor revision (2024).
    """
    return _build_preset_processor(
        name="ibm_eagle_r3",
        n_qubits=127,
        median_t1=300.0,       # us
        median_t2=150.0,       # us
        median_readout_error=0.012,
        median_cx_error=0.008,
        median_freq=5.0,       # GHz
        basis_gates=["cx", "id", "rz", "sx", "x"],
        cx_gate_name="cx",
    )


def ibm_heron_r2() -> TransmonProcessor:
    """IBM Heron r2 (156 qubits) preset based on published specs.

    Heron uses ECR as the native two-qubit gate and features improved
    coherence times and lower error rates compared to Eagle.
    """
    return _build_preset_processor(
        name="ibm_heron_r2",
        n_qubits=156,
        median_t1=350.0,       # us
        median_t2=200.0,       # us
        median_readout_error=0.008,
        median_cx_error=0.004,
        median_freq=5.1,       # GHz
        basis_gates=["ecr", "id", "rz", "sx", "x"],
        cx_gate_name="ecr",
    )
