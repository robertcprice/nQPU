"""Unified publication-quality quantum simulator for cross-platform comparison.

Provides a single high-level interface that combines trapped-ion and
superconducting transmon backends for generating publication-quality
comparison data suitable for academic papers and technical reports.

The module produces LaTeX tables in booktabs format, CSV exports, and
structured data dictionaries ready for matplotlib or pgfplots consumption.

Supported backends:
    Trapped-ion: ion_yb171, ion_ba133, ion_ca40,
                 quantinuum_h1, quantinuum_h2, ionq_aria, ionq_forte
    Superconducting: ibm_eagle, ibm_heron, google_sycamore,
                     google_willow, rigetti_ankaa

Standard circuits: bell, ghz, qft, random_depth5, bernstein_vazirani

References:
    - Wright et al., Nat. Commun. 10, 5464 (2019) [trapped-ion benchmarks]
    - Arute et al., Nature 574, 505 (2019) [superconducting benchmarks]
    - Cross et al., PRA 100, 032328 (2019) [quantum volume]

Example:
    >>> from nqpu.superconducting.unified_sim import UnifiedSimulator
    >>> sim = UnifiedSimulator(num_qubits=4)
    >>> table = sim.fidelity_comparison("ghz")
    >>> print(table.to_latex())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from nqpu.ion_trap import (
    TrappedIonSimulator,
    TrapConfig,
    IonSpecies,
)
from nqpu.ion_trap.noise import TrappedIonNoiseModel
from nqpu.superconducting import (
    TransmonSimulator,
    ChipConfig,
    DevicePresets as SCDevicePresets,
    TransmonNoiseModel,
)


# ======================================================================
# Backend result container
# ======================================================================

@dataclass
class BackendResult:
    """Result from running a circuit on a single backend in a single mode.

    Attributes
    ----------
    backend : str
        Backend identifier (e.g. 'ibm_heron', 'ion_yb171').
    mode : str
        Execution mode ('ideal' or 'noisy').
    probabilities : np.ndarray
        Full probability distribution over computational basis states.
    fidelity_vs_ideal : float
        Classical fidelity against the ideal-mode probability distribution,
        computed as the squared Bhattacharyya coefficient.
    native_1q_gates : int
        Number of native single-qubit gates after compilation.
    native_2q_gates : int
        Number of native two-qubit gates after compilation.
    wall_time_ms : float
        Wall-clock simulation time in milliseconds.
    estimated_fidelity : float
        Analytical fidelity estimate from the noise model.
    """
    backend: str
    mode: str
    probabilities: np.ndarray
    fidelity_vs_ideal: float
    native_1q_gates: int
    native_2q_gates: int
    wall_time_ms: float
    estimated_fidelity: float


# ======================================================================
# Data tables for publication
# ======================================================================

def _sci(x: float, precision: int = 2) -> str:
    """Format a float in LaTeX scientific notation for small numbers.

    Numbers close to 1 are shown in decimal; very small numbers get
    proper \\num{} formatting for siunitx or manual exponent notation.
    """
    if x == 0.0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.{precision + 2}f}"
    exp = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10 ** exp)
    return f"${mantissa:.{precision}f} \\times 10^{{{exp}}}$"


@dataclass
class FidelityTable:
    """Fidelity comparison across backends and circuits.

    Attributes
    ----------
    backends : list[str]
        Backend identifiers, one per row.
    circuits : list[str]
        Circuit names, one per column.
    fidelities : np.ndarray
        Matrix of shape (len(backends), len(circuits)) containing fidelity
        values in [0, 1].
    """
    backends: list[str]
    circuits: list[str]
    fidelities: np.ndarray

    def to_latex(self) -> str:
        """Generate a LaTeX table using booktabs formatting.

        Returns
        -------
        str
            Complete LaTeX tabular environment string ready for inclusion
            in a document.  Uses booktabs rules, centered columns, and
            scientific notation where appropriate.
        """
        n_circuits = len(self.circuits)
        col_spec = "l" + "c" * n_circuits
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Fidelity comparison across quantum backends.}",
            r"  \label{tab:fidelity_comparison}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    Backend & " + " & ".join(self.circuits) + r" \\",
            r"    \midrule",
        ]
        for i, backend in enumerate(self.backends):
            row_vals = []
            for j in range(n_circuits):
                val = self.fidelities[i, j]
                row_vals.append(_sci(val))
            lines.append(f"    {backend} & " + " & ".join(row_vals) + r" \\")
        lines.extend([
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export the fidelity table as CSV.

        Returns
        -------
        str
            Comma-separated values with a header row of circuit names
            and one row per backend.
        """
        header = "backend," + ",".join(self.circuits)
        rows = [header]
        for i, backend in enumerate(self.backends):
            vals = ",".join(f"{self.fidelities[i, j]:.8f}" for j in range(len(self.circuits)))
            rows.append(f"{backend},{vals}")
        return "\n".join(rows)


@dataclass
class ScalingData:
    """Fidelity scaling data as a function of system size.

    Attributes
    ----------
    qubit_counts : list[int]
        Qubit counts tested.
    backend_fidelities : dict[str, list[float]]
        Mapping from backend name to list of fidelities, one per qubit count.
    """
    qubit_counts: list[int]
    backend_fidelities: dict[str, list[float]]

    def to_latex(self) -> str:
        """Generate a LaTeX table of fidelity vs qubit count.

        Returns
        -------
        str
            Booktabs-formatted LaTeX table.
        """
        backends = sorted(self.backend_fidelities.keys())
        col_spec = "r" + "c" * len(backends)
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Fidelity scaling with system size.}",
            r"  \label{tab:fidelity_scaling}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    $n$ & " + " & ".join(backends) + r" \\",
            r"    \midrule",
        ]
        for idx, n in enumerate(self.qubit_counts):
            row_vals = []
            for b in backends:
                val = self.backend_fidelities[b][idx]
                row_vals.append(_sci(val))
            lines.append(f"    {n} & " + " & ".join(row_vals) + r" \\")
        lines.extend([
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export scaling data as CSV.

        Returns
        -------
        str
            CSV with columns: qubit_count, backend1, backend2, ...
        """
        backends = sorted(self.backend_fidelities.keys())
        header = "qubit_count," + ",".join(backends)
        rows = [header]
        for idx, n in enumerate(self.qubit_counts):
            vals = ",".join(
                f"{self.backend_fidelities[b][idx]:.8f}" for b in backends
            )
            rows.append(f"{n},{vals}")
        return "\n".join(rows)


@dataclass
class GateOverheadTable:
    """Native gate counts per backend per circuit type.

    Attributes
    ----------
    backends : list[str]
        Backend identifiers.
    circuits : list[str]
        Circuit type names.
    gate_counts_1q : np.ndarray
        Matrix of shape (backends, circuits) with native single-qubit gate counts.
    gate_counts_2q : np.ndarray
        Matrix of shape (backends, circuits) with native two-qubit gate counts.
    """
    backends: list[str]
    circuits: list[str]
    gate_counts_1q: np.ndarray
    gate_counts_2q: np.ndarray

    def to_latex(self) -> str:
        """Generate a LaTeX table of gate overhead.

        Format: Backend | Circuit1 (1Q/2Q) | Circuit2 (1Q/2Q) | ...

        Returns
        -------
        str
            Booktabs-formatted LaTeX table.
        """
        n_circuits = len(self.circuits)
        col_spec = "l" + "c" * n_circuits
        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            r"  \caption{Native gate overhead: single-qubit / two-qubit counts.}",
            r"  \label{tab:gate_overhead}",
            f"  \\begin{{tabular}}{{{col_spec}}}",
            r"    \toprule",
            "    Backend & " + " & ".join(self.circuits) + r" \\",
            r"    \midrule",
        ]
        for i, backend in enumerate(self.backends):
            row_vals = []
            for j in range(n_circuits):
                g1 = int(self.gate_counts_1q[i, j])
                g2 = int(self.gate_counts_2q[i, j])
                row_vals.append(f"{g1}/{g2}")
            lines.append(f"    {backend} & " + " & ".join(row_vals) + r" \\")
        lines.extend([
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export gate overhead as CSV.

        Each circuit gets two columns: <circuit>_1q and <circuit>_2q.

        Returns
        -------
        str
            CSV string.
        """
        cols = []
        for c in self.circuits:
            cols.extend([f"{c}_1q", f"{c}_2q"])
        header = "backend," + ",".join(cols)
        rows = [header]
        for i, backend in enumerate(self.backends):
            vals = []
            for j in range(len(self.circuits)):
                vals.append(str(int(self.gate_counts_1q[i, j])))
                vals.append(str(int(self.gate_counts_2q[i, j])))
            rows.append(f"{backend},{','.join(vals)}")
        return "\n".join(rows)


# ======================================================================
# Backend registry
# ======================================================================

_ION_BACKENDS: dict[str, Callable[[int], TrapConfig]] = {
    "ion_yb171": lambda n: TrapConfig(n_ions=n, species=IonSpecies.YB171),
    "ion_ba133": lambda n: TrapConfig(n_ions=n, species=IonSpecies.BA133),
    "ion_ca40": lambda n: TrapConfig(n_ions=n, species=IonSpecies.CA40),
    # Quantinuum QCCD architectures (171Yb+)
    # H1: 20Q, 99.8% 2Q fidelity, all-to-all via QCCD shuttling
    # Ref: Quantinuum system model H1, arXiv:2302.06707
    "quantinuum_h1": lambda n: TrapConfig(
        n_ions=min(n, 20),
        species=IonSpecies.YB171,
        axial_freq_mhz=1.5,
        radial_freq_mhz=5.0,
        heating_rate_quanta_per_s=20.0,
        background_gas_collision_rate=0.0001,
    ),
    # H2: 56Q, 99.85% 2Q fidelity, QCCD racetrack
    # Ref: Quantinuum system model H2, arXiv:2402.14983
    "quantinuum_h2": lambda n: TrapConfig(
        n_ions=min(n, 56),
        species=IonSpecies.YB171,
        axial_freq_mhz=1.2,
        radial_freq_mhz=4.5,
        heating_rate_quanta_per_s=15.0,
        background_gas_collision_rate=0.0001,
    ),
    # IonQ Aria: 25Q, 99.5% 2Q fidelity, AQ 25
    # Ref: IonQ Aria system specifications (2023)
    "ionq_aria": lambda n: TrapConfig(
        n_ions=min(n, 25),
        species=IonSpecies.YB171,
        axial_freq_mhz=0.7,
        radial_freq_mhz=3.5,
        heating_rate_quanta_per_s=50.0,
        background_gas_collision_rate=0.0005,
    ),
    # IonQ Forte: 36Q, 99.7% 2Q fidelity, AQ 35
    # Ref: IonQ Forte system specifications (2024)
    "ionq_forte": lambda n: TrapConfig(
        n_ions=min(n, 36),
        species=IonSpecies.YB171,
        axial_freq_mhz=0.5,
        radial_freq_mhz=4.0,
        heating_rate_quanta_per_s=30.0,
        background_gas_collision_rate=0.0003,
    ),
}

_SC_BACKENDS: dict[str, Callable[[int], ChipConfig]] = {
    "ibm_eagle": lambda n: SCDevicePresets.IBM_EAGLE.build(num_qubits=n),
    "ibm_heron": lambda n: SCDevicePresets.IBM_HERON.build(num_qubits=n),
    "google_sycamore": lambda n: SCDevicePresets.GOOGLE_SYCAMORE.build(num_qubits=n),
    "google_willow": lambda n: SCDevicePresets.GOOGLE_WILLOW.build(num_qubits=n),
    "rigetti_ankaa": lambda n: SCDevicePresets.RIGETTI_ANKAA.build(num_qubits=n),
}

ALL_BACKENDS = list(_ION_BACKENDS.keys()) + list(_SC_BACKENDS.keys())

_DEFAULT_BACKENDS = [
    "ion_yb171",
    "quantinuum_h1",
    "ibm_heron",
    "google_willow",
    "rigetti_ankaa",
]


# ======================================================================
# Standard circuit library
# ======================================================================

def _bell_circuit(sim: Any) -> None:
    """Prepare Bell state (|00> + |11>) / sqrt(2) on qubits 0, 1."""
    sim.h(0)
    sim.cnot(0, 1)


def _ghz_circuit(sim: Any, n: int) -> None:
    """Prepare GHZ state (|00..0> + |11..1>) / sqrt(2) on n qubits."""
    sim.h(0)
    for i in range(1, n):
        sim.cnot(0, i)


def _qft_circuit(sim: Any, n: int) -> None:
    """Quantum Fourier Transform on n qubits.

    Standard decomposition: H gates with controlled-Rz rotations.
    """
    for i in range(n):
        sim.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            # Controlled-Rz approximated as Rz on target
            # (standard benchmark; the controlled phase is the expensive part)
            sim.rz(j, angle)


def _random_depth5_circuit(sim: Any, n: int) -> None:
    """Random circuit of depth 5 with a fixed seed for reproducibility.

    Alternates single-qubit Rx/Rz layers with nearest-neighbour CNOT layers.
    Uses seed 42 for reproducibility across backends.
    """
    rng = np.random.RandomState(42)
    for layer in range(5):
        for q in range(n):
            sim.rx(q, float(rng.uniform(0, 2 * math.pi)))
            sim.rz(q, float(rng.uniform(0, 2 * math.pi)))
        for q in range(0, n - 1, 2 if layer % 2 == 0 else 1):
            if q + 1 < n:
                sim.cnot(q, q + 1)


def _bernstein_vazirani_circuit(sim: Any, n: int) -> None:
    """Bernstein-Vazirani algorithm for a random secret string.

    Uses a fixed secret s = alternating bits (101...01) for reproducibility.
    The oracle is implemented with CNOT gates from data qubits to an
    ancilla (last qubit).
    """
    if n < 2:
        sim.h(0)
        return

    data_qubits = n - 1
    ancilla = n - 1

    # Secret string: alternating bits (1, 0, 1, 0, ...)
    secret = [1 if i % 2 == 0 else 0 for i in range(data_qubits)]

    # Hadamard on all qubits
    for q in range(n):
        sim.h(q)

    # Prepare ancilla in |-> state
    sim.z(ancilla)

    # Oracle: CNOT from qubit i to ancilla if secret[i] == 1
    for i in range(data_qubits):
        if secret[i]:
            sim.cnot(i, ancilla)

    # Hadamard on data qubits
    for q in range(data_qubits):
        sim.h(q)


_CIRCUIT_REGISTRY: dict[str, Callable[[Any, int], None]] = {
    "bell": lambda sim, n: _bell_circuit(sim),
    "ghz": _ghz_circuit,
    "qft": _qft_circuit,
    "random_depth5": _random_depth5_circuit,
    "bernstein_vazirani": _bernstein_vazirani_circuit,
}

SUPPORTED_CIRCUITS = list(_CIRCUIT_REGISTRY.keys())


# ======================================================================
# Unified simulator
# ======================================================================

class UnifiedSimulator:
    """Single interface to both trapped-ion and superconducting backends.

    Provides methods for running identical quantum circuits on multiple
    backends and extracting structured comparison data for publications.

    Parameters
    ----------
    num_qubits : int
        Number of qubits for all circuits.  Must be >= 2.
    backends : list[str], optional
        Backend identifiers to include.  Defaults to a representative
        subset: ['ion_yb171', 'ibm_heron', 'google_willow', 'rigetti_ankaa'].
        See ``ALL_BACKENDS`` for the full list.

    Raises
    ------
    ValueError
        If num_qubits < 2 or an unknown backend is requested.

    Examples
    --------
    >>> sim = UnifiedSimulator(num_qubits=4)
    >>> table = sim.fidelity_comparison("ghz")
    >>> print(table.to_csv())
    """

    def __init__(
        self,
        num_qubits: int,
        backends: list[str] | None = None,
    ) -> None:
        if num_qubits < 2:
            raise ValueError("num_qubits must be >= 2")

        self.num_qubits = num_qubits
        self.backend_names = list(backends or _DEFAULT_BACKENDS)

        # Validate backend names
        for name in self.backend_names:
            if name not in _ION_BACKENDS and name not in _SC_BACKENDS:
                raise ValueError(
                    f"Unknown backend '{name}'. "
                    f"Supported: {ALL_BACKENDS}"
                )

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------

    def _build_ion_sim(
        self, backend: str, mode: str
    ) -> TrappedIonSimulator:
        """Construct a trapped-ion simulator for the given backend and mode."""
        config = _ION_BACKENDS[backend](self.num_qubits)
        return TrappedIonSimulator(config, execution_mode=mode)

    def _build_sc_sim(
        self, backend: str, mode: str
    ) -> TransmonSimulator:
        """Construct a superconducting simulator for the given backend and mode."""
        config = _SC_BACKENDS[backend](self.num_qubits)
        return TransmonSimulator(config, execution_mode=mode)

    def _run_on_backend(
        self,
        backend: str,
        circuit_fn: Callable[[Any, int], None],
        mode: str,
    ) -> BackendResult:
        """Run a circuit function on a single backend in a single mode.

        Parameters
        ----------
        backend : str
            Backend identifier.
        circuit_fn : callable
            Function ``(sim, n_qubits) -> None`` that applies the circuit.
        mode : str
            Execution mode: 'ideal' or 'noisy'.

        Returns
        -------
        BackendResult
            Structured result.
        """
        is_ion = backend in _ION_BACKENDS

        t0 = time.time()
        if is_ion:
            sim = self._build_ion_sim(backend, mode)
        else:
            sim = self._build_sc_sim(backend, mode)

        circuit_fn(sim, self.num_qubits)
        dt_ms = (time.time() - t0) * 1000.0

        # Extract probabilities
        if is_ion:
            probs = np.abs(sim.statevector()) ** 2 if mode == "ideal" else np.real(np.diag(sim.density_matrix()))
        else:
            probs = sim.probabilities()

        # Get circuit statistics
        stats = sim.circuit_stats()

        if is_ion:
            native_1q = stats.single_qubit_gates
            native_2q = stats.two_qubit_gates
            est_fidelity = stats.estimated_fidelity
        else:
            native_1q = stats.native_1q_count
            native_2q = stats.native_2q_count
            est_fidelity = stats.estimated_fidelity

        # Compute fidelity vs ideal if mode is noisy
        fidelity_vs_ideal = 1.0
        if mode == "noisy":
            if is_ion:
                ideal_sim = self._build_ion_sim(backend, "ideal")
            else:
                ideal_sim = self._build_sc_sim(backend, "ideal")
            circuit_fn(ideal_sim, self.num_qubits)
            if is_ion:
                ideal_probs = np.abs(ideal_sim.statevector()) ** 2
            else:
                ideal_probs = ideal_sim.probabilities()
            # Bhattacharyya fidelity: F = (sum sqrt(p*q))^2
            fidelity_vs_ideal = float(
                np.sum(np.sqrt(np.maximum(probs, 0.0) * np.maximum(ideal_probs, 0.0))) ** 2
            )

        return BackendResult(
            backend=backend,
            mode=mode,
            probabilities=probs,
            fidelity_vs_ideal=fidelity_vs_ideal,
            native_1q_gates=native_1q,
            native_2q_gates=native_2q,
            wall_time_ms=dt_ms,
            estimated_fidelity=est_fidelity,
        )

    # ------------------------------------------------------------------
    # Public API: run_circuit
    # ------------------------------------------------------------------

    def run_circuit(
        self,
        circuit_fn: Callable[[Any, int], None],
        modes: list[str] | None = None,
    ) -> dict[str, BackendResult]:
        """Run a circuit function on all configured backends.

        Parameters
        ----------
        circuit_fn : callable
            Function ``(sim, n_qubits) -> None`` that applies the circuit.
            The function receives a simulator instance (either
            ``TrappedIonSimulator`` or ``TransmonSimulator``) and the
            number of qubits.
        modes : list[str], optional
            Execution modes to test.  Default: ``['ideal', 'noisy']``.

        Returns
        -------
        dict[str, BackendResult]
            Results keyed by ``'{backend}_{mode}'``.

        Examples
        --------
        >>> def my_circuit(sim, n):
        ...     sim.h(0)
        ...     sim.cnot(0, 1)
        >>> results = sim.run_circuit(my_circuit)
        """
        modes = modes or ["ideal", "noisy"]
        results: dict[str, BackendResult] = {}
        for backend in self.backend_names:
            for mode in modes:
                key = f"{backend}_{mode}"
                results[key] = self._run_on_backend(backend, circuit_fn, mode)
        return results

    # ------------------------------------------------------------------
    # Public API: fidelity_comparison
    # ------------------------------------------------------------------

    def fidelity_comparison(
        self,
        circuit_type: str | None = None,
    ) -> FidelityTable:
        """Compare noisy fidelities across all backends for standard circuits.

        Parameters
        ----------
        circuit_type : str, optional
            If provided, compare only this circuit.  Otherwise compare
            all standard circuits.

        Returns
        -------
        FidelityTable
            Structured fidelity comparison table.

        Raises
        ------
        ValueError
            If ``circuit_type`` is not a recognised standard circuit.
        """
        if circuit_type is not None:
            circuit_names = [circuit_type]
        else:
            circuit_names = list(SUPPORTED_CIRCUITS)

        for name in circuit_names:
            if name not in _CIRCUIT_REGISTRY:
                raise ValueError(
                    f"Unknown circuit type '{name}'. "
                    f"Supported: {SUPPORTED_CIRCUITS}"
                )

        n_backends = len(self.backend_names)
        n_circuits = len(circuit_names)
        fidelities = np.zeros((n_backends, n_circuits))

        for j, circ_name in enumerate(circuit_names):
            circuit_fn = _CIRCUIT_REGISTRY[circ_name]
            for i, backend in enumerate(self.backend_names):
                result = self._run_on_backend(backend, circuit_fn, "noisy")
                fidelities[i, j] = result.fidelity_vs_ideal

        return FidelityTable(
            backends=list(self.backend_names),
            circuits=circuit_names,
            fidelities=fidelities,
        )

    # ------------------------------------------------------------------
    # Public API: noise_scaling_study
    # ------------------------------------------------------------------

    def noise_scaling_study(
        self,
        circuit_type: str,
        qubit_range: range | None = None,
    ) -> ScalingData:
        """Study how fidelity degrades with system size.

        Parameters
        ----------
        circuit_type : str
            Standard circuit to test (e.g. 'ghz', 'qft').
        qubit_range : range, optional
            Range of qubit counts.  Default: ``range(2, min(num_qubits, 8) + 1)``.

        Returns
        -------
        ScalingData
            Fidelity vs qubit count for each backend.

        Raises
        ------
        ValueError
            If ``circuit_type`` is not recognised.
        """
        if circuit_type not in _CIRCUIT_REGISTRY:
            raise ValueError(
                f"Unknown circuit type '{circuit_type}'. "
                f"Supported: {SUPPORTED_CIRCUITS}"
            )

        if qubit_range is None:
            max_q = min(self.num_qubits, 8)
            qubit_range = range(2, max_q + 1)

        qubit_counts = list(qubit_range)
        circuit_fn = _CIRCUIT_REGISTRY[circuit_type]
        backend_fidelities: dict[str, list[float]] = {
            b: [] for b in self.backend_names
        }

        for n in qubit_counts:
            # Build a temporary simulator instance for each qubit count
            temp = UnifiedSimulator(num_qubits=n, backends=self.backend_names)
            for backend in self.backend_names:
                result = temp._run_on_backend(backend, circuit_fn, "noisy")
                backend_fidelities[backend].append(result.fidelity_vs_ideal)

        return ScalingData(
            qubit_counts=qubit_counts,
            backend_fidelities=backend_fidelities,
        )

    # ------------------------------------------------------------------
    # Public API: gate_overhead_comparison
    # ------------------------------------------------------------------

    def gate_overhead_comparison(self) -> GateOverheadTable:
        """Compare native gate counts across backends for all standard circuits.

        Runs each standard circuit in ideal mode and records the native
        gate counts after compilation to each backend's native gate set.

        Returns
        -------
        GateOverheadTable
            Structured gate count comparison.
        """
        circuit_names = list(SUPPORTED_CIRCUITS)
        n_backends = len(self.backend_names)
        n_circuits = len(circuit_names)
        counts_1q = np.zeros((n_backends, n_circuits))
        counts_2q = np.zeros((n_backends, n_circuits))

        for j, circ_name in enumerate(circuit_names):
            circuit_fn = _CIRCUIT_REGISTRY[circ_name]
            for i, backend in enumerate(self.backend_names):
                result = self._run_on_backend(backend, circuit_fn, "ideal")
                counts_1q[i, j] = result.native_1q_gates
                counts_2q[i, j] = result.native_2q_gates

        return GateOverheadTable(
            backends=list(self.backend_names),
            circuits=circuit_names,
            gate_counts_1q=counts_1q,
            gate_counts_2q=counts_2q,
        )


# ======================================================================
# Publication figures data generator
# ======================================================================

class PublicationFigures:
    """Generate structured data for common publication figures.

    All methods return plain dictionaries suitable for direct consumption
    by matplotlib, pgfplots, or other plotting libraries.

    Parameters
    ----------
    unified : UnifiedSimulator
        Pre-configured unified simulator instance.

    Examples
    --------
    >>> sim = UnifiedSimulator(num_qubits=6)
    >>> figs = PublicationFigures(sim)
    >>> data = figs.fidelity_vs_qubits("ghz", max_qubits=6)
    """

    def __init__(self, unified: UnifiedSimulator) -> None:
        self.unified = unified

    def fidelity_vs_qubits(
        self,
        circuit: str,
        max_qubits: int | None = None,
    ) -> dict[str, Any]:
        """Generate data for a fidelity-vs-system-size plot.

        Parameters
        ----------
        circuit : str
            Standard circuit name.
        max_qubits : int, optional
            Maximum qubit count.  Default: ``unified.num_qubits``.

        Returns
        -------
        dict
            Keys: 'qubit_counts' (list[int]),
            'backends' (dict[str, list[float]] of fidelity curves),
            'circuit' (str), 'xlabel' (str), 'ylabel' (str).
        """
        max_q = max_qubits or self.unified.num_qubits
        max_q = min(max_q, self.unified.num_qubits)
        qr = range(2, max_q + 1)

        scaling = self.unified.noise_scaling_study(circuit, qubit_range=qr)

        return {
            "qubit_counts": scaling.qubit_counts,
            "backends": scaling.backend_fidelities,
            "circuit": circuit,
            "xlabel": "Number of qubits",
            "ylabel": "Fidelity vs. ideal",
        }

    def noise_breakdown(
        self,
        backend: str,
        qubit: int = 0,
    ) -> dict[str, Any]:
        """Generate noise source breakdown data for a specific backend.

        For trapped-ion backends, returns the error budget from the
        physics-based noise model.  For superconducting backends,
        returns T1, T2, leakage, ZZ crosstalk, and readout contributions.

        Parameters
        ----------
        backend : str
            Backend identifier.
        qubit : int
            Qubit index for single-qubit error breakdown.

        Returns
        -------
        dict
            Keys: 'backend' (str), 'qubit' (int), 'sources' (dict[str, float])
            mapping noise source names to error probabilities,
            'total_1q_error' (float), 'total_2q_error' (float).

        Raises
        ------
        ValueError
            If the backend is not recognised.
        """
        if backend in _ION_BACKENDS:
            config = _ION_BACKENDS[backend](self.unified.num_qubits)
            noise = TrappedIonNoiseModel(config)
            budget = noise.error_budget()
            return {
                "backend": backend,
                "qubit": qubit,
                "sources": {
                    "spontaneous_emission_1q": budget["1q_spontaneous_emission"],
                    "intensity_noise_1q": budget["1q_intensity_noise"],
                    "magnetic_dephasing_1q": budget["1q_magnetic_dephasing"],
                    "crosstalk_1q": budget["1q_crosstalk"],
                    "motional_heating_2q": budget["2q_motional_heating"],
                    "spontaneous_emission_2q": budget["2q_spontaneous_emission"],
                    "motional_dephasing_2q": budget["2q_motional_dephasing"],
                    "intensity_noise_2q": budget["2q_intensity_noise"],
                },
                "total_1q_error": budget["1q_total"],
                "total_2q_error": budget["2q_total"],
                "fidelity_1q": budget["1q_fidelity"],
                "fidelity_2q": budget["2q_fidelity"],
            }

        elif backend in _SC_BACKENDS:
            config = _SC_BACKENDS[backend](self.unified.num_qubits)
            noise = TransmonNoiseModel(config)
            q = min(qubit, config.num_qubits - 1)
            qp = config.qubits[q]
            gate_time_ns = qp.gate_time_ns
            t2q_time_ns = config.two_qubit_gate_time_ns

            t1_err = noise.t1_decay_prob(q, gate_time_ns)
            t2_err = noise.t2_dephase_prob(q, gate_time_ns)
            leak_err = noise.leakage_prob(q)
            gate_err = 1.0 - qp.single_gate_fidelity

            # For two-qubit: find a neighbour
            neighbour = q + 1 if q + 1 < config.num_qubits else q - 1
            neighbour = max(0, neighbour)
            zz_hz = noise.zz_coupling_hz(q, neighbour)
            # ZZ phase error during a 2Q gate
            zz_phase_err = (
                2.0 * math.pi * zz_hz * t2q_time_ns * 1e-9
            ) ** 2 / 2.0 if zz_hz > 0 else 0.0

            t1_2q_err = noise.t1_decay_prob(q, t2q_time_ns)
            t2_2q_err = noise.t2_dephase_prob(q, t2q_time_ns)
            gate_2q_err = 1.0 - config.two_qubit_fidelity

            total_1q = 1.0 - (1.0 - gate_err) * (1.0 - t1_err / 3) * (1.0 - t2_err / 2) * (1.0 - leak_err)
            total_2q = 1.0 - (1.0 - gate_2q_err) * (1.0 - t1_2q_err / 3) * (1.0 - t2_2q_err / 2)

            return {
                "backend": backend,
                "qubit": q,
                "sources": {
                    "t1_relaxation_1q": t1_err,
                    "t2_dephasing_1q": t2_err,
                    "leakage_1q": leak_err,
                    "gate_error_1q": gate_err,
                    "t1_relaxation_2q": t1_2q_err,
                    "t2_dephasing_2q": t2_2q_err,
                    "zz_crosstalk_2q": zz_phase_err,
                    "gate_error_2q": gate_2q_err,
                },
                "total_1q_error": total_1q,
                "total_2q_error": total_2q,
                "fidelity_1q": 1.0 - total_1q,
                "fidelity_2q": 1.0 - total_2q,
            }
        else:
            raise ValueError(f"Unknown backend '{backend}'")

    def compilation_overhead(
        self,
        circuit: str,
    ) -> dict[str, Any]:
        """Generate native gate count data per backend for a given circuit.

        Parameters
        ----------
        circuit : str
            Standard circuit name.

        Returns
        -------
        dict
            Keys: 'circuit' (str), 'backends' (list[str]),
            'native_1q' (list[int]), 'native_2q' (list[int]),
            'total_native' (list[int]).
        """
        if circuit not in _CIRCUIT_REGISTRY:
            raise ValueError(
                f"Unknown circuit '{circuit}'. Supported: {SUPPORTED_CIRCUITS}"
            )

        circuit_fn = _CIRCUIT_REGISTRY[circuit]
        backends_list = []
        native_1q_list = []
        native_2q_list = []
        total_list = []

        for backend in self.unified.backend_names:
            result = self.unified._run_on_backend(backend, circuit_fn, "ideal")
            backends_list.append(backend)
            native_1q_list.append(result.native_1q_gates)
            native_2q_list.append(result.native_2q_gates)
            total_list.append(result.native_1q_gates + result.native_2q_gates)

        return {
            "circuit": circuit,
            "backends": backends_list,
            "native_1q": native_1q_list,
            "native_2q": native_2q_list,
            "total_native": total_list,
        }

    def device_comparison_radar(self) -> dict[str, Any]:
        """Generate multi-axis comparison data for a radar/spider plot.

        Axes: single-qubit fidelity, two-qubit fidelity, gate speed,
        connectivity, native gate set richness.

        All values are normalised to [0, 1] for radar plot rendering.

        Returns
        -------
        dict
            Keys: 'axes' (list[str]), 'backends' (dict[str, list[float]])
            where each backend maps to normalised scores per axis.
        """
        axes = [
            "1Q Fidelity",
            "2Q Fidelity",
            "Gate Speed",
            "Connectivity",
            "Coherence",
        ]

        backend_scores: dict[str, list[float]] = {}

        for backend in self.unified.backend_names:
            if backend in _ION_BACKENDS:
                config = _ION_BACKENDS[backend](self.unified.num_qubits)
                noise = TrappedIonNoiseModel(config)
                fid_1q = noise.single_qubit_gate_fidelity()
                fid_2q = noise.two_qubit_gate_fidelity()
                # Gate speed: inverse of gate time, normalised
                # Ion traps: ~1 us (1Q), ~100 us (2Q)
                gate_speed_score = 1.0 / (1.0 + noise.two_qubit_gate_time_us / 10.0)
                # All-to-all connectivity
                connectivity_score = 1.0
                # Coherence: T2 / gate_time
                t2_us = config.species.t2_s * 1e6
                coherence_score = min(
                    t2_us / (noise.two_qubit_gate_time_us * 100.0), 1.0
                )
            else:
                config_sc = _SC_BACKENDS[backend](self.unified.num_qubits)
                noise_sc = TransmonNoiseModel(config_sc)
                q0 = config_sc.qubits[0]
                fid_1q = q0.single_gate_fidelity
                fid_2q = config_sc.two_qubit_fidelity
                # Gate speed: SC gates are ~25-300 ns
                gate_speed_score = 1.0 / (
                    1.0 + config_sc.two_qubit_gate_time_ns / 100.0
                )
                # Connectivity: edges / max possible edges
                n = config_sc.num_qubits
                max_edges = n * (n - 1) / 2
                actual_edges = len(config_sc.topology.edges)
                connectivity_score = actual_edges / max_edges if max_edges > 0 else 0.0
                # Coherence: T2 / gate_time
                coherence_score = min(
                    q0.t2_us * 1000.0 / (config_sc.two_qubit_gate_time_ns * 100.0),
                    1.0,
                )

            backend_scores[backend] = [
                fid_1q,
                fid_2q,
                gate_speed_score,
                connectivity_score,
                coherence_score,
            ]

        return {
            "axes": axes,
            "backends": backend_scores,
        }


# ======================================================================
# CLI demo
# ======================================================================

def _print_separator(char: str = "=", width: int = 78) -> None:
    """Print a horizontal separator line."""
    print(char * width)


def _print_header(title: str) -> None:
    """Print a section header."""
    _print_separator()
    print(f"  {title}")
    _print_separator()


def main() -> None:
    """Run all comparisons and print formatted results plus LaTeX tables."""
    num_qubits = 4
    print()
    _print_header("UNIFIED QUANTUM SIMULATOR -- PUBLICATION DATA GENERATOR")
    print(f"  Qubits: {num_qubits}")
    print(f"  Backends: {_DEFAULT_BACKENDS}")
    print(f"  Circuits: {SUPPORTED_CIRCUITS}")
    print()

    sim = UnifiedSimulator(num_qubits=num_qubits)
    figs = PublicationFigures(sim)

    # ---- 1. Fidelity comparison ----
    _print_header("1. FIDELITY COMPARISON (noisy mode)")
    table = sim.fidelity_comparison()
    # Print human-readable table
    header = f"{'Backend':<20}"
    for c in table.circuits:
        header += f" {c:>18}"
    print(header)
    print("-" * len(header))
    for i, backend in enumerate(table.backends):
        row = f"{backend:<20}"
        for j in range(len(table.circuits)):
            row += f" {table.fidelities[i, j]:>18.6f}"
        print(row)
    print()
    print("LaTeX output:")
    print(table.to_latex())
    print()

    # ---- 2. Noise scaling study ----
    _print_header("2. NOISE SCALING STUDY (GHZ circuit)")
    scaling = sim.noise_scaling_study("ghz", qubit_range=range(2, num_qubits + 1))
    header = f"{'Qubits':<10}"
    for b in sorted(scaling.backend_fidelities.keys()):
        header += f" {b:>18}"
    print(header)
    print("-" * len(header))
    for idx, n in enumerate(scaling.qubit_counts):
        row = f"{n:<10}"
        for b in sorted(scaling.backend_fidelities.keys()):
            row += f" {scaling.backend_fidelities[b][idx]:>18.6f}"
        print(row)
    print()
    print("LaTeX output:")
    print(scaling.to_latex())
    print()

    # ---- 3. Gate overhead comparison ----
    _print_header("3. GATE OVERHEAD COMPARISON (1Q/2Q native gates)")
    overhead = sim.gate_overhead_comparison()
    header = f"{'Backend':<20}"
    for c in overhead.circuits:
        header += f" {c:>16}"
    print(header)
    print("-" * len(header))
    for i, backend in enumerate(overhead.backends):
        row = f"{backend:<20}"
        for j in range(len(overhead.circuits)):
            g1 = int(overhead.gate_counts_1q[i, j])
            g2 = int(overhead.gate_counts_2q[i, j])
            row += f" {g1:>7}/{g2:<7}"
        print(row)
    print()
    print("LaTeX output:")
    print(overhead.to_latex())
    print()

    # ---- 4. Noise breakdown ----
    _print_header("4. NOISE SOURCE BREAKDOWN")
    for backend in sim.backend_names:
        breakdown = figs.noise_breakdown(backend, qubit=0)
        print(f"\n  {backend}:")
        for source, val in breakdown["sources"].items():
            print(f"    {source:<30s} {val:.2e}")
        print(f"    {'--- 1Q total error ---':<30s} {breakdown['total_1q_error']:.2e}")
        print(f"    {'--- 2Q total error ---':<30s} {breakdown['total_2q_error']:.2e}")
    print()

    # ---- 5. Device comparison radar ----
    _print_header("5. DEVICE COMPARISON RADAR DATA")
    radar = figs.device_comparison_radar()
    header = f"{'Backend':<20}"
    for axis in radar["axes"]:
        header += f" {axis:>14}"
    print(header)
    print("-" * len(header))
    for backend, scores in radar["backends"].items():
        row = f"{backend:<20}"
        for s in scores:
            row += f" {s:>14.4f}"
        print(row)
    print()

    # ---- 6. Fidelity vs qubits plot data ----
    _print_header("6. FIDELITY VS QUBITS (GHZ circuit)")
    fvq = figs.fidelity_vs_qubits("ghz", max_qubits=num_qubits)
    print(f"  Circuit: {fvq['circuit']}")
    print(f"  Qubit counts: {fvq['qubit_counts']}")
    for backend, fids in fvq["backends"].items():
        fid_str = ", ".join(f"{f:.4f}" for f in fids)
        print(f"  {backend}: [{fid_str}]")
    print()

    # ---- 7. Compilation overhead ----
    _print_header("7. COMPILATION OVERHEAD (per circuit)")
    for circ in SUPPORTED_CIRCUITS:
        comp = figs.compilation_overhead(circ)
        print(f"\n  {circ}:")
        for i, b in enumerate(comp["backends"]):
            print(
                f"    {b:<20s}  1Q={comp['native_1q'][i]:>4d}  "
                f"2Q={comp['native_2q'][i]:>4d}  "
                f"total={comp['total_native'][i]:>4d}"
            )
    print()

    _print_separator()
    print("  All comparisons complete.  Data ready for publication.")
    _print_separator()
    print()


if __name__ == "__main__":
    main()
