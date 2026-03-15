"""Quantum Characterization, Verification, and Validation (QCVV) benchmarks.

Provides experiment generators for benchmarking transmon processors:
    - Randomized Benchmarking (RB)
    - Cross-Entropy Benchmarking (XEB)
    - Quantum Volume (QV)
    - GHZ state preparation
    - Bell state fidelity

References:
    - Knill et al., PRA 77, 012307 (2008) [RB]
    - Boixo et al., Nature Phys. 14, 595 (2018) [XEB]
    - Cross et al., PRA 100, 032328 (2019) [QV]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .chip import ChipConfig, DevicePresets
from .simulator import TransmonSimulator, CircuitStats


@dataclass
class BenchmarkResult:
    """Result of a QCVV benchmark experiment."""
    name: str
    num_qubits: int
    metric_name: str
    metric_value: float
    raw_data: dict[str, Any]
    circuit_stats: CircuitStats | None = None


class TransmonQCVV:
    """QCVV benchmark suite for transmon processors."""

    def __init__(self, config: ChipConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Bell state fidelity
    # ------------------------------------------------------------------

    def bell_state_fidelity(
        self,
        qubit_a: int = 0,
        qubit_b: int = 1,
        shots: int = 1024,
        execution_mode: str = "noisy",
    ) -> BenchmarkResult:
        """Measure Bell state |00>+|11> preparation fidelity."""
        sim = TransmonSimulator(self.config, execution_mode=execution_mode)
        sim.h(qubit_a)
        sim.cnot(qubit_a, qubit_b)
        probs = sim.probabilities()

        # Bell fidelity = P(00) + P(11)
        idx_00 = 0
        idx_11 = (1 << qubit_a) | (1 << qubit_b)
        fidelity = float(probs[idx_00] + probs[idx_11])

        return BenchmarkResult(
            name="Bell State Fidelity",
            num_qubits=2,
            metric_name="fidelity",
            metric_value=fidelity,
            raw_data={
                "P_00": float(probs[idx_00]),
                "P_11": float(probs[idx_11]),
                "qubits": (qubit_a, qubit_b),
            },
            circuit_stats=sim.circuit_stats(),
        )

    # ------------------------------------------------------------------
    # GHZ state
    # ------------------------------------------------------------------

    def ghz_fidelity(
        self,
        num_qubits: int | None = None,
        shots: int = 1024,
        execution_mode: str = "noisy",
    ) -> BenchmarkResult:
        """Measure GHZ state preparation fidelity."""
        n = num_qubits or self.config.num_qubits
        sim = TransmonSimulator(self.config, execution_mode=execution_mode)
        sim.h(0)
        for i in range(1, n):
            sim.cnot(0, i)
        probs = sim.probabilities()

        fidelity = float(probs[0] + probs[(1 << n) - 1])

        return BenchmarkResult(
            name="GHZ State Fidelity",
            num_qubits=n,
            metric_name="fidelity",
            metric_value=fidelity,
            raw_data={
                "P_000": float(probs[0]),
                "P_111": float(probs[(1 << n) - 1]),
            },
            circuit_stats=sim.circuit_stats(),
        )

    # ------------------------------------------------------------------
    # Randomized Benchmarking
    # ------------------------------------------------------------------

    def randomized_benchmarking(
        self,
        qubit: int = 0,
        sequence_lengths: list[int] | None = None,
        n_sequences: int = 10,
        shots: int = 256,
        execution_mode: str = "noisy",
    ) -> BenchmarkResult:
        """Run single-qubit randomized benchmarking.

        Applies random Clifford sequences of increasing length and measures
        the survival probability. Fits an exponential decay to extract
        the average gate error.
        """
        if sequence_lengths is None:
            sequence_lengths = [1, 2, 4, 8, 16, 32, 64]

        # Single-qubit Clifford group generators
        cliffords = ["h", "sx", "x", "rz_pi2", "rz_pi"]

        survival_probs = []
        for length in sequence_lengths:
            survivals = []
            for _ in range(n_sequences):
                sim = TransmonSimulator(self.config, execution_mode=execution_mode)
                for _ in range(length):
                    gate = random.choice(cliffords)
                    if gate == "h":
                        sim.h(qubit)
                    elif gate == "sx":
                        sim.sx(qubit)
                    elif gate == "x":
                        sim.x(qubit)
                    elif gate == "rz_pi2":
                        sim.rz(qubit, math.pi / 2)
                    elif gate == "rz_pi":
                        sim.rz(qubit, math.pi)

                probs = sim.probabilities()
                # Survival = probability of measuring |0>
                p0 = sum(probs[i] for i in range(len(probs)) if (i >> qubit) & 1 == 0)
                survivals.append(p0)
            survival_probs.append(float(np.mean(survivals)))

        # Fit exponential decay: p(m) = A * r^m + B
        # Simplified: estimate r from first and last points
        if len(sequence_lengths) >= 2 and survival_probs[0] > 0.5:
            m0, m1 = sequence_lengths[0], sequence_lengths[-1]
            p0, p1 = survival_probs[0], survival_probs[-1]
            if p1 > 0.5 and p0 > 0.5:
                r = ((p1 - 0.5) / max(p0 - 0.5, 1e-10)) ** (1.0 / max(m1 - m0, 1))
            else:
                r = 0.99
            epg = (1.0 - r) / 2.0  # error per gate (d=2)
        else:
            r = 0.99
            epg = 0.005

        return BenchmarkResult(
            name="Randomized Benchmarking",
            num_qubits=1,
            metric_name="error_per_gate",
            metric_value=epg,
            raw_data={
                "sequence_lengths": sequence_lengths,
                "survival_probs": survival_probs,
                "decay_rate": r,
                "qubit": qubit,
            },
        )

    # ------------------------------------------------------------------
    # Quantum Volume
    # ------------------------------------------------------------------

    def quantum_volume(
        self,
        max_depth: int | None = None,
        n_trials: int = 20,
        shots: int = 256,
        execution_mode: str = "noisy",
    ) -> BenchmarkResult:
        """Estimate quantum volume.

        Tests random SU(4) circuits of width=depth=m for increasing m.
        QV = 2^m where m is the largest depth achieving >2/3 heavy output
        probability.
        """
        max_m = min(max_depth or self.config.num_qubits, self.config.num_qubits, 6)
        best_m = 0

        qv_data = {}
        for m in range(1, max_m + 1):
            heavy_counts = 0
            for _ in range(n_trials):
                sim = TransmonSimulator(self.config, execution_mode=execution_mode)
                # Random SU(4) layers
                for _ in range(m):
                    perm = list(range(m))
                    random.shuffle(perm)
                    for q in range(m):
                        sim.rx(q, random.uniform(0, 2 * math.pi))
                        sim.rz(q, random.uniform(0, 2 * math.pi))
                    for i in range(0, m - 1, 2):
                        sim.cnot(perm[i], perm[i + 1])

                probs = sim.probabilities()
                median_prob = float(np.median(probs))
                heavy_prob = sum(p for p in probs if p > median_prob)
                if heavy_prob > 2.0 / 3.0:
                    heavy_counts += 1

            fraction = heavy_counts / n_trials
            qv_data[m] = fraction
            if fraction > 2.0 / 3.0:
                best_m = m

        qv = 2 ** best_m

        return BenchmarkResult(
            name="Quantum Volume",
            num_qubits=max_m,
            metric_name="quantum_volume",
            metric_value=qv,
            raw_data={
                "per_depth_pass_rate": qv_data,
                "best_depth": best_m,
            },
        )

    # ------------------------------------------------------------------
    # Cross-Entropy Benchmarking (XEB)
    # ------------------------------------------------------------------

    def xeb_fidelity(
        self,
        num_qubits: int = 2,
        depth: int = 10,
        n_circuits: int = 10,
        shots: int = 256,
        execution_mode: str = "noisy",
    ) -> BenchmarkResult:
        """Estimate cross-entropy benchmarking fidelity.

        Compares noisy circuit output to ideal distribution to estimate
        the linear XEB fidelity.
        """
        xeb_values = []

        for _ in range(n_circuits):
            # Build random circuit
            ideal_sim = TransmonSimulator(self.config, execution_mode="ideal")
            noisy_sim = TransmonSimulator(self.config, execution_mode=execution_mode)

            for layer in range(depth):
                for q in range(num_qubits):
                    angle_x = random.uniform(0, 2 * math.pi)
                    angle_z = random.uniform(0, 2 * math.pi)
                    ideal_sim.rx(q, angle_x)
                    ideal_sim.rz(q, angle_z)
                    noisy_sim.rx(q, angle_x)
                    noisy_sim.rz(q, angle_z)
                # Entangling layer
                for q in range(0, num_qubits - 1, 2 if layer % 2 == 0 else 1):
                    if q + 1 < num_qubits:
                        ideal_sim.cz(q, q + 1)
                        noisy_sim.cz(q, q + 1)

            ideal_probs = ideal_sim.probabilities()
            noisy_probs = noisy_sim.probabilities()

            # Linear XEB: F_xeb = 2^n * sum(p_ideal * p_noisy) - 1
            dim = 2 ** num_qubits
            xeb_f = dim * float(np.sum(ideal_probs * noisy_probs)) - 1.0
            xeb_values.append(xeb_f)

        mean_xeb = float(np.mean(xeb_values))

        return BenchmarkResult(
            name="Cross-Entropy Benchmarking",
            num_qubits=num_qubits,
            metric_name="xeb_fidelity",
            metric_value=mean_xeb,
            raw_data={
                "per_circuit_xeb": xeb_values,
                "depth": depth,
                "n_circuits": n_circuits,
            },
        )


# ------------------------------------------------------------------
# Cross-backend comparison utilities
# ------------------------------------------------------------------

def compare_backends(
    circuit_fn,
    configs: dict[str, ChipConfig],
    execution_modes: list[str] | None = None,
    shots: int = 1024,
) -> dict[str, dict[str, Any]]:
    """Run the same circuit on multiple backend configurations.

    Parameters
    ----------
    circuit_fn : callable
        Function that takes a TransmonSimulator and applies gates.
    configs : dict
        Mapping of name -> ChipConfig for each backend to test.
    execution_modes : list
        Modes to test (default: ['ideal', 'noisy']).
    shots : int
        Number of measurement shots.

    Returns
    -------
    dict
        Results keyed by config name, then by mode.
    """
    modes = execution_modes or ["ideal", "noisy"]
    results = {}

    for name, config in configs.items():
        results[name] = {}
        for mode in modes:
            sim = TransmonSimulator(config, execution_mode=mode)
            circuit_fn(sim)
            probs = sim.probabilities()
            stats = sim.circuit_stats()
            counts = sim.measure_all(shots=shots)
            results[name][mode] = {
                "probabilities": probs,
                "stats": stats,
                "counts": counts,
                "fidelity_estimate": stats.estimated_fidelity,
            }

    return results
