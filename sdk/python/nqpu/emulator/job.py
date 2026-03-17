"""Job submission and result types for QPU emulation.

Defines the data structures returned by :meth:`QPU.run`:

- :class:`Counts` -- measurement outcome histogram with utility methods.
- :class:`EmulatorResult` -- full result including fidelity estimate, timing,
  and optional statevector.
- :class:`Job` -- wrapper carrying status, result, and error information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class Counts(dict):
    """Measurement counts dictionary with utility methods.

    Inherits from ``dict[str, int]`` where keys are bitstrings
    (e.g. ``"010"``) and values are shot counts.

    Examples
    --------
    >>> c = Counts({"00": 480, "11": 520})
    >>> c.total_shots
    1000
    >>> c.most_probable()
    '11'
    """

    @property
    def total_shots(self) -> int:
        """Total number of measurement shots."""
        return sum(self.values())

    def probabilities(self) -> dict[str, float]:
        """Convert counts to a probability distribution.

        Returns
        -------
        dict[str, float]
            Mapping from bitstring to empirical probability.
        """
        total = self.total_shots
        if total == 0:
            return {}
        return {k: v / total for k, v in self.items()}

    def most_probable(self) -> str:
        """Return the bitstring with the highest count.

        Returns
        -------
        str
            Most frequently observed bitstring.

        Raises
        ------
        ValueError
            If the counts dictionary is empty.
        """
        if not self:
            raise ValueError("Cannot find most probable outcome of empty counts")
        return max(self, key=self.get)  # type: ignore[arg-type]

    def marginal(self, qubits: list[int], n_qubits: int) -> Counts:
        """Marginalize counts over a subset of qubits.

        Parameters
        ----------
        qubits : list[int]
            Qubit indices to keep (0-indexed from MSB in bitstring).
        n_qubits : int
            Total number of qubits in the original bitstrings.

        Returns
        -------
        Counts
            New counts dictionary marginalized over the selected qubits.
        """
        marginal: Counts = Counts()
        for bitstring, count in self.items():
            bits = [bitstring[q] for q in qubits]
            key = "".join(bits)
            marginal[key] = marginal.get(key, 0) + count
        return marginal

    def entropy(self) -> float:
        """Shannon entropy of the empirical distribution (in bits).

        Returns
        -------
        float
            Entropy value.  Zero for a deterministic distribution.
        """
        probs = self.probabilities()
        if not probs:
            return 0.0
        h = 0.0
        for p in probs.values():
            if p > 0:
                h -= p * np.log2(p)
        return float(h)


@dataclass
class EmulatorResult:
    """Result of a QPU emulation run.

    Attributes
    ----------
    counts : Counts
        Measurement outcome histogram.
    statevector : np.ndarray or None
        Final statevector (only populated when ``shots=0``).
    fidelity_estimate : float
        Estimated circuit fidelity based on hardware error rates.
    circuit_depth : int
        Depth of the executed circuit.
    native_gate_count : int
        Total native gate count after decomposition.
    estimated_runtime_us : float
        Estimated hardware wall-clock time in microseconds.
    hardware_profile : str
        Name of the hardware profile used.
    metadata : dict
        Additional backend-specific metadata.
    """

    counts: Counts
    statevector: np.ndarray | None = None
    fidelity_estimate: float = 1.0
    circuit_depth: int = 0
    native_gate_count: int = 0
    estimated_runtime_us: float = 0.0
    hardware_profile: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_qubits(self) -> int:
        """Infer the number of qubits from the result data."""
        if self.statevector is not None:
            return int(np.log2(len(self.statevector)))
        if self.counts:
            return len(next(iter(self.counts)))
        return 0


@dataclass
class Job:
    """A submitted QPU emulation job.

    Attributes
    ----------
    job_id : str
        Unique identifier for the job.
    status : str
        Job status: ``"completed"`` or ``"failed"``.
    result : EmulatorResult or None
        Result data if the job completed successfully.
    error : str or None
        Error message if the job failed.
    """

    job_id: str
    status: str = "completed"
    result: EmulatorResult | None = None
    error: str | None = None

    def successful(self) -> bool:
        """Return True if the job completed without errors."""
        return self.status == "completed" and self.result is not None
