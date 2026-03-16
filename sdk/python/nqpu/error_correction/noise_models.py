"""QEC-specific noise models for threshold studies and decoder benchmarking.

Provides noise channels that produce Pauli errors in symplectic form,
suitable for use with the codes and decoders in this package.

Noise models:
  - :class:`DepolarizingNoise` -- symmetric X/Y/Z errors with equal rate.
  - :class:`PhenomenologicalNoise` -- data errors + syndrome measurement errors.
  - :class:`CircuitLevelNoise` -- full circuit noise with noisy ancillas.
  - :class:`BiasedNoise` -- asymmetric X vs Z error rates (e.g. for XZZX codes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# Abstract noise model
# ------------------------------------------------------------------ #

class NoiseModel(ABC):
    """Abstract base for QEC noise models."""

    @abstractmethod
    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """Sample a random Pauli error on n qubits.

        Returns a length-2n binary vector in symplectic form ``[x | z]``.
        """
        ...

    @abstractmethod
    def sample_syndrome(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample an error and its (possibly noisy) syndrome.

        Returns
        -------
        error : np.ndarray
            The actual data error (length 2n, symplectic form).
        syndrome : np.ndarray
            The (possibly noisy) syndrome.
        """
        ...


# ------------------------------------------------------------------ #
# Depolarizing Noise
# ------------------------------------------------------------------ #

class DepolarizingNoise(NoiseModel):
    """Symmetric depolarizing noise channel.

    Each qubit independently suffers an X, Y, or Z error with probability
    ``p/3`` each (total error probability ``p``).

    Parameters
    ----------
    p : float
        Total depolarizing error probability per qubit (0 <= p <= 1).
    """

    def __init__(self, p: float = 0.01) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Error probability must be in [0, 1], got {p}")
        self.p = p

    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        p_each = self.p / 3.0
        error = np.zeros(2 * n, dtype=np.int8)
        for q in range(n):
            r = rng.random()
            if r < p_each:
                error[q] = 1  # X
            elif r < 2 * p_each:
                error[n + q] = 1  # Z
            elif r < 3 * p_each:
                error[q] = 1  # Y = XZ
                error[n + q] = 1
        return error

    def sample_syndrome(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        error = self.sample_error(code.n, rng)
        syndrome = code.syndrome(error)
        return error, syndrome

    def expected_error_weight(self, n: int) -> float:
        """Expected number of erroneous qubits."""
        return n * self.p


# ------------------------------------------------------------------ #
# Phenomenological Noise
# ------------------------------------------------------------------ #

class PhenomenologicalNoise(NoiseModel):
    """Phenomenological noise model with measurement errors.

    Data qubits suffer depolarizing noise with rate ``p``, and each
    syndrome bit is independently flipped with probability ``q``.

    This is the standard model for surface code threshold studies
    (Raussendorf et al. 2007).

    Parameters
    ----------
    p : float
        Data error probability per qubit.
    q : float or None
        Syndrome measurement error probability. If None, defaults to ``p``.
    num_rounds : int
        Number of syndrome measurement rounds (default 1).
    """

    def __init__(
        self, p: float = 0.01, q: Optional[float] = None, num_rounds: int = 1
    ) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Data error probability must be in [0, 1], got {p}")
        self.p = p
        self.q = q if q is not None else p
        self.num_rounds = num_rounds

    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        p_each = self.p / 3.0
        error = np.zeros(2 * n, dtype=np.int8)
        for q_idx in range(n):
            r = rng.random()
            if r < p_each:
                error[q_idx] = 1
            elif r < 2 * p_each:
                error[n + q_idx] = 1
            elif r < 3 * p_each:
                error[q_idx] = 1
                error[n + q_idx] = 1
        return error

    def sample_syndrome(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        n = code.n
        total_error = np.zeros(2 * n, dtype=np.int8)

        # Accumulate errors over rounds
        for _ in range(self.num_rounds):
            round_error = self.sample_error(n, rng)
            total_error = (total_error + round_error) % 2

        # Perfect syndrome
        syndrome = code.syndrome(total_error)

        # Add measurement noise
        noisy_syndrome = syndrome.copy()
        for i in range(len(noisy_syndrome)):
            if rng.random() < self.q:
                noisy_syndrome[i] = (noisy_syndrome[i] + 1) % 2

        return total_error, noisy_syndrome

    def sample_multi_round(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Sample multiple rounds of noisy syndromes.

        Returns
        -------
        total_error : np.ndarray
            Accumulated data error over all rounds.
        syndrome_history : list of np.ndarray
            Noisy syndrome for each round.
        """
        if rng is None:
            rng = np.random.default_rng()

        n = code.n
        total_error = np.zeros(2 * n, dtype=np.int8)
        syndrome_history = []

        for _ in range(self.num_rounds):
            round_error = self.sample_error(n, rng)
            total_error = (total_error + round_error) % 2

            syndrome = code.syndrome(total_error)
            noisy = syndrome.copy()
            for i in range(len(noisy)):
                if rng.random() < self.q:
                    noisy[i] = (noisy[i] + 1) % 2
            syndrome_history.append(noisy)

        return total_error, syndrome_history


# ------------------------------------------------------------------ #
# Circuit-Level Noise
# ------------------------------------------------------------------ #

class CircuitLevelNoise(NoiseModel):
    """Circuit-level noise model with noisy syndrome extraction.

    Models the full syndrome extraction circuit including:
      - Data qubit errors (depolarizing with rate ``p_data``)
      - CNOT gate errors (two-qubit depolarizing with rate ``p_gate``)
      - Measurement errors (readout flip with rate ``p_meas``)
      - Ancilla preparation errors (wrong eigenstate with rate ``p_prep``)

    Parameters
    ----------
    p_data : float
        Idle data qubit error rate per round.
    p_gate : float
        Two-qubit gate error rate.
    p_meas : float
        Measurement error rate.
    p_prep : float
        Ancilla preparation error rate.
    """

    def __init__(
        self,
        p_data: float = 0.001,
        p_gate: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.001,
    ) -> None:
        self.p_data = p_data
        self.p_gate = p_gate
        self.p_meas = p_meas
        self.p_prep = p_prep

    @property
    def effective_error_rate(self) -> float:
        """Approximate effective error rate combining all sources."""
        return self.p_data + self.p_gate + self.p_meas + self.p_prep

    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        # Effective per-qubit error from all sources
        p_eff = min(self.p_data + self.p_gate, 1.0)
        p_each = p_eff / 3.0
        error = np.zeros(2 * n, dtype=np.int8)
        for q in range(n):
            r = rng.random()
            if r < p_each:
                error[q] = 1
            elif r < 2 * p_each:
                error[n + q] = 1
            elif r < 3 * p_each:
                error[q] = 1
                error[n + q] = 1
        return error

    def sample_syndrome(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        n = code.n
        error = self.sample_error(n, rng)
        syndrome = code.syndrome(error)

        # Prep errors: flip some syndrome bits (wrong ancilla initialization)
        noisy_syndrome = syndrome.copy()
        for i in range(len(noisy_syndrome)):
            if rng.random() < self.p_prep:
                noisy_syndrome[i] = (noisy_syndrome[i] + 1) % 2

        # Measurement errors
        for i in range(len(noisy_syndrome)):
            if rng.random() < self.p_meas:
                noisy_syndrome[i] = (noisy_syndrome[i] + 1) % 2

        return error, noisy_syndrome


# ------------------------------------------------------------------ #
# Biased Noise
# ------------------------------------------------------------------ #

class BiasedNoise(NoiseModel):
    """Biased (asymmetric) noise channel.

    X errors occur with rate ``p_x`` and Z errors with rate ``p_z``.
    Y errors occur with rate ``p_x * p_z`` (independent X and Z).

    This model is relevant for hardware with asymmetric noise (e.g.
    cat qubits, where Z errors are exponentially suppressed) and for
    XZZX surface codes that exploit noise bias.

    Parameters
    ----------
    p_x : float
        X error probability per qubit.
    p_z : float
        Z error probability per qubit.
    """

    def __init__(self, p_x: float = 0.01, p_z: float = 0.001) -> None:
        if not (0.0 <= p_x <= 1.0 and 0.0 <= p_z <= 1.0):
            raise ValueError("Error probabilities must be in [0, 1]")
        self.p_x = p_x
        self.p_z = p_z

    @property
    def bias_ratio(self) -> float:
        """Ratio p_z / p_x (< 1 means Z-biased suppression)."""
        if self.p_x == 0:
            return float("inf")
        return self.p_z / self.p_x

    @property
    def total_error_rate(self) -> float:
        """Total probability of any error per qubit (approximate)."""
        return self.p_x + self.p_z - self.p_x * self.p_z

    def sample_error(
        self, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        error = np.zeros(2 * n, dtype=np.int8)
        for q in range(n):
            if rng.random() < self.p_x:
                error[q] = 1  # X part
            if rng.random() < self.p_z:
                error[n + q] = 1  # Z part
        return error

    def sample_syndrome(
        self,
        code,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()
        error = self.sample_error(code.n, rng)
        syndrome = code.syndrome(error)
        return error, syndrome
