"""Probabilistic Error Cancellation (PEC) for quantum error mitigation.

PEC works by expressing the inverse of a noisy quantum channel as a
quasi-probability distribution over physically implementable operations
(Pauli gates).  Monte Carlo sampling over this distribution, with
importance weighting to handle negative quasi-probabilities, yields
unbiased estimates of the ideal (noiseless) expectation value.

The sampling overhead is governed by the one-norm
gamma = sum(|alpha_i|) of the quasi-probability coefficients.  The
number of samples needed scales as O(gamma^{2L} / epsilon^2) where
L is the number of noisy gates and epsilon is the target precision.

Key classes:
  - :class:`NoiseChannel` -- Pauli channel representation.
  - :class:`PECDecomposition` -- Quasi-probability decomposition of the
    inverse channel into implementable Pauli operations.
  - :class:`PECEstimator` -- Monte Carlo estimator that samples
    corrected circuits and computes mitigated expectation values.
  - :class:`PECResult` -- Container for estimation results.

References:
    - Temme, Bravyi, Gambetta, PRL 119, 180509 (2017)
    - van den Berg et al., Nature Physics 19, 1116 (2023)
    - Endo, Benjamin, Li, PRX 8, 031027 (2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Re-use the Gate type alias from zne
Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]

# Pauli labels
PAULI_LABELS = ("I", "X", "Y", "Z")

# Single-qubit Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI_MATRICES = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


# =====================================================================
# Noise channel representation
# =====================================================================


class ChannelType(Enum):
    """Type of noise channel."""

    DEPOLARIZING = auto()
    PAULI = auto()
    AMPLITUDE_DAMPING = auto()


@dataclass
class NoiseChannel:
    """Single-qubit Pauli noise channel.

    Represents the channel:
        E(rho) = p_I * rho + p_X * X rho X + p_Y * Y rho Y + p_Z * Z rho Z

    where p_I + p_X + p_Y + p_Z = 1.

    Parameters
    ----------
    probabilities : dict
        Mapping from Pauli label ("I", "X", "Y", "Z") to probability.
    """

    probabilities: Dict[str, float] = field(
        default_factory=lambda: {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
    )

    def __post_init__(self) -> None:
        for label in PAULI_LABELS:
            if label not in self.probabilities:
                self.probabilities[label] = 0.0
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-10:
            raise ValueError(
                f"Pauli probabilities must sum to 1, got {total}"
            )
        for label, p in self.probabilities.items():
            if p < -1e-10:
                raise ValueError(
                    f"Probability for {label} is negative: {p}"
                )

    @classmethod
    def depolarizing(cls, error_rate: float) -> "NoiseChannel":
        """Create a depolarizing channel with the given error rate.

        E(rho) = (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)

        Parameters
        ----------
        error_rate : float
            Total depolarizing probability p in [0, 0.75).
        """
        if error_rate < 0.0 or error_rate >= 0.75:
            raise ValueError(
                f"Depolarizing error rate must be in [0, 0.75), got {error_rate}"
            )
        return cls(
            probabilities={
                "I": 1.0 - error_rate,
                "X": error_rate / 3.0,
                "Y": error_rate / 3.0,
                "Z": error_rate / 3.0,
            }
        )

    @classmethod
    def pauli_channel(cls, px: float, py: float, pz: float) -> "NoiseChannel":
        """Create a general Pauli channel.

        Parameters
        ----------
        px, py, pz : float
            Probabilities of X, Y, Z errors.  Must satisfy px+py+pz <= 1.
        """
        pi = 1.0 - px - py - pz
        if pi < -1e-10:
            raise ValueError("Pauli probabilities exceed 1")
        return cls(probabilities={"I": max(0.0, pi), "X": px, "Y": py, "Z": pz})

    @property
    def error_rate(self) -> float:
        """Total error probability (1 - p_I)."""
        return 1.0 - self.probabilities["I"]

    def pauli_transfer_matrix(self) -> np.ndarray:
        """Compute the 4x4 Pauli Transfer Matrix (PTM).

        The PTM Lambda satisfies:
            Lambda[i,j] = (1/2) Tr(sigma_i E(sigma_j))

        For a Pauli channel this is diagonal with eigenvalues determined
        by the Pauli probabilities.
        """
        p = self.probabilities
        # For a Pauli channel, the PTM is diagonal:
        # lambda_I = 1  (trace preservation)
        # lambda_X = p_I + p_X - p_Y - p_Z
        # lambda_Y = p_I - p_X + p_Y - p_Z
        # lambda_Z = p_I - p_X - p_Y + p_Z
        ptm = np.diag([
            1.0,
            p["I"] + p["X"] - p["Y"] - p["Z"],
            p["I"] - p["X"] + p["Y"] - p["Z"],
            p["I"] - p["X"] - p["Y"] + p["Z"],
        ])
        return ptm


# =====================================================================
# Two-qubit noise channel
# =====================================================================


@dataclass
class TwoQubitNoiseChannel:
    """Two-qubit Pauli noise channel.

    Represents the channel parameterized by 16 Pauli-pair probabilities:
        E(rho) = sum_{i,j in {I,X,Y,Z}} p_{ij} (sigma_i x sigma_j) rho (sigma_i x sigma_j)

    Parameters
    ----------
    probabilities : dict
        Mapping from (P1, P2) label pairs to probabilities.
    """

    probabilities: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.probabilities:
            # Default: identity channel
            self.probabilities = {
                (p1, p2): (1.0 if (p1 == "I" and p2 == "I") else 0.0)
                for p1 in PAULI_LABELS
                for p2 in PAULI_LABELS
            }
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-10:
            raise ValueError(
                f"Two-qubit Pauli probabilities must sum to 1, got {total}"
            )

    @classmethod
    def depolarizing(cls, error_rate: float) -> "TwoQubitNoiseChannel":
        """Two-qubit depolarizing channel.

        E(rho) = (1-p) rho + (p/15) sum_{(i,j) != (I,I)} sigma_{ij} rho sigma_{ij}
        """
        if error_rate < 0.0 or error_rate >= 15.0 / 16.0:
            raise ValueError(
                f"Two-qubit depolarizing rate must be in [0, 15/16), got {error_rate}"
            )
        probs: Dict[Tuple[str, str], float] = {}
        for p1 in PAULI_LABELS:
            for p2 in PAULI_LABELS:
                if p1 == "I" and p2 == "I":
                    probs[(p1, p2)] = 1.0 - error_rate
                else:
                    probs[(p1, p2)] = error_rate / 15.0
        return cls(probabilities=probs)


# =====================================================================
# PEC decomposition
# =====================================================================


@dataclass
class PECOperation:
    """A single operation in the PEC quasi-probability decomposition.

    Attributes
    ----------
    pauli : str
        Pauli correction label ("I", "X", "Y", "Z").
    coefficient : float
        Quasi-probability coefficient (can be negative).
    qubit : int
        Target qubit for the correction.
    """

    pauli: str
    coefficient: float
    qubit: int


@dataclass
class PECDecomposition:
    """Quasi-probability decomposition of the inverse noise channel.

    The ideal gate can be expressed as:
        G_ideal = gamma * sum_i (alpha_i / gamma) * sign_i * G_noisy . P_i

    where gamma = sum(|alpha_i|) is the one-norm and P_i are Pauli
    corrections applied after the noisy gate.

    Attributes
    ----------
    operations : list of PECOperation
        The quasi-probability distribution over Pauli corrections.
    gamma : float
        One-norm (sampling overhead factor).
    """

    operations: List[PECOperation] = field(default_factory=list)
    gamma: float = 1.0

    @classmethod
    def from_noise_channel(
        cls, channel: NoiseChannel, qubit: int
    ) -> "PECDecomposition":
        """Decompose the inverse of a single-qubit Pauli channel.

        For a Pauli channel with probabilities {p_I, p_X, p_Y, p_Z},
        the inverse channel is also a Pauli channel with coefficients
        derived from inverting the PTM eigenvalues.

        Parameters
        ----------
        channel : NoiseChannel
            The noise channel to invert.
        qubit : int
            Target qubit index.
        """
        ptm = channel.pauli_transfer_matrix()
        eigenvalues = np.diag(ptm)

        # Check invertibility: all eigenvalues must be non-zero
        for i, ev in enumerate(eigenvalues):
            if abs(ev) < 1e-12:
                raise ValueError(
                    f"Channel is not invertible: PTM eigenvalue {i} is zero"
                )

        # Inverse PTM eigenvalues
        inv_eigenvalues = 1.0 / eigenvalues

        # Convert inverse PTM diagonal to Pauli channel coefficients
        # Using the relation between PTM eigenvalues and Pauli probabilities:
        # lambda_0 = 1 (always)
        # lambda_1 = p_I + p_X - p_Y - p_Z
        # lambda_2 = p_I - p_X + p_Y - p_Z
        # lambda_3 = p_I - p_X - p_Y + p_Z
        #
        # Inverse: p_I = (1 + lambda_1 + lambda_2 + lambda_3) / 4
        #          p_X = (1 + lambda_1 - lambda_2 - lambda_3) / 4
        #          p_Y = (1 - lambda_1 + lambda_2 - lambda_3) / 4
        #          p_Z = (1 - lambda_1 - lambda_2 + lambda_3) / 4
        eta = inv_eigenvalues
        alpha_I = (1.0 + eta[1] + eta[2] + eta[3]) / 4.0
        alpha_X = (1.0 + eta[1] - eta[2] - eta[3]) / 4.0
        alpha_Y = (1.0 - eta[1] + eta[2] - eta[3]) / 4.0
        alpha_Z = (1.0 - eta[1] - eta[2] + eta[3]) / 4.0

        ops = [
            PECOperation("I", float(alpha_I), qubit),
            PECOperation("X", float(alpha_X), qubit),
            PECOperation("Y", float(alpha_Y), qubit),
            PECOperation("Z", float(alpha_Z), qubit),
        ]
        gamma = sum(abs(op.coefficient) for op in ops)

        return cls(operations=ops, gamma=gamma)

    @classmethod
    def from_depolarizing(
        cls, error_rate: float, qubit: int
    ) -> "PECDecomposition":
        """Decompose the inverse of a depolarizing channel.

        For depolarizing noise E(rho) = (1-p)rho + (p/3)(XrhoX + YrhoY + ZrhoZ):
            PTM eigenvalue: lambda = 1 - 4p/3
            Inverse eigenvalue: eta = 1/(1 - 4p/3)
            alpha_I = (1 + 3*eta) / 4
            alpha_{X,Y,Z} = (1 - eta) / 4

        Parameters
        ----------
        error_rate : float
            Depolarizing probability in [0, 0.75).
        qubit : int
            Target qubit.
        """
        channel = NoiseChannel.depolarizing(error_rate)
        return cls.from_noise_channel(channel, qubit)

    def sample_correction(
        self, rng: Optional[np.random.Generator] = None
    ) -> Tuple[str, float]:
        """Sample a Pauli correction from the quasi-probability distribution.

        Returns
        -------
        (pauli_label, sign) : (str, float)
            The sampled Pauli and the sign (+1 or -1) for importance weighting.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample proportional to |coefficient|
        abs_coeffs = np.array([abs(op.coefficient) for op in self.operations])
        probs = abs_coeffs / self.gamma
        idx = rng.choice(len(self.operations), p=probs)
        op = self.operations[idx]
        sign = 1.0 if op.coefficient >= 0 else -1.0
        return op.pauli, sign

    def cost_estimate(self, circuit_depth: int) -> float:
        """Estimate the sampling overhead for a circuit of given depth.

        The number of samples needed scales as gamma^(2*depth) / epsilon^2.
        This returns gamma^(2*depth), the multiplicative overhead.

        Parameters
        ----------
        circuit_depth : int
            Number of noisy gates in the circuit.
        """
        return self.gamma ** (2 * circuit_depth)


# =====================================================================
# PEC result
# =====================================================================


@dataclass
class PECResult:
    """Result of probabilistic error cancellation.

    Attributes
    ----------
    estimated_value : float
        Mitigated expectation value.
    std_error : float
        Standard error of the Monte Carlo estimate.
    num_samples : int
        Number of Monte Carlo samples used.
    gamma : float
        One-norm of the quasi-probability distribution.
    sign_ratio : float
        Fraction of samples with positive sign products.
    raw_values : list of float
        Individual signed, weighted sample values.
    """

    estimated_value: float
    std_error: float = 0.0
    num_samples: int = 0
    gamma: float = 1.0
    sign_ratio: float = 1.0
    raw_values: List[float] = field(default_factory=list)


# =====================================================================
# PEC estimator
# =====================================================================


class PECEstimator:
    """Monte Carlo estimator for Probabilistic Error Cancellation.

    Parameters
    ----------
    noise_channel : NoiseChannel
        The per-gate noise channel.
    num_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        noise_channel: NoiseChannel,
        num_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        self.noise_channel = noise_channel
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def estimate(
        self,
        circuit: List[Gate],
        executor: Callable[[List[Gate]], float],
    ) -> PECResult:
        """Estimate the mitigated expectation value via PEC sampling.

        For each sample:
        1. Walk through the circuit gate by gate.
        2. After each gate, sample a Pauli correction from the PEC
           decomposition and insert it into the circuit.
        3. Track the accumulated sign from the quasi-probabilities.
        4. Execute the modified circuit and weight the result.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit.
        executor : callable
            Function that executes a circuit and returns an expectation value.

        Returns
        -------
        PECResult
        """
        # Build per-qubit decompositions
        qubits = set()
        for _, q, _ in circuit:
            for qi in q:
                qubits.add(qi)

        decompositions = {
            q: PECDecomposition.from_noise_channel(self.noise_channel, q)
            for q in qubits
        }

        gamma_per_gate = list(decompositions.values())[0].gamma
        num_gates = len(circuit)
        gamma_total = gamma_per_gate ** num_gates

        sample_values: List[float] = []
        positive_signs = 0

        for _ in range(self.num_samples):
            modified_circuit: List[Gate] = []
            sign_product = 1.0

            for gate_name, gate_qubits, gate_params in circuit:
                # Add the original gate
                modified_circuit.append((gate_name, gate_qubits, gate_params))

                # Sample a PEC correction for the first target qubit
                target_q = gate_qubits[0]
                decomp = decompositions[target_q]
                pauli, sign = decomp.sample_correction(self.rng)
                sign_product *= sign

                # Insert the Pauli correction gate (skip identity)
                if pauli != "I":
                    modified_circuit.append(
                        (pauli, (target_q,), ())
                    )

            # Execute and weight
            raw_value = executor(modified_circuit)
            weighted = gamma_total * sign_product * raw_value
            sample_values.append(weighted)

            if sign_product > 0:
                positive_signs += 1

        values_arr = np.array(sample_values)
        estimated = float(np.mean(values_arr))
        std_err = float(np.std(values_arr) / np.sqrt(self.num_samples))
        sign_ratio = positive_signs / max(1, self.num_samples)

        return PECResult(
            estimated_value=estimated,
            std_error=std_err,
            num_samples=self.num_samples,
            gamma=gamma_per_gate,
            sign_ratio=sign_ratio,
            raw_values=sample_values,
        )


# =====================================================================
# Convenience function
# =====================================================================


def run_pec(
    circuit: List[Gate],
    executor: Callable[[List[Gate]], float],
    error_rate: float = 0.01,
    num_samples: int = 1000,
    seed: Optional[int] = None,
) -> PECResult:
    """One-shot PEC convenience function.

    Parameters
    ----------
    circuit : list of Gate
        Circuit to mitigate.
    executor : callable
        Executes a circuit, returns expectation value.
    error_rate : float
        Per-gate depolarizing error rate.
    num_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        Random seed.

    Returns
    -------
    PECResult
    """
    channel = NoiseChannel.depolarizing(error_rate)
    estimator = PECEstimator(channel, num_samples=num_samples, seed=seed)
    return estimator.estimate(circuit, executor)
