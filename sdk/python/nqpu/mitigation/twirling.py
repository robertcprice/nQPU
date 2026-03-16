"""Pauli Twirling and Randomized Compiling for quantum error mitigation.

Pauli twirling converts arbitrary coherent noise into stochastic Pauli
noise by sandwiching gates with randomly sampled Pauli frames.  This
makes the noise diagonal in the Pauli basis, which is easier to model,
characterize, and correct with techniques like PEC or ZNE.

For a two-qubit gate G, the twirling protocol finds pairs (P_before, P_after)
such that P_after . G . P_before = G (up to global phase) for every Pauli
P_before.  Averaging over all such random insertions converts any coherent
error into a Pauli channel.

Key classes:
  - :class:`PauliFrame` -- A single Pauli frame (before + after corrections).
  - :class:`PauliTwirler` -- Generates random twirled circuit instances.
  - :class:`TwirledCircuit` -- Stores base circuit and randomized variants.
  - :class:`RandomizedCompiling` -- Full randomized compiling pipeline.

References:
    - Wallman & Emerson, PRA 94, 052325 (2016)
    - Hashim et al., PRX 11, 041039 (2021)
    - Kern, Emerson, et al., QIP 4, 104 (2005)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]

# Pauli group multiplication table (result, phase_exponent)
# Phase is i^exponent: P_a * P_b = i^phase * P_result
# Using indices: I=0, X=1, Y=2, Z=3
_PAULI_MULT_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = {
    # I * P = P
    (0, 0): (0, 0), (0, 1): (1, 0), (0, 2): (2, 0), (0, 3): (3, 0),
    # X * P
    (1, 0): (1, 0), (1, 1): (0, 0), (1, 2): (3, 1), (1, 3): (2, 3),
    # Y * P
    (2, 0): (2, 0), (2, 1): (3, 3), (2, 2): (0, 0), (2, 3): (1, 1),
    # Z * P
    (3, 0): (3, 0), (3, 1): (2, 1), (3, 2): (1, 3), (3, 3): (0, 0),
}

PAULI_LABELS = ["I", "X", "Y", "Z"]
PAULI_INDEX = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _pauli_multiply(a: int, b: int) -> Tuple[int, int]:
    """Multiply two single-qubit Paulis, return (result_index, phase_exponent).

    The phase convention is: P_a * P_b = i^phase * P_result.
    """
    return _PAULI_MULT_TABLE[(a, b)]


# =====================================================================
# Twirling tables
# =====================================================================

# For CNOT: (P1 x P2) . CNOT = CNOT . (P1' x P2')
# The correction pairs for each Pauli pair before CNOT.
# Format: {(before_ctrl, before_tgt): (after_ctrl, after_tgt)}
_CNOT_TWIRL_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = {
    (0, 0): (0, 0),  # II -> II
    (0, 1): (0, 1),  # IX -> IX
    (0, 2): (3, 2),  # IY -> ZY
    (0, 3): (3, 3),  # IZ -> ZZ
    (1, 0): (1, 1),  # XI -> XX
    (1, 1): (1, 0),  # XX -> XI
    (1, 2): (2, 3),  # XY -> YZ  (phase adjusted)
    (1, 3): (2, 2),  # XZ -> YY  (phase adjusted)
    (2, 0): (2, 1),  # YI -> YX
    (2, 1): (2, 0),  # YX -> YI
    (2, 2): (1, 3),  # YY -> XZ  (phase adjusted)
    (2, 3): (1, 2),  # YZ -> XY  (phase adjusted)
    (3, 0): (3, 0),  # ZI -> ZI
    (3, 1): (3, 1),  # ZX -> ZX
    (3, 2): (0, 2),  # ZY -> IY
    (3, 3): (0, 3),  # ZZ -> IZ
}

# For CZ: (P1 x P2) . CZ = CZ . (P1' x P2')
_CZ_TWIRL_TABLE: Dict[Tuple[int, int], Tuple[int, int]] = {
    (0, 0): (0, 0),
    (0, 1): (3, 1),  # IX -> ZX
    (0, 2): (3, 2),  # IY -> ZY
    (0, 3): (0, 3),  # IZ -> IZ
    (1, 0): (1, 3),  # XI -> XZ
    (1, 1): (2, 2),  # XX -> YY  (phase adjusted)
    (1, 2): (2, 1),  # XY -> YX  (phase adjusted)
    (1, 3): (1, 0),  # XZ -> XI
    (2, 0): (2, 3),  # YI -> YZ
    (2, 1): (1, 2),  # YX -> XY  (phase adjusted)
    (2, 2): (1, 1),  # YY -> XX  (phase adjusted)
    (2, 3): (2, 0),  # YZ -> YI
    (3, 0): (3, 0),  # ZI -> ZI
    (3, 1): (0, 1),  # ZX -> IX
    (3, 2): (0, 2),  # ZY -> IY
    (3, 3): (3, 3),  # ZZ -> ZZ
}


def _get_twirl_table(
    gate_name: str,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Return the twirling correction table for a given two-qubit gate."""
    upper = gate_name.upper()
    if upper in ("CNOT", "CX"):
        return _CNOT_TWIRL_TABLE
    elif upper == "CZ":
        return _CZ_TWIRL_TABLE
    else:
        raise ValueError(
            f"No twirling table for gate '{gate_name}'. "
            "Supported: CNOT, CX, CZ."
        )


# =====================================================================
# Pauli frame
# =====================================================================


@dataclass
class PauliFrame:
    """A single Pauli frame for twirling.

    Attributes
    ----------
    before : dict
        Mapping from qubit index to Pauli label to apply before the gate.
    after : dict
        Mapping from qubit index to Pauli label to apply after the gate.
    """

    before: Dict[int, str] = field(default_factory=dict)
    after: Dict[int, str] = field(default_factory=dict)


# =====================================================================
# Twirled circuit
# =====================================================================


@dataclass
class TwirledCircuit:
    """A base circuit plus its randomized twirled variants.

    Attributes
    ----------
    base_circuit : list of Gate
        The original circuit.
    variants : list of list of Gate
        Randomized circuit variants (each is functionally equivalent
        to the base circuit in the ideal case).
    frames : list of PauliFrame
        The Pauli frames used for each variant.
    """

    base_circuit: List[Gate] = field(default_factory=list)
    variants: List[List[Gate]] = field(default_factory=list)
    frames: List[PauliFrame] = field(default_factory=list)


# =====================================================================
# Pauli twirler
# =====================================================================


class PauliTwirler:
    """Generate twirled circuit variants.

    Inserts random Pauli frames around two-qubit gates so that coherent
    noise is converted to Pauli (stochastic) noise upon averaging.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def twirl_gate(
        self, gate_name: str, ctrl: int, tgt: int
    ) -> PauliFrame:
        """Generate a random twirling frame for a two-qubit gate.

        Parameters
        ----------
        gate_name : str
            Name of the two-qubit gate ("CNOT", "CX", or "CZ").
        ctrl : int
            Control qubit.
        tgt : int
            Target qubit.

        Returns
        -------
        PauliFrame
            The before/after Pauli assignments for the twirl.
        """
        table = _get_twirl_table(gate_name)

        # Random Pauli pair before the gate
        before_ctrl = int(self.rng.integers(0, 4))
        before_tgt = int(self.rng.integers(0, 4))

        # Look up the required correction after the gate
        after_ctrl, after_tgt = table[(before_ctrl, before_tgt)]

        return PauliFrame(
            before={ctrl: PAULI_LABELS[before_ctrl], tgt: PAULI_LABELS[before_tgt]},
            after={ctrl: PAULI_LABELS[after_ctrl], tgt: PAULI_LABELS[after_tgt]},
        )

    def twirl_single_qubit(self, qubit: int) -> PauliFrame:
        """Generate a random single-qubit Pauli twirl.

        For single-qubit gates, we insert P before and P after (since
        P . G . P = G up to a commutation phase that is corrected).
        This is simpler than the two-qubit case.

        Parameters
        ----------
        qubit : int
            Target qubit.

        Returns
        -------
        PauliFrame
        """
        p_idx = int(self.rng.integers(0, 4))
        label = PAULI_LABELS[p_idx]
        return PauliFrame(
            before={qubit: label},
            after={qubit: label},  # P . G . P cancels for Clifford gates
        )

    def twirl_circuit(
        self, circuit: List[Gate], num_samples: int = 100
    ) -> TwirledCircuit:
        """Generate multiple twirled variants of a circuit.

        Only two-qubit gates (CNOT, CX, CZ) are twirled.  Single-qubit
        gates pass through unchanged.

        Parameters
        ----------
        circuit : list of Gate
            Base circuit.
        num_samples : int
            Number of randomized variants to generate.

        Returns
        -------
        TwirledCircuit
        """
        result = TwirledCircuit(base_circuit=list(circuit))

        for _ in range(num_samples):
            variant: List[Gate] = []
            frames: List[PauliFrame] = []

            for gate_name, qubits, params in circuit:
                upper = gate_name.upper()
                if upper in ("CNOT", "CX", "CZ") and len(qubits) == 2:
                    ctrl, tgt = qubits
                    frame = self.twirl_gate(gate_name, ctrl, tgt)
                    frames.append(frame)

                    # Insert Pauli before
                    for q, pauli in frame.before.items():
                        if pauli != "I":
                            variant.append((pauli, (q,), ()))

                    # Original gate
                    variant.append((gate_name, qubits, params))

                    # Insert Pauli after
                    for q, pauli in frame.after.items():
                        if pauli != "I":
                            variant.append((pauli, (q,), ()))
                else:
                    variant.append((gate_name, qubits, params))

            result.variants.append(variant)
            result.frames.extend(frames)

        return result


# =====================================================================
# Twirl-and-average
# =====================================================================


def twirl_and_average(
    circuit: List[Gate],
    executor: Callable[[List[Gate]], float],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Twirl a circuit and average results to suppress coherent noise.

    Parameters
    ----------
    circuit : list of Gate
        Circuit to twirl.
    executor : callable
        Executes a circuit and returns an expectation value.
    num_samples : int
        Number of twirled variants to average.
    seed : int or None
        Random seed.

    Returns
    -------
    (mean, std_error) : (float, float)
        Mean and standard error of the twirled estimates.
    """
    twirler = PauliTwirler(seed=seed)
    twirled = twirler.twirl_circuit(circuit, num_samples=num_samples)

    values = [executor(variant) for variant in twirled.variants]
    arr = np.array(values)
    mean = float(np.mean(arr))
    std_err = float(np.std(arr) / np.sqrt(len(arr)))
    return mean, std_err


# =====================================================================
# Randomized compiling
# =====================================================================


class RandomizedCompiling:
    """Full randomized compiling pipeline.

    Transforms a circuit into an ensemble of logically equivalent circuits
    where coherent errors have been converted to stochastic Pauli noise.
    The averaged result over the ensemble gives improved accuracy.

    Parameters
    ----------
    num_compilations : int
        Number of randomized circuit variants per execution.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        num_compilations: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        self.num_compilations = num_compilations
        self.twirler = PauliTwirler(seed=seed)

    def compile(self, circuit: List[Gate]) -> TwirledCircuit:
        """Generate an ensemble of randomized compilations.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit.

        Returns
        -------
        TwirledCircuit
            Contains base circuit and randomized variants.
        """
        return self.twirler.twirl_circuit(
            circuit, num_samples=self.num_compilations
        )

    def execute(
        self,
        circuit: List[Gate],
        executor: Callable[[List[Gate]], float],
    ) -> Tuple[float, float]:
        """Compile and execute with randomized averaging.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit.
        executor : callable
            Circuit executor returning expectation value.

        Returns
        -------
        (mean, std_error) : (float, float)
        """
        twirled = self.compile(circuit)
        values = [executor(v) for v in twirled.variants]
        arr = np.array(values)
        return float(np.mean(arr)), float(np.std(arr) / np.sqrt(len(arr)))
