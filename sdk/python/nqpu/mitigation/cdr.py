"""Clifford Data Regression (CDR) for quantum error mitigation.

CDR learns a noise correction model from near-Clifford circuits where
the ideal expectation value can be computed classically (Clifford
circuits can be simulated in polynomial time via the stabilizer
formalism).  The learned model is then applied to correct noisy results
from arbitrary circuits.

The training procedure:
1. Take the target circuit and replace each non-Clifford gate with the
   nearest Clifford gate to produce "training circuits".
2. Compute the ideal expectation value of each training circuit
   classically.
3. Execute each training circuit on the noisy hardware/simulator.
4. Fit a regression model (linear or polynomial) mapping noisy -> ideal.
5. Apply the learned correction to the noisy result of the original
   circuit.

Key classes:
  - :class:`CDRModel` -- Fitted correction model (linear or polynomial).
  - :class:`CDREstimator` -- Full CDR training and correction pipeline.
  - :class:`CDRResult` -- Container for corrected results.

References:
    - Czarnik et al., Quantum 5, 592 (2021)
    - Lowe et al., PRR 3, 033098 (2021)
    - Strikis et al., PRX Quantum 2, 040330 (2021)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Gate = Tuple[str, Tuple[int, ...], Tuple[float, ...]]

# =====================================================================
# Clifford gate utilities
# =====================================================================

# The single-qubit Clifford gates in our gate set
CLIFFORD_GATES = frozenset({
    "I", "X", "Y", "Z", "H", "S", "SDG", "CNOT", "CX", "CZ", "SWAP",
})

# Nearest Clifford substitutions for common rotation gates.
# For Rz(theta), the nearest Clifford is the closest multiple of pi/2.
# For Rx(theta), same principle.
_CLIFFORD_RZ_ANGLES = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
_CLIFFORD_RZ_GATES = [
    ("I", ()),        # Rz(0) = I
    ("S", ()),        # Rz(pi/2) = S
    ("Z", ()),        # Rz(pi) = Z
    ("SDG", ()),      # Rz(3pi/2) = Sdg
    ("I", ()),        # Rz(2pi) = I
]


def _nearest_clifford_rz(theta: float) -> Tuple[str, Tuple[float, ...]]:
    """Find the nearest Clifford gate for Rz(theta).

    Returns (gate_name, params).
    """
    # Normalize angle to [0, 2pi)
    theta_mod = theta % (2 * np.pi)
    best_idx = 0
    best_dist = float("inf")
    for i, cliff_angle in enumerate(_CLIFFORD_RZ_ANGLES):
        dist = abs(theta_mod - cliff_angle)
        dist = min(dist, 2 * np.pi - dist)  # Handle wrap-around
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return _CLIFFORD_RZ_GATES[best_idx]


def _nearest_clifford_rx(theta: float) -> Tuple[str, Tuple[float, ...]]:
    """Find the nearest Clifford gate for Rx(theta)."""
    theta_mod = theta % (2 * np.pi)
    # Nearest Clifford: 0 -> I, pi/2 -> (H.S.H), pi -> X, 3pi/2 -> (H.SDG.H)
    # Simplified: just map to I or X for the nearest
    candidates = [
        (0.0, "I", ()),
        (np.pi, "X", ()),
    ]
    best_name, best_params = "I", ()
    best_dist = float("inf")
    for angle, name, params in candidates:
        dist = abs(theta_mod - angle)
        dist = min(dist, 2 * np.pi - dist)
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_params = params
    return best_name, best_params


def _nearest_clifford_ry(theta: float) -> Tuple[str, Tuple[float, ...]]:
    """Find the nearest Clifford gate for Ry(theta)."""
    theta_mod = theta % (2 * np.pi)
    candidates = [
        (0.0, "I", ()),
        (np.pi, "Y", ()),
    ]
    best_name, best_params = "I", ()
    best_dist = float("inf")
    for angle, name, params in candidates:
        dist = abs(theta_mod - angle)
        dist = min(dist, 2 * np.pi - dist)
        if dist < best_dist:
            best_dist = dist
            best_name = name
            best_params = params
    return best_name, best_params


def replace_non_clifford(circuit: List[Gate]) -> List[Gate]:
    """Replace non-Clifford gates with their nearest Clifford equivalents.

    Parameters
    ----------
    circuit : list of Gate
        Original circuit.

    Returns
    -------
    list of Gate
        Near-Clifford circuit.
    """
    clifford_circuit: List[Gate] = []
    for name, qubits, params in circuit:
        upper = name.upper()
        if upper in CLIFFORD_GATES:
            clifford_circuit.append((name, qubits, params))
        elif upper == "RZ" and params:
            cliff_name, cliff_params = _nearest_clifford_rz(params[0])
            clifford_circuit.append((cliff_name, qubits, cliff_params))
        elif upper == "RX" and params:
            cliff_name, cliff_params = _nearest_clifford_rx(params[0])
            clifford_circuit.append((cliff_name, qubits, cliff_params))
        elif upper == "RY" and params:
            cliff_name, cliff_params = _nearest_clifford_ry(params[0])
            clifford_circuit.append((cliff_name, qubits, cliff_params))
        elif upper in ("T", "TDG"):
            # T is close to S/Sdg or I depending on context
            # Nearest Clifford for T: S (pi/4 -> pi/2)
            clifford_circuit.append(("S" if upper == "T" else "SDG", qubits, ()))
        elif upper in ("P", "PHASE", "U1") and params:
            cliff_name, cliff_params = _nearest_clifford_rz(params[0])
            clifford_circuit.append((cliff_name, qubits, cliff_params))
        else:
            # Unknown gate -- keep as-is (best effort)
            clifford_circuit.append((name, qubits, params))
    return clifford_circuit


# =====================================================================
# CDR model
# =====================================================================


@dataclass
class CDRTrainingPoint:
    """A single training data point.

    Attributes
    ----------
    ideal_value : float
        Classically computed expectation value of the Clifford circuit.
    noisy_value : float
        Value obtained from noisy execution.
    """

    ideal_value: float
    noisy_value: float


@dataclass
class CDRModel:
    """Fitted CDR correction model.

    For linear regression: y_ideal = slope * y_noisy + intercept.
    For polynomial: y_ideal = sum(coeffs[k] * y_noisy^k).

    Attributes
    ----------
    coefficients : np.ndarray
        Regression coefficients (highest degree first, numpy polyfit convention).
    degree : int
        Polynomial degree (1 for linear).
    training_data : list of CDRTrainingPoint
        Training points used for fitting.
    r_squared : float
        R-squared goodness of fit.
    """

    coefficients: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    degree: int = 1
    training_data: List[CDRTrainingPoint] = field(default_factory=list)
    r_squared: float = 0.0

    @classmethod
    def train(
        cls,
        ideal_values: Sequence[float],
        noisy_values: Sequence[float],
        degree: int = 1,
    ) -> "CDRModel":
        """Fit a CDR model from paired ideal/noisy data.

        Parameters
        ----------
        ideal_values : sequence of float
            Classically computed ideal expectation values.
        noisy_values : sequence of float
            Noisy expectation values from hardware/simulator.
        degree : int
            Polynomial regression degree.  1 = linear.

        Returns
        -------
        CDRModel
        """
        x = np.array(noisy_values, dtype=np.float64)
        y = np.array(ideal_values, dtype=np.float64)

        if len(x) != len(y):
            raise ValueError(
                f"Ideal and noisy arrays must have equal length, "
                f"got {len(y)} and {len(x)}"
            )
        if len(x) == 0:
            raise ValueError("Training data must not be empty")

        # Clamp degree to available data
        degree = min(degree, len(x) - 1)
        degree = max(degree, 0)

        # Filter out NaN/Inf values
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) == 0:
            raise ValueError("No valid (finite) training data points")

        # Re-clamp degree after filtering
        degree = min(degree, len(x) - 1)
        degree = max(degree, 0)

        if degree == 0 or len(x) == 1:
            # Constant correction: use mean ideal
            coeffs = np.array([float(np.mean(y))])
            r_sq = 0.0
        else:
            # Check for degenerate x values (all identical)
            x_range = float(np.max(x) - np.min(x))
            if x_range < 1e-15:
                # All noisy values identical -- fall back to mean correction
                coeffs = np.array([float(np.mean(y))])
                degree = 0
                r_sq = 0.0
            else:
                try:
                    coeffs = np.polyfit(x, y, degree)
                except (np.linalg.LinAlgError, ValueError):
                    # SVD failed -- fall back to linear or constant
                    if degree > 1:
                        try:
                            coeffs = np.polyfit(x, y, 1)
                            degree = 1
                        except (np.linalg.LinAlgError, ValueError):
                            coeffs = np.array([float(np.mean(y))])
                            degree = 0
                    else:
                        coeffs = np.array([float(np.mean(y))])
                        degree = 0
                # R-squared
                predicted = np.polyval(coeffs, x)
                ss_res = float(np.sum((y - predicted) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

        training_points = [
            CDRTrainingPoint(ideal_value=float(yi), noisy_value=float(xi))
            for xi, yi in zip(x, y)
        ]

        return cls(
            coefficients=coeffs,
            degree=degree,
            training_data=training_points,
            r_squared=float(max(0.0, min(1.0, r_sq))),
        )

    def correct(self, noisy_value: float) -> float:
        """Apply the learned correction to a noisy value.

        Parameters
        ----------
        noisy_value : float
            Noisy expectation value to correct.

        Returns
        -------
        float
            Mitigated expectation value.
        """
        return float(np.polyval(self.coefficients, noisy_value))

    @property
    def slope(self) -> float:
        """Linear slope (only meaningful for degree-1 models)."""
        if self.degree >= 1 and len(self.coefficients) >= 2:
            return float(self.coefficients[-2])
        return 1.0

    @property
    def intercept(self) -> float:
        """Linear intercept (constant term)."""
        return float(self.coefficients[-1])


# =====================================================================
# CDR result
# =====================================================================


@dataclass
class CDRResult:
    """Result of Clifford Data Regression correction.

    Attributes
    ----------
    corrected_value : float
        Mitigated expectation value.
    raw_noisy_value : float
        Original noisy value before correction.
    model : CDRModel
        The fitted correction model.
    num_training_circuits : int
        Number of training circuits used.
    """

    corrected_value: float
    raw_noisy_value: float
    model: CDRModel = field(default_factory=CDRModel)
    num_training_circuits: int = 0


# =====================================================================
# CDR estimator
# =====================================================================


class CDREstimator:
    """Full Clifford Data Regression pipeline.

    Generates near-Clifford training circuits, collects ideal and noisy
    values, fits a correction model, and applies it.

    Parameters
    ----------
    num_training_circuits : int
        Number of near-Clifford circuits to generate for training.
    degree : int
        Polynomial degree for the regression model.
    seed : int or None
        Random seed for generating training circuit variants.
    """

    def __init__(
        self,
        num_training_circuits: int = 20,
        degree: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.num_training_circuits = num_training_circuits
        self.degree = degree
        self.rng = np.random.default_rng(seed)

    def generate_training_circuits(
        self, circuit: List[Gate]
    ) -> List[List[Gate]]:
        """Generate near-Clifford training circuits.

        For each training circuit, randomly select a subset of non-Clifford
        gates to replace with their nearest Clifford equivalents.  The
        fully-Clifford version is always included as the first circuit.

        Parameters
        ----------
        circuit : list of Gate
            Original circuit.

        Returns
        -------
        list of list of Gate
            Training circuits.
        """
        training_circuits: List[List[Gate]] = []

        # Always include the fully Clifford version
        training_circuits.append(replace_non_clifford(circuit))

        # Find non-Clifford gate indices
        non_cliff_indices: List[int] = []
        for i, (name, _, _) in enumerate(circuit):
            if name.upper() not in CLIFFORD_GATES:
                non_cliff_indices.append(i)

        if not non_cliff_indices:
            # All gates are already Clifford -- replicate
            for _ in range(self.num_training_circuits - 1):
                training_circuits.append(list(circuit))
            return training_circuits

        # Generate variants by randomly replacing subsets
        for _ in range(self.num_training_circuits - 1):
            variant = list(circuit)
            # Randomly choose which non-Clifford gates to replace
            mask = self.rng.random(len(non_cliff_indices)) < 0.5
            for j, idx in enumerate(non_cliff_indices):
                if mask[j]:
                    name, qubits, params = variant[idx]
                    cliff = replace_non_clifford([(name, qubits, params)])
                    variant[idx] = cliff[0]
            training_circuits.append(variant)

        return training_circuits

    def estimate(
        self,
        circuit: List[Gate],
        noisy_executor: Callable[[List[Gate]], float],
        ideal_executor: Callable[[List[Gate]], float],
    ) -> CDRResult:
        """Run the full CDR pipeline.

        Parameters
        ----------
        circuit : list of Gate
            Target circuit to mitigate.
        noisy_executor : callable
            Executes a circuit on noisy hardware/simulator.
        ideal_executor : callable
            Computes the ideal expectation value (must handle Clifford
            circuits efficiently).

        Returns
        -------
        CDRResult
        """
        # Generate training circuits
        training_circuits = self.generate_training_circuits(circuit)

        # Collect training data
        ideal_values: List[float] = []
        noisy_values: List[float] = []

        for tc in training_circuits:
            ideal_values.append(ideal_executor(tc))
            noisy_values.append(noisy_executor(tc))

        # Fit model
        model = CDRModel.train(ideal_values, noisy_values, degree=self.degree)

        # Apply correction to the noisy result of the original circuit
        raw_noisy = noisy_executor(circuit)
        corrected = model.correct(raw_noisy)

        return CDRResult(
            corrected_value=corrected,
            raw_noisy_value=raw_noisy,
            model=model,
            num_training_circuits=len(training_circuits),
        )


# =====================================================================
# Convenience function
# =====================================================================


def cdr_correct(
    noisy_value: float,
    ideal_values: Sequence[float],
    noisy_values: Sequence[float],
    degree: int = 1,
) -> float:
    """Apply CDR correction using pre-collected training data.

    Parameters
    ----------
    noisy_value : float
        Noisy result to correct.
    ideal_values : sequence of float
        Ideal values from training circuits.
    noisy_values : sequence of float
        Noisy values from training circuits.
    degree : int
        Polynomial regression degree.

    Returns
    -------
    float
        Corrected value.
    """
    model = CDRModel.train(ideal_values, noisy_values, degree=degree)
    return model.correct(noisy_value)
