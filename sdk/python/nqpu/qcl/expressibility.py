"""Circuit expressibility and trainability analysis tools.

Provides quantitative measures for evaluating parameterized quantum
circuits before committing to expensive training runs:

- :class:`ExpressibilityAnalyzer` -- KL divergence from Haar-random and
  entangling capability via the Meyer-Wallach measure.
- :class:`BarrenPlateauDetector` -- gradient variance analysis for
  detecting vanishing gradients.
- :class:`EffectiveDimension` -- Fisher information-based model capacity.

These tools help answer key design questions:
  "Is this circuit expressive enough for my problem?"
  "Will the gradients vanish during training?"
  "What is the effective model capacity?"

References
----------
- Sim et al., Adv. Quantum Technol. 2, 1900070 (2019) [expressibility]
- Meyer & Wallach, J. Math. Phys. 43, 4273 (2002) [entangling measure]
- McClean et al., Nat. Commun. 9, 4812 (2018) [barren plateaus]
- Abbas et al., Nat. Comput. Sci. 1, 403 (2021) [effective dimension]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .circuits import AnsatzCircuit, ParameterizedCircuit, StatevectorSimulator
from .gradients import CostFn, ParameterShiftRule


# ------------------------------------------------------------------
# Expressibility analyzer
# ------------------------------------------------------------------


@dataclass
class ExpressibilityResult:
    """Results from expressibility analysis.

    Attributes
    ----------
    kl_divergence : float
        KL divergence from the Haar-random distribution.
        Lower values indicate higher expressibility.
    entangling_capability : float
        Meyer-Wallach entangling measure averaged over random params.
        Values near 1 indicate maximal entangling power.
    fidelity_samples : np.ndarray
        Raw fidelity samples used for KL estimation.
    """

    kl_divergence: float
    entangling_capability: float
    fidelity_samples: np.ndarray


class ExpressibilityAnalyzer:
    """Measure how well a circuit explores the Hilbert space.

    Expressibility is quantified by the KL divergence between the
    distribution of fidelities ``|<psi1|psi2>|^2`` for randomly
    parameterized states and the Haar-random distribution.

    The Haar-random fidelity distribution for an n-qubit system is
    ``P(F) = (2^n - 1)(1 - F)^(2^n - 2)``.

    Parameters
    ----------
    n_samples : int
        Number of random state pairs to sample.
    n_bins : int
        Number of histogram bins for KL estimation.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 500,
        n_bins: int = 75,
        seed: int | None = None,
    ) -> None:
        if n_samples < 10:
            raise ValueError("n_samples must be >= 10")
        self.n_samples = n_samples
        self.n_bins = n_bins
        self.rng = np.random.default_rng(seed)

    def analyze(self, ansatz: AnsatzCircuit) -> ExpressibilityResult:
        """Analyze the expressibility of an ansatz circuit.

        Parameters
        ----------
        ansatz : AnsatzCircuit
            The circuit to analyze.

        Returns
        -------
        ExpressibilityResult
        """
        n_qubits = ansatz.n_qubits
        dim = 1 << n_qubits
        n_params = ansatz.n_params
        names = ansatz.parameter_names

        # Sample fidelities
        fidelities = np.zeros(self.n_samples, dtype=np.float64)
        for k in range(self.n_samples):
            # Two random parameter sets
            p1 = self.rng.uniform(0, 2 * np.pi, size=n_params)
            p2 = self.rng.uniform(0, 2 * np.pi, size=n_params)
            bindings1 = dict(zip(names, p1))
            bindings2 = dict(zip(names, p2))
            sv1 = ansatz.run(bindings1)
            sv2 = ansatz.run(bindings2)
            fidelities[k] = abs(np.vdot(sv1, sv2)) ** 2

        # KL divergence from Haar distribution
        kl_div = self._kl_from_haar(fidelities, dim)

        # Entangling capability (Meyer-Wallach measure)
        ent_cap = self._entangling_capability(ansatz, n_params, names, n_qubits)

        return ExpressibilityResult(
            kl_divergence=kl_div,
            entangling_capability=ent_cap,
            fidelity_samples=fidelities,
        )

    def _kl_from_haar(
        self, fidelities: np.ndarray, dim: int
    ) -> float:
        """Compute KL divergence of fidelity distribution from Haar-random."""
        # Histogram the circuit fidelities
        bins = np.linspace(0, 1, self.n_bins + 1)
        hist, _ = np.histogram(fidelities, bins=bins, density=True)
        bin_width = 1.0 / self.n_bins
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        # Haar-random fidelity PDF: P(F) = (dim-1)(1-F)^(dim-2)
        haar_pdf = np.zeros(self.n_bins, dtype=np.float64)
        for i, f in enumerate(bin_centers):
            if f < 1.0:
                haar_pdf[i] = (dim - 1) * (1.0 - f) ** (dim - 2)
            else:
                haar_pdf[i] = 0.0

        # Normalize both to be valid probability distributions
        p = hist * bin_width
        q = haar_pdf * bin_width

        # Handle zeros
        eps = 1e-10
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)

        # Renormalize
        p = p / p.sum()
        q = q / q.sum()

        # KL divergence: sum p * log(p/q)
        kl = float(np.sum(p * np.log(p / q)))
        return max(0.0, kl)

    def _entangling_capability(
        self,
        ansatz: AnsatzCircuit,
        n_params: int,
        names: list[str],
        n_qubits: int,
    ) -> float:
        """Compute Meyer-Wallach entangling measure averaged over random params.

        Q(|psi>) = (2/n) sum_k (1 - tr(rho_k^2))

        where rho_k is the reduced density matrix of qubit k.
        """
        if n_qubits < 2:
            return 0.0

        n_mw_samples = min(self.n_samples, 100)
        mw_values = np.zeros(n_mw_samples, dtype=np.float64)

        for s in range(n_mw_samples):
            params = self.rng.uniform(0, 2 * np.pi, size=n_params)
            bindings = dict(zip(names, params))
            sv = ansatz.run(bindings)
            mw_values[s] = self._meyer_wallach(sv, n_qubits)

        return float(np.mean(mw_values))

    @staticmethod
    def _meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
        """Compute the Meyer-Wallach entanglement measure for a pure state."""
        dim = 1 << n_qubits
        total = 0.0

        for k in range(n_qubits):
            # Compute reduced density matrix of qubit k by tracing out the rest
            rho = np.zeros((2, 2), dtype=np.complex128)
            step = 1 << k
            for i in range(dim):
                for j in range(dim):
                    # i and j must agree on all qubits except possibly k
                    if (i ^ j) & ~step == 0:
                        # Row/col of rho_k
                        ri = (i >> k) & 1
                        ci = (j >> k) & 1
                        rho[ri, ci] += state[i] * np.conj(state[j])

            # Purity: tr(rho^2)
            purity = np.real(np.trace(rho @ rho))
            total += 1.0 - purity

        return float(2.0 * total / n_qubits)


# ------------------------------------------------------------------
# Barren plateau detector
# ------------------------------------------------------------------


@dataclass
class BarrenPlateauResult:
    """Results from barren plateau analysis.

    Attributes
    ----------
    gradient_variances : np.ndarray
        Variance of each gradient component across random initializations.
    mean_variance : float
        Average variance across all components.
    is_barren : bool
        True if mean variance < threshold (likely barren plateau).
    cost_concentration : float
        Variance of cost function values across random initializations.
    n_samples : int
        Number of random initializations used.
    """

    gradient_variances: np.ndarray
    mean_variance: float
    is_barren: bool
    cost_concentration: float
    n_samples: int


class BarrenPlateauDetector:
    """Detect barren plateaus (vanishing gradients) in parameterized circuits.

    Barren plateaus occur when gradient variance decreases exponentially
    with circuit depth, making optimization infeasible for deep circuits.
    This detector samples gradients at random parameter points and
    computes variance statistics.

    Parameters
    ----------
    n_samples : int
        Number of random parameter initializations.
    threshold : float
        Gradient variance below this is flagged as barren.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 50,
        threshold: float = 1e-4,
        seed: int | None = None,
    ) -> None:
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2")
        self.n_samples = n_samples
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def detect(
        self, cost_fn: CostFn, n_params: int
    ) -> BarrenPlateauResult:
        """Analyze a cost function for barren plateau behavior.

        Parameters
        ----------
        cost_fn : callable
            ``f(params) -> float``.
        n_params : int
            Number of parameters.

        Returns
        -------
        BarrenPlateauResult
        """
        psr = ParameterShiftRule()
        grads = np.zeros((self.n_samples, n_params), dtype=np.float64)
        costs = np.zeros(self.n_samples, dtype=np.float64)

        for k in range(self.n_samples):
            theta = self.rng.uniform(0, 2 * np.pi, size=n_params)
            result = psr.compute(cost_fn, theta)
            grads[k] = result.gradient
            costs[k] = cost_fn(theta)

        variances = np.var(grads, axis=0)
        mean_var = float(np.mean(variances))
        cost_var = float(np.var(costs))

        return BarrenPlateauResult(
            gradient_variances=variances,
            mean_variance=mean_var,
            is_barren=mean_var < self.threshold,
            cost_concentration=cost_var,
            n_samples=self.n_samples,
        )

    def compare_depths(
        self,
        make_cost_fn: Callable[[int], CostFn],
        n_params_fn: Callable[[int], int],
        depths: list[int],
    ) -> dict[int, BarrenPlateauResult]:
        """Compare barren plateau behavior across circuit depths.

        Parameters
        ----------
        make_cost_fn : callable
            ``make_cost_fn(depth) -> cost_fn`` factory.
        n_params_fn : callable
            ``n_params_fn(depth) -> int`` giving param count for each depth.
        depths : list[int]
            Circuit depths to test.

        Returns
        -------
        dict[int, BarrenPlateauResult]
            Results keyed by depth.
        """
        results = {}
        for d in depths:
            cost_fn = make_cost_fn(d)
            n_params = n_params_fn(d)
            results[d] = self.detect(cost_fn, n_params)
        return results


# ------------------------------------------------------------------
# Effective dimension
# ------------------------------------------------------------------


class EffectiveDimension:
    """Fisher information-based effective dimension measure.

    Quantifies the model capacity of a parameterized quantum circuit
    by analyzing the spectrum of the empirical Fisher information matrix.
    Higher effective dimension indicates greater model expressiveness.

    Parameters
    ----------
    n_samples : int
        Number of data/parameter samples for Fisher estimation.
    seed : int or None
        Random seed.
    """

    def __init__(
        self, n_samples: int = 100, seed: int | None = None
    ) -> None:
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def compute(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        n_params: int,
        X: np.ndarray,
        n_data: int | None = None,
    ) -> dict[str, float | np.ndarray]:
        """Compute the effective dimension.

        Parameters
        ----------
        model_fn : callable
            ``model_fn(x, params) -> output_vector``.  The output should
            be a probability vector or prediction vector.
        n_params : int
            Number of model parameters.
        X : np.ndarray
            Sample inputs (n_samples, n_features).
        n_data : int or None
            Dataset size for normalization (defaults to len(X)).

        Returns
        -------
        dict
            Keys:

            - ``'effective_dimension'``: scalar effective dimension
            - ``'normalized_dimension'``: effective_dim / n_params
            - ``'fisher_eigenvalues'``: eigenvalues of the Fisher matrix
        """
        X = np.asarray(X, dtype=np.float64)
        if n_data is None:
            n_data = len(X)

        # Sample random parameters
        n_param_samples = min(self.n_samples, 50)
        fisher_sum = np.zeros((n_params, n_params), dtype=np.float64)

        for _ in range(n_param_samples):
            theta = self.rng.uniform(0, 2 * np.pi, size=n_params)
            F_theta = self._empirical_fisher(model_fn, theta, X)
            fisher_sum += F_theta

        F_avg = fisher_sum / n_param_samples

        # Eigenvalues of the average Fisher
        eigvals = np.linalg.eigvalsh(F_avg)
        eigvals = np.clip(eigvals, 0, None)

        # Effective dimension formula (Abbas et al., 2021)
        # d_eff = 2 * log(det(I + n*F/(2*pi*ln(n)))) / log(n)
        gamma = n_data / (2.0 * np.pi * max(np.log(n_data), 1.0))
        log_terms = np.log(1.0 + gamma * eigvals)
        log_det = np.sum(log_terms)
        log_n = max(np.log(n_data), 1.0)
        eff_dim = 2.0 * log_det / log_n

        return {
            "effective_dimension": float(eff_dim),
            "normalized_dimension": float(eff_dim / max(n_params, 1)),
            "fisher_eigenvalues": eigvals,
        }

    def _empirical_fisher(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        params: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        """Estimate the empirical Fisher information matrix.

        Uses the outer product of gradients of the log-likelihood.
        """
        n_params = len(params)
        n_data = len(X)
        eps = 1e-5
        F = np.zeros((n_params, n_params), dtype=np.float64)

        for xi in X:
            # Compute gradient of log-output via finite differences
            out0 = np.asarray(model_fn(xi, params), dtype=np.float64).ravel()
            out0 = np.clip(out0, 1e-10, None)
            log_out0 = np.log(out0)

            grad_log = np.zeros(n_params, dtype=np.float64)
            for p in range(n_params):
                shifted = params.copy()
                shifted[p] += eps
                out_p = np.asarray(model_fn(xi, shifted), dtype=np.float64).ravel()
                out_p = np.clip(out_p, 1e-10, None)
                log_out_p = np.log(out_p)
                # Sum of differences in log-probs (averaged over outputs)
                grad_log[p] = np.mean(log_out_p - log_out0) / eps

            F += np.outer(grad_log, grad_log)

        return F / max(n_data, 1)
