"""Benchmark harness for quantum-inspired classical algorithms.

Provides a ``BenchmarkSuite`` that runs head-to-head comparisons
between the three optimiser families, the three sampling methods,
and the three linear-algebra routines in this package.  Each
convenience function (``run_optimization_benchmark``, etc.) produces
a ``BenchmarkComparison`` with an ASCII summary table and quality
ratios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .optimization import (
    IsingProblem,
    QAOAInspiredOptimizer,
    QuantumWalkOptimizer,
    SimulatedQuantumAnnealing,
)
from .sampling import (
    DequantizedSampler,
    QIMonteCarlo,
    TNSampler,
)
from .linear_algebra import (
    QIPCA,
    QIRegression,
    QISVD,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single algorithm run on a single problem instance."""

    algorithm: str
    problem_size: int
    solution_quality: float
    time_seconds: float
    iterations: int
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkComparison:
    """Comparison of several algorithms on the same problem family."""

    results: List[BenchmarkResult]

    def summary_table(self) -> str:
        """Return an ASCII table comparing all algorithms."""
        if not self.results:
            return "(no results)"

        # Column widths
        alg_w = max(len(r.algorithm) for r in self.results)
        alg_w = max(alg_w, len("Algorithm"))
        header = (
            f"{'Algorithm':<{alg_w}}  "
            f"{'Size':>6}  "
            f"{'Quality':>12}  "
            f"{'Time (s)':>10}  "
            f"{'Iters':>7}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        for r in self.results:
            lines.append(
                f"{r.algorithm:<{alg_w}}  "
                f"{r.problem_size:>6}  "
                f"{r.solution_quality:>12.6f}  "
                f"{r.time_seconds:>10.4f}  "
                f"{r.iterations:>7}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def quality_ratios(self) -> Dict[str, float]:
        """Ratio of each algorithm's quality to the best found.

        Returns a dict mapping algorithm name to a ratio where 1.0
        means the algorithm achieved the best (lowest) solution quality.
        Higher values indicate proportionally worse solutions.

        For minimisation: best = lowest value, ratio = 1 + (q - best) / scale.
        """
        if not self.results:
            return {}
        best = min(r.solution_quality for r in self.results)
        worst = max(r.solution_quality for r in self.results)
        span = worst - best
        if span < 1e-15:
            # All algorithms found the same quality
            return {r.algorithm: 1.0 for r in self.results}
        return {
            r.algorithm: 1.0 + (r.solution_quality - best) / span
            for r in self.results
        }


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Run optimisation / sampling / linear-algebra benchmarks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    # -- optimisation ------------------------------------------------------

    def run_optimization(
        self,
        problem_sizes: Optional[List[int]] = None,
        n_trials: int = 1,
        seed: Optional[int] = None,
    ) -> BenchmarkComparison:
        """Compare SQA, QAOA-inspired, quantum-walk, and random search."""
        if problem_sizes is None:
            problem_sizes = [6]
        seed = seed if seed is not None else self.seed
        results: List[BenchmarkResult] = []

        for size in problem_sizes:
            problem = IsingProblem.random(size, seed=seed)
            for trial in range(n_trials):
                trial_seed = seed + trial

                # SQA
                sqa = SimulatedQuantumAnnealing(
                    n_replicas=8, n_sweeps=60, seed=trial_seed,
                )
                t0 = time.perf_counter()
                sqa_res = sqa.solve(problem)
                dt = time.perf_counter() - t0
                results.append(BenchmarkResult(
                    algorithm="SQA",
                    problem_size=size,
                    solution_quality=sqa_res.best_energy,
                    time_seconds=dt,
                    iterations=sqa.n_sweeps,
                    metadata={"trial": trial},
                ))

                # QAOA-inspired (only for small sizes)
                if size <= 14:
                    qaoa = QAOAInspiredOptimizer(
                        depth=1, n_optimization_rounds=3,
                        angle_resolution=10, seed=trial_seed,
                    )
                    t0 = time.perf_counter()
                    qaoa_res = qaoa.solve(problem)
                    dt = time.perf_counter() - t0
                    results.append(BenchmarkResult(
                        algorithm="QAOA-Inspired",
                        problem_size=size,
                        solution_quality=qaoa_res.best_cost,
                        time_seconds=dt,
                        iterations=qaoa.n_optimization_rounds,
                        metadata={"trial": trial, "depth": qaoa.depth},
                    ))

                # Quantum walk (only for small sizes)
                if size <= 14:
                    qw = QuantumWalkOptimizer(
                        walk_time=3.0, n_steps=20, seed=trial_seed,
                    )
                    t0 = time.perf_counter()
                    qw_res = qw.solve(problem)
                    dt = time.perf_counter() - t0
                    results.append(BenchmarkResult(
                        algorithm="QuantumWalk",
                        problem_size=size,
                        solution_quality=qw_res.best_energy,
                        time_seconds=dt,
                        iterations=qw.n_steps,
                        metadata={"trial": trial},
                    ))

                # Random search baseline
                rng = np.random.default_rng(trial_seed)
                t0 = time.perf_counter()
                n_random = 200
                best_random = np.inf
                for _ in range(n_random):
                    spins = rng.choice([-1, 1], size=size).astype(np.float64)
                    e = problem.energy(spins)
                    if e < best_random:
                        best_random = e
                dt = time.perf_counter() - t0
                results.append(BenchmarkResult(
                    algorithm="RandomSearch",
                    problem_size=size,
                    solution_quality=best_random,
                    time_seconds=dt,
                    iterations=n_random,
                    metadata={"trial": trial},
                ))

        return BenchmarkComparison(results=results)

    # -- sampling ----------------------------------------------------------

    def run_sampling(
        self,
        distribution_sizes: Optional[List[int]] = None,
        n_trials: int = 1,
        seed: Optional[int] = None,
    ) -> BenchmarkComparison:
        """Compare dequantized, TN, and standard MCMC sampling."""
        if distribution_sizes is None:
            distribution_sizes = [8]
        seed = seed if seed is not None else self.seed
        results: List[BenchmarkResult] = []

        for size in distribution_sizes:
            rng = np.random.default_rng(seed)
            # Create a target distribution (softmax of random energies)
            energies = rng.standard_normal(size)
            target = np.exp(-energies)
            target /= target.sum()

            for trial in range(n_trials):
                trial_seed = seed + trial
                n_samples = 500

                # Dequantized sampler (sample from a matrix whose rows
                # define distributions)
                mat = np.outer(target, np.ones(size)) + 0.1 * rng.standard_normal((size, size))
                ds = DequantizedSampler(seed=trial_seed)
                t0 = time.perf_counter()
                row_idx, _ = ds.sample_row(mat, n_samples=n_samples)
                dt = time.perf_counter() - t0
                # Quality: how close the row-sample distribution is to target
                empirical = np.bincount(row_idx, minlength=size).astype(np.float64)
                empirical /= empirical.sum()
                tv = 0.5 * np.sum(np.abs(empirical - target))
                results.append(BenchmarkResult(
                    algorithm="Dequantized",
                    problem_size=size,
                    solution_quality=tv,
                    time_seconds=dt,
                    iterations=n_samples,
                    metadata={"trial": trial},
                ))

                # TN sampler
                tn = TNSampler(bond_dimension=4, seed=trial_seed)
                tensors = TNSampler.random_mps(
                    n_sites=max(3, int(np.log2(size)) + 1),
                    phys_dim=2,
                    bond_dim=4,
                    seed=trial_seed,
                )
                t0 = time.perf_counter()
                tn_res = tn.sample(tensors, n_samples=n_samples)
                dt = time.perf_counter() - t0
                # Quality: entropy of samples (higher is more exploratory)
                unique_count = len(np.unique(tn_res.samples, axis=0))
                diversity = unique_count / n_samples
                results.append(BenchmarkResult(
                    algorithm="TNSampler",
                    problem_size=size,
                    solution_quality=1.0 - diversity,  # lower is better
                    time_seconds=dt,
                    iterations=n_samples,
                    metadata={"trial": trial},
                ))

                # QI-MCMC
                mc = QIMonteCarlo(n_steps=n_samples, n_burnin=100, seed=trial_seed)
                t0 = time.perf_counter()
                mc_res = mc.sample(target, n_samples=n_samples)
                dt = time.perf_counter() - t0
                tv_mc = mc_res.mixing_diagnostics["total_variation_distance"]
                results.append(BenchmarkResult(
                    algorithm="QI-MCMC",
                    problem_size=size,
                    solution_quality=tv_mc,
                    time_seconds=dt,
                    iterations=n_samples,
                    metadata={"trial": trial, "ess": mc_res.effective_sample_size},
                ))

        return BenchmarkComparison(results=results)

    # -- linear algebra ----------------------------------------------------

    def run_linear_algebra(
        self,
        matrix_sizes: Optional[List[int]] = None,
        ranks: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> BenchmarkComparison:
        """Compare QI-SVD vs numpy SVD, QI-Regression, QI-PCA."""
        if matrix_sizes is None:
            matrix_sizes = [50]
        if ranks is None:
            ranks = [5]
        seed = seed if seed is not None else self.seed
        results: List[BenchmarkResult] = []

        rng = np.random.default_rng(seed)

        for m in matrix_sizes:
            n = int(m * 0.8) if int(m * 0.8) > 0 else m
            for k in ranks:
                # Generate a low-rank + noise matrix
                U_true = rng.standard_normal((m, k))
                V_true = rng.standard_normal((n, k))
                A = U_true @ V_true.T + 0.1 * rng.standard_normal((m, n))

                # NumPy SVD (baseline)
                t0 = time.perf_counter()
                U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)
                recon_np = (U_np[:, :k] * S_np[:k]) @ Vt_np[:k, :]
                err_np = np.linalg.norm(A - recon_np) / np.linalg.norm(A)
                dt = time.perf_counter() - t0
                results.append(BenchmarkResult(
                    algorithm="NumPy-SVD",
                    problem_size=m,
                    solution_quality=err_np,
                    time_seconds=dt,
                    iterations=1,
                    metadata={"rank": k, "n_cols": n},
                ))

                # QI-SVD
                qi_svd = QISVD(oversampling=10, seed=seed)
                t0 = time.perf_counter()
                qi_res = qi_svd.fit(A, k=k)
                dt = time.perf_counter() - t0
                results.append(BenchmarkResult(
                    algorithm="QI-SVD",
                    problem_size=m,
                    solution_quality=qi_res.relative_error,
                    time_seconds=dt,
                    iterations=qi_res.n_samples_used,
                    metadata={"rank": k, "n_cols": n},
                ))

                # QI-Regression
                x_true = rng.standard_normal(n)
                b = A @ x_true + 0.05 * rng.standard_normal(m)
                qi_reg = QIRegression(oversampling_factor=4, seed=seed)
                t0 = time.perf_counter()
                reg_res = qi_reg.fit(A, b)
                dt = time.perf_counter() - t0
                results.append(BenchmarkResult(
                    algorithm="QI-Regression",
                    problem_size=m,
                    solution_quality=1.0 - reg_res.r_squared,
                    time_seconds=dt,
                    iterations=reg_res.n_samples_used,
                    metadata={"rank": k, "n_cols": n},
                ))

                # QI-PCA
                qi_pca = QIPCA(oversampling=10, seed=seed)
                t0 = time.perf_counter()
                pca_res = qi_pca.fit(A, k=k)
                dt = time.perf_counter() - t0
                # Quality: sum of explained variance ratio (higher is better)
                evr_sum = float(pca_res.explained_variance_ratio.sum())
                results.append(BenchmarkResult(
                    algorithm="QI-PCA",
                    problem_size=m,
                    solution_quality=1.0 - min(evr_sum, 1.0),
                    time_seconds=dt,
                    iterations=pca_res.n_samples,
                    metadata={"rank": k, "n_cols": n},
                ))

        return BenchmarkComparison(results=results)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def run_optimization_benchmark(
    n: int = 10, seed: int = 42
) -> BenchmarkComparison:
    """Quick optimisation benchmark at the given problem size."""
    suite = BenchmarkSuite(seed=seed)
    return suite.run_optimization(problem_sizes=[n], n_trials=1, seed=seed)


def run_sampling_benchmark(
    n: int = 8, seed: int = 42
) -> BenchmarkComparison:
    """Quick sampling benchmark with the given distribution size."""
    suite = BenchmarkSuite(seed=seed)
    return suite.run_sampling(distribution_sizes=[n], n_trials=1, seed=seed)


def run_linear_algebra_benchmark(
    m: int = 50, n: int = 40, k: int = 5, seed: int = 42
) -> BenchmarkComparison:
    """Quick linear-algebra benchmark."""
    suite = BenchmarkSuite(seed=seed)
    return suite.run_linear_algebra(
        matrix_sizes=[m], ranks=[k], seed=seed,
    )
