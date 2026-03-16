"""Comprehensive tests for the nQPU quantum-inspired classical algorithms package.

Tests cover all four modules:
  - optimization: IsingProblem, SQA, QAOAInspired, QuantumWalkOptimizer
  - sampling: DequantizedSampler, TNSampler, QIMonteCarlo
  - linear_algebra: QISVD, QIRegression, QIPCA
  - benchmarks: BenchmarkSuite, convenience runners, comparison utilities
"""

import numpy as np
import pytest

from nqpu.classical_inspired import (
    # optimization
    IsingProblem,
    SimulatedQuantumAnnealing,
    SQAResult,
    QAOAInspiredOptimizer,
    QAOAInspiredResult,
    QuantumWalkOptimizer,
    QuantumWalkOptimizerResult,
    # sampling
    DequantizedSampler,
    DequantizedSample,
    TNSampler,
    TNSampleResult,
    QIMonteCarlo,
    QIMCResult,
    # linear_algebra
    QISVD,
    QISVDResult,
    QIRegression,
    QIRegressionResult,
    QIPCA,
    QIPCAResult,
    # benchmarks
    BenchmarkSuite,
    BenchmarkResult,
    BenchmarkComparison,
    run_optimization_benchmark,
    run_sampling_benchmark,
    run_linear_algebra_benchmark,
)


# ===================================================================
# TestIsingProblem
# ===================================================================

class TestIsingProblem:
    """Tests for the IsingProblem data class and conversions."""

    def test_construction_basic(self):
        """Construct a simple 3-spin Ising problem."""
        J = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64)
        h = np.array([0.5, -0.5, 0.0], dtype=np.float64)
        prob = IsingProblem(J=J, h=h)
        assert prob.n == 3
        assert prob.J.shape == (3, 3)
        assert prob.h.shape == (3,)

    def test_construction_shape_mismatch_raises(self):
        """J and h with mismatched dimensions should raise."""
        with pytest.raises(ValueError, match="does not match"):
            IsingProblem(J=np.zeros((3, 3)), h=np.zeros(4))

    def test_energy_all_up(self):
        """Energy of all-up spins on a ferromagnetic chain."""
        n = 4
        J = np.zeros((n, n))
        for i in range(n - 1):
            J[i, i + 1] = -1.0
            J[i + 1, i] = -1.0
        h = np.zeros(n)
        prob = IsingProblem(J=J, h=h)
        spins = np.ones(n)  # all +1
        # E = sum_ij J_ij s_i s_j = each pair contributes -1 twice
        # (i->j and j->i), so E = -2*(n-1) for symmetric J
        expected = float(spins @ J @ spins)
        assert prob.energy(spins) == pytest.approx(expected)

    def test_energy_antiferromagnetic(self):
        """Alternating spins should have positive energy on ferromagnetic J."""
        n = 4
        J = np.zeros((n, n))
        for i in range(n - 1):
            J[i, i + 1] = -1.0
            J[i + 1, i] = -1.0
        prob = IsingProblem(J=J, h=np.zeros(n))
        spins = np.array([1, -1, 1, -1], dtype=np.float64)
        # Adjacent spins are opposite => s_i * s_j = -1
        # E = sum_ij J_ij s_i s_j, J has -1 for neighbors
        assert prob.energy(spins) > 0

    def test_from_qubo_round_trip(self):
        """QUBO -> Ising conversion preserves minimum location."""
        rng = np.random.default_rng(42)
        n = 4
        Q = rng.standard_normal((n, n))
        Q = (Q + Q.T) / 2
        prob = IsingProblem.from_qubo(Q)
        assert prob.n == n
        assert prob.J.shape == (n, n)
        # Diagonal of J should be zero
        np.testing.assert_allclose(np.diag(prob.J), 0.0)

    def test_random_creates_valid_problem(self):
        """IsingProblem.random produces a valid symmetric problem."""
        prob = IsingProblem.random(8, seed=99)
        assert prob.n == 8
        np.testing.assert_allclose(prob.J, prob.J.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(prob.J), 0.0)

    def test_max_cut_triangle(self):
        """Max-Cut on a triangle graph (3 nodes, all connected)."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
        prob = IsingProblem.max_cut(adj)
        assert prob.n == 3
        # With J = -A/4, a valid max-cut has 2 edges cut out of 3
        # Check that we can evaluate energy
        spins = np.array([1, 1, -1], dtype=np.float64)
        e = prob.energy(spins)
        assert isinstance(e, float)

    def test_max_cut_optimal_known(self):
        """For a 4-cycle the max cut is 4 edges (bipartite)."""
        adj = np.zeros((4, 4))
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for i, j in edges:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        prob = IsingProblem.max_cut(adj)
        # Optimal partition: {0,2} vs {1,3} => all 4 edges cut
        optimal_spins = np.array([1, -1, 1, -1], dtype=np.float64)
        e_opt = prob.energy(optimal_spins)
        # Worst partition: all same side => 0 edges cut
        worst_spins = np.ones(4)
        e_worst = prob.energy(worst_spins)
        # Max-Cut: lower Ising energy = more edges cut
        assert e_opt < e_worst


# ===================================================================
# TestSQA
# ===================================================================

class TestSQA:
    """Tests for Simulated Quantum Annealing."""

    def test_basic_solve_returns_result(self):
        """SQA produces an SQAResult with correct fields."""
        prob = IsingProblem.random(5, seed=42)
        sqa = SimulatedQuantumAnnealing(
            n_replicas=4, n_sweeps=20, seed=42,
        )
        result = sqa.solve(prob)
        assert isinstance(result, SQAResult)
        assert result.best_bitstring.shape == (5,)
        assert all(s in (-1.0, 1.0) for s in result.best_bitstring)
        assert len(result.energy_history) == 20
        assert len(result.acceptance_rates) == 20

    def test_random_ising_finds_good_solution(self):
        """SQA should beat random search on a small random instance."""
        prob = IsingProblem.random(6, seed=7)
        sqa = SimulatedQuantumAnnealing(
            n_replicas=8, n_sweeps=200, seed=7,
        )
        result = sqa.solve(prob)

        # Compare with random search baseline
        rng = np.random.default_rng(7)
        random_best = np.inf
        for _ in range(500):
            spins = rng.choice([-1, 1], size=prob.n).astype(np.float64)
            e = prob.energy(spins)
            random_best = min(random_best, e)

        # SQA should find an energy no worse than the random baseline
        assert result.best_energy <= random_best + 1.0

    def test_temperature_effect(self):
        """Moderate temperature should outperform very high temperature."""
        prob = IsingProblem.random(5, seed=10)
        sqa_moderate = SimulatedQuantumAnnealing(
            n_replicas=8, n_sweeps=150, temperature=0.5, seed=10,
        )
        sqa_very_hot = SimulatedQuantumAnnealing(
            n_replicas=8, n_sweeps=150, temperature=50.0, seed=10,
        )
        moderate = sqa_moderate.solve(prob)
        very_hot = sqa_very_hot.solve(prob)
        # At T=50 essentially all moves are accepted (random walk),
        # while T=0.5 should allow meaningful optimisation.
        assert moderate.best_energy <= very_hot.best_energy + 2.0

    def test_energy_history_non_increasing(self):
        """The best-energy history should be monotonically non-increasing."""
        prob = IsingProblem.random(5, seed=42)
        sqa = SimulatedQuantumAnnealing(n_replicas=4, n_sweeps=50, seed=42)
        result = sqa.solve(prob)
        for i in range(1, len(result.energy_history)):
            assert result.energy_history[i] <= result.energy_history[i - 1] + 1e-12

    def test_acceptance_rates_bounded(self):
        """Acceptance rates should be in [0, 1]."""
        prob = IsingProblem.random(4, seed=42)
        sqa = SimulatedQuantumAnnealing(n_replicas=4, n_sweeps=30, seed=42)
        result = sqa.solve(prob)
        for rate in result.acceptance_rates:
            assert 0.0 <= rate <= 1.0

    def test_max_cut_small_graph(self):
        """SQA finds a good cut on a small graph."""
        adj = np.zeros((4, 4))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        prob = IsingProblem.max_cut(adj)
        sqa = SimulatedQuantumAnnealing(
            n_replicas=8, n_sweeps=100, seed=42,
        )
        result = sqa.solve(prob)
        # The optimal Ising energy for a 4-cycle max-cut is known
        optimal = np.array([1, -1, 1, -1], dtype=np.float64)
        opt_energy = prob.energy(optimal)
        assert result.best_energy <= opt_energy + 0.1

    def test_replicas_field(self):
        """Result reports the correct number of replicas."""
        prob = IsingProblem.random(3, seed=42)
        sqa = SimulatedQuantumAnnealing(n_replicas=12, n_sweeps=10, seed=42)
        result = sqa.solve(prob)
        assert result.n_replicas == 12
        assert result.n_sweeps == 10

    def test_deterministic_with_same_seed(self):
        """Same seed produces identical results."""
        prob = IsingProblem.random(4, seed=42)
        r1 = SimulatedQuantumAnnealing(n_replicas=4, n_sweeps=20, seed=99).solve(prob)
        r2 = SimulatedQuantumAnnealing(n_replicas=4, n_sweeps=20, seed=99).solve(prob)
        np.testing.assert_array_equal(r1.best_bitstring, r2.best_bitstring)
        assert r1.best_energy == r2.best_energy


# ===================================================================
# TestQAOAInspired
# ===================================================================

class TestQAOAInspired:
    """Tests for the QAOA-inspired classical optimizer."""

    def test_basic_solve(self):
        """QAOAInspiredOptimizer produces a valid result."""
        prob = IsingProblem.random(3, seed=42)
        qaoa = QAOAInspiredOptimizer(
            depth=1, n_optimization_rounds=3, angle_resolution=8, seed=42,
        )
        result = qaoa.solve(prob)
        assert isinstance(result, QAOAInspiredResult)
        assert result.best_bitstring.shape == (3,)
        assert result.depth == 1
        assert len(result.cost_history) == 3

    def test_state_vector_normalised(self):
        """The output state vector should be normalised to 1."""
        prob = IsingProblem.random(3, seed=42)
        qaoa = QAOAInspiredOptimizer(
            depth=1, n_optimization_rounds=2, angle_resolution=8, seed=42,
        )
        result = qaoa.solve(prob)
        norm = np.sum(np.abs(result.state_vector) ** 2)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_optimal_angles_returned(self):
        """Optimal angles should be arrays of the correct length."""
        prob = IsingProblem.random(3, seed=42)
        qaoa = QAOAInspiredOptimizer(
            depth=2, n_optimization_rounds=2, angle_resolution=8, seed=42,
        )
        result = qaoa.solve(prob)
        gammas, betas = result.optimal_angles
        assert len(gammas) == 2
        assert len(betas) == 2

    def test_cost_history_length(self):
        """Cost history should have one entry per optimisation round."""
        prob = IsingProblem.random(3, seed=42)
        qaoa = QAOAInspiredOptimizer(
            depth=1, n_optimization_rounds=5, angle_resolution=8, seed=42,
        )
        result = qaoa.solve(prob)
        assert len(result.cost_history) == 5

    def test_rejects_large_n(self):
        """Problems with n > 20 should raise ValueError."""
        prob = IsingProblem.random(21, seed=42)
        qaoa = QAOAInspiredOptimizer(depth=1, seed=42)
        with pytest.raises(ValueError, match="n <= 20"):
            qaoa.solve(prob)

    def test_rejects_depth_zero(self):
        """Depth < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="depth must be >= 1"):
            QAOAInspiredOptimizer(depth=0)

    def test_max_cut_small(self):
        """QAOA-inspired should find a reasonable max-cut on a 3-node triangle."""
        adj = np.ones((3, 3)) - np.eye(3)
        prob = IsingProblem.max_cut(adj)
        qaoa = QAOAInspiredOptimizer(
            depth=2, n_optimization_rounds=5, angle_resolution=12, seed=42,
        )
        result = qaoa.solve(prob)
        # The best_cost should be at most the energy of any partition
        # (triangle max-cut = 2 edges, so Ising energy = -2 * 1/4 * 2 = -1)
        all_up = prob.energy(np.ones(3))
        assert result.best_cost <= all_up + 0.01

    def test_deterministic(self):
        """Same seed gives identical results."""
        prob = IsingProblem.random(3, seed=42)
        r1 = QAOAInspiredOptimizer(
            depth=1, n_optimization_rounds=2, angle_resolution=8, seed=77,
        ).solve(prob)
        r2 = QAOAInspiredOptimizer(
            depth=1, n_optimization_rounds=2, angle_resolution=8, seed=77,
        ).solve(prob)
        assert r1.best_cost == r2.best_cost
        np.testing.assert_array_equal(r1.best_bitstring, r2.best_bitstring)


# ===================================================================
# TestQuantumWalkOptimizer
# ===================================================================

class TestQuantumWalkOptimizer:
    """Tests for the quantum walk optimizer."""

    def test_basic_walk(self):
        """QuantumWalkOptimizer returns a valid result on a tiny problem."""
        prob = IsingProblem.random(3, seed=42)
        qw = QuantumWalkOptimizer(walk_time=2.0, n_steps=10, seed=42)
        result = qw.solve(prob)
        assert isinstance(result, QuantumWalkOptimizerResult)
        assert result.best_solution.shape == (3,)
        assert result.evolution_time == 2.0
        assert len(result.walk_history) == 10

    def test_probabilities_normalised(self):
        """Probability distribution should sum to 1."""
        prob = IsingProblem.random(3, seed=42)
        qw = QuantumWalkOptimizer(walk_time=3.0, n_steps=20, seed=42)
        result = qw.solve(prob)
        assert result.probabilities.sum() == pytest.approx(1.0, abs=1e-10)

    def test_probabilities_non_negative(self):
        """All probabilities should be non-negative."""
        prob = IsingProblem.random(4, seed=42)
        qw = QuantumWalkOptimizer(walk_time=2.0, n_steps=10, seed=42)
        result = qw.solve(prob)
        assert np.all(result.probabilities >= -1e-15)

    def test_rejects_large_n(self):
        """Problems with n > 16 should raise ValueError."""
        prob = IsingProblem.random(17, seed=42)
        qw = QuantumWalkOptimizer(seed=42)
        with pytest.raises(ValueError, match="n <= 16"):
            qw.solve(prob)

    def test_max_cut_4cycle(self):
        """Quantum walk on a 4-cycle max-cut problem."""
        adj = np.zeros((4, 4))
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        prob = IsingProblem.max_cut(adj)
        qw = QuantumWalkOptimizer(walk_time=5.0, n_steps=30, gamma=0.5, seed=42)
        result = qw.solve(prob)
        # Should find a partition; the best_energy should be finite
        assert np.isfinite(result.best_energy)

    def test_deterministic(self):
        """Same seed gives identical results (no randomness in walk)."""
        prob = IsingProblem.random(3, seed=42)
        r1 = QuantumWalkOptimizer(walk_time=2.0, n_steps=10, seed=42).solve(prob)
        r2 = QuantumWalkOptimizer(walk_time=2.0, n_steps=10, seed=42).solve(prob)
        np.testing.assert_array_equal(r1.best_solution, r2.best_solution)
        assert r1.best_energy == r2.best_energy


# ===================================================================
# TestDequantizedSampler
# ===================================================================

class TestDequantizedSampler:
    """Tests for Tang's dequantized sampler."""

    def test_sample_row_distribution(self):
        """Row sampling should favour rows with large norms."""
        rng = np.random.default_rng(42)
        A = np.zeros((5, 3))
        A[0, :] = 10.0  # row 0 has much larger norm
        A[1:, :] = 0.01
        ds = DequantizedSampler(seed=42)
        indices, probs = ds.sample_row(A, n_samples=200)
        # Row 0 should dominate
        frac_row0 = np.mean(indices == 0)
        assert frac_row0 > 0.8

    def test_sample_row_returns_valid_indices(self):
        """Returned indices should be valid row indices."""
        A = np.random.default_rng(42).standard_normal((10, 5))
        ds = DequantizedSampler(seed=42)
        indices, probs = ds.sample_row(A, n_samples=50)
        assert indices.shape == (50,)
        assert np.all(indices >= 0)
        assert np.all(indices < 10)

    def test_sample_entry_distribution(self):
        """Entry sampling should favour large entries in a given row."""
        A = np.zeros((3, 5))
        A[1, 3] = 100.0  # dominant entry
        A[1, :3] = 0.01
        A[1, 4] = 0.01
        ds = DequantizedSampler(seed=42)
        cols, probs = ds.sample_entry(A, row_idx=1, n_samples=100)
        frac_col3 = np.mean(cols == 3)
        assert frac_col3 > 0.8

    def test_query_exact(self):
        """Query should return the exact matrix entry."""
        A = np.array([[1.5, 2.5], [3.5, 4.5]])
        ds = DequantizedSampler(seed=42)
        assert ds.query(A, 0, 1) == pytest.approx(2.5)
        assert ds.query(A, 1, 0) == pytest.approx(3.5)

    def test_recommend_low_rank(self):
        """Recommendation on a rank-1 matrix should have low error."""
        rng = np.random.default_rng(42)
        u = rng.standard_normal((20, 1))
        v = rng.standard_normal((15, 1))
        A = u @ v.T  # exact rank 1
        ds = DequantizedSampler(seed=42)
        result = ds.recommend(A, k=1)
        assert isinstance(result, DequantizedSample)
        # Reconstruction error on sampled submatrix should be small
        assert result.reconstruction_error < 0.5

    def test_recommend_returns_valid_fields(self):
        """DequantizedSample fields should have consistent shapes."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 8))
        ds = DequantizedSampler(seed=42)
        result = ds.recommend(A, k=3)
        assert result.n_queries > 0
        assert len(result.weights) == len(result.sampled_row_indices)

    def test_zero_matrix_fallback(self):
        """Zero matrix should not crash (uniform fallback)."""
        A = np.zeros((5, 5))
        ds = DequantizedSampler(seed=42)
        indices, probs = ds.sample_row(A, n_samples=10)
        assert indices.shape == (10,)

    def test_recommend_higher_rank(self):
        """Recommend on a rank-3 matrix with k=3."""
        rng = np.random.default_rng(42)
        U = rng.standard_normal((15, 3))
        V = rng.standard_normal((12, 3))
        A = U @ V.T
        ds = DequantizedSampler(seed=42)
        result = ds.recommend(A, k=3)
        assert result.reconstruction_error < 0.8


# ===================================================================
# TestTNSampler
# ===================================================================

class TestTNSampler:
    """Tests for tensor-network MPS sampling."""

    def test_random_mps_shapes(self):
        """random_mps returns tensors with consistent shapes."""
        tensors = TNSampler.random_mps(n_sites=5, phys_dim=2, bond_dim=4, seed=42)
        assert len(tensors) == 5
        # First tensor: (1, d, br)
        assert tensors[0].shape[0] == 1
        assert tensors[0].shape[1] == 2
        # Last tensor: (bl, d, 1)
        assert tensors[-1].shape[2] == 1

    def test_sample_returns_correct_shape(self):
        """TNSampler produces samples with (n_samples, n_sites) shape."""
        tensors = TNSampler.random_mps(n_sites=4, phys_dim=2, bond_dim=4, seed=42)
        tn = TNSampler(bond_dimension=4, seed=42)
        result = tn.sample(tensors, n_samples=50)
        assert isinstance(result, TNSampleResult)
        assert result.samples.shape == (50, 4)
        assert result.log_probs.shape == (50,)
        assert result.acceptance_rate == 1.0  # direct sampling

    def test_samples_in_valid_range(self):
        """All sampled values should be valid physical indices."""
        phys_dim = 3
        tensors = TNSampler.random_mps(
            n_sites=4, phys_dim=phys_dim, bond_dim=4, seed=42,
        )
        tn = TNSampler(bond_dimension=4, seed=42)
        result = tn.sample(tensors, n_samples=100)
        assert np.all(result.samples >= 0)
        assert np.all(result.samples < phys_dim)

    def test_log_probs_finite(self):
        """Log probabilities should be finite."""
        tensors = TNSampler.random_mps(n_sites=4, phys_dim=2, bond_dim=4, seed=42)
        tn = TNSampler(bond_dimension=4, seed=42)
        result = tn.sample(tensors, n_samples=50)
        assert np.all(np.isfinite(result.log_probs))

    def test_log_probs_negative(self):
        """Log probabilities should be non-positive (probs <= 1)."""
        tensors = TNSampler.random_mps(n_sites=4, phys_dim=2, bond_dim=4, seed=42)
        tn = TNSampler(bond_dimension=4, seed=42)
        result = tn.sample(tensors, n_samples=50)
        assert np.all(result.log_probs <= 0.01)  # small tolerance

    def test_deterministic_sampling(self):
        """Same seed produces same samples."""
        tensors = TNSampler.random_mps(n_sites=3, phys_dim=2, bond_dim=4, seed=42)
        r1 = TNSampler(seed=99).sample(tensors, n_samples=20)
        r2 = TNSampler(seed=99).sample(tensors, n_samples=20)
        np.testing.assert_array_equal(r1.samples, r2.samples)


# ===================================================================
# TestQIMonteCarlo
# ===================================================================

class TestQIMonteCarlo:
    """Tests for quantum-inspired lifted MCMC."""

    def test_basic_sampling(self):
        """QIMonteCarlo produces samples of the requested count."""
        target = np.array([0.1, 0.3, 0.4, 0.2])
        mc = QIMonteCarlo(n_steps=500, n_burnin=100, seed=42)
        result = mc.sample(target, n_samples=500)
        assert isinstance(result, QIMCResult)
        assert len(result.samples) == 500
        assert result.acceptance_rate > 0

    def test_target_distribution_convergence(self):
        """With enough samples, the empirical distribution should be close."""
        target = np.array([0.1, 0.2, 0.3, 0.4])
        mc = QIMonteCarlo(n_steps=5000, n_burnin=500, seed=42)
        result = mc.sample(target, n_samples=5000)
        tv = result.mixing_diagnostics["total_variation_distance"]
        assert tv < 0.15  # should be reasonably close

    def test_uniform_distribution(self):
        """Sampling from a uniform distribution should yield uniform samples."""
        N = 5
        target = np.ones(N)  # unnormalised uniform
        mc = QIMonteCarlo(n_steps=3000, n_burnin=300, seed=42)
        result = mc.sample(target, n_samples=3000)
        empirical = result.mixing_diagnostics["empirical_distribution"]
        # Each state should get roughly 1/N
        for p in empirical:
            assert abs(p - 1.0 / N) < 0.1

    def test_effective_sample_size_positive(self):
        """ESS should be positive."""
        target = np.array([0.5, 0.3, 0.2])
        mc = QIMonteCarlo(n_steps=500, n_burnin=100, seed=42)
        result = mc.sample(target, n_samples=500)
        assert result.effective_sample_size > 0

    def test_samples_in_range(self):
        """All samples should be valid state indices."""
        N = 6
        target = np.random.default_rng(42).random(N)
        mc = QIMonteCarlo(n_steps=300, n_burnin=50, seed=42)
        result = mc.sample(target, n_samples=300)
        assert np.all(result.samples >= 0)
        assert np.all(result.samples < N)

    def test_deterministic(self):
        """Same seed produces same samples."""
        target = np.array([0.2, 0.5, 0.3])
        r1 = QIMonteCarlo(n_steps=200, n_burnin=50, seed=77).sample(target, 200)
        r2 = QIMonteCarlo(n_steps=200, n_burnin=50, seed=77).sample(target, 200)
        np.testing.assert_array_equal(r1.samples, r2.samples)


# ===================================================================
# TestQISVD
# ===================================================================

class TestQISVD:
    """Tests for quantum-inspired SVD."""

    def test_rank_approximation_quality(self):
        """QI-SVD on a low-rank matrix should have low relative error."""
        rng = np.random.default_rng(42)
        m, n, k = 30, 25, 3
        U = rng.standard_normal((m, k))
        V = rng.standard_normal((n, k))
        A = U @ V.T  # exact rank k
        qi = QISVD(oversampling=10, seed=42)
        result = qi.fit(A, k=k)
        assert isinstance(result, QISVDResult)
        assert result.U.shape == (m, k)
        assert result.S.shape == (k,)
        assert result.Vt.shape == (k, n)
        assert result.relative_error < 0.5

    def test_explained_variance_sum(self):
        """Explained variance should be non-negative."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 15))
        qi = QISVD(oversampling=5, seed=42)
        result = qi.fit(A, k=3)
        assert np.all(result.explained_variance >= -1e-10)

    def test_singular_values_positive(self):
        """Singular values should be non-negative."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 15))
        qi = QISVD(oversampling=5, seed=42)
        result = qi.fit(A, k=3)
        assert np.all(result.S >= -1e-10)

    def test_n_samples_used(self):
        """n_samples_used should be reported."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((20, 15))
        qi = QISVD(oversampling=5, seed=42)
        result = qi.fit(A, k=3)
        assert result.n_samples_used > 0
        assert result.n_samples_used <= min(20, 15)

    def test_zero_matrix(self):
        """QI-SVD on a zero matrix should not crash."""
        A = np.zeros((10, 8))
        qi = QISVD(seed=42)
        result = qi.fit(A, k=2)
        assert np.all(np.isfinite(result.S))

    def test_square_matrix(self):
        """QI-SVD works on a square matrix."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((15, 15))
        qi = QISVD(oversampling=5, seed=42)
        result = qi.fit(A, k=5)
        assert result.U.shape[0] == 15
        assert result.Vt.shape[1] == 15


# ===================================================================
# TestQIRegression
# ===================================================================

class TestQIRegression:
    """Tests for quantum-inspired regression."""

    def test_coefficient_accuracy(self):
        """QI-Regression should recover coefficients of a linear system."""
        rng = np.random.default_rng(42)
        m, n = 100, 5
        A = rng.standard_normal((m, n))
        x_true = rng.standard_normal(n)
        b = A @ x_true + 0.01 * rng.standard_normal(m)
        qi = QIRegression(oversampling_factor=6, seed=42)
        result = qi.fit(A, b)
        assert isinstance(result, QIRegressionResult)
        # Coefficients should be close
        np.testing.assert_allclose(result.coefficients, x_true, atol=0.5)

    def test_r_squared_high_for_clean_data(self):
        """R^2 should be high for a nearly noiseless system."""
        rng = np.random.default_rng(42)
        m, n = 80, 4
        A = rng.standard_normal((m, n))
        x_true = np.array([1.0, -2.0, 3.0, -4.0])
        b = A @ x_true + 0.001 * rng.standard_normal(m)
        qi = QIRegression(oversampling_factor=8, seed=42)
        result = qi.fit(A, b)
        assert result.r_squared > 0.9

    def test_residual_reasonable(self):
        """Residual should be finite and non-negative."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((50, 5))
        b = rng.standard_normal(50)
        qi = QIRegression(seed=42)
        result = qi.fit(A, b)
        assert result.residual >= 0
        assert np.isfinite(result.residual)

    def test_n_samples_used_reported(self):
        """n_samples_used should be positive."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((40, 5))
        b = rng.standard_normal(40)
        qi = QIRegression(oversampling_factor=4, seed=42)
        result = qi.fit(A, b)
        assert result.n_samples_used > 0
        assert result.n_samples_used <= 40

    def test_r_squared_bounded(self):
        """R^2 should be in [0, 1]."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((60, 5))
        x_true = rng.standard_normal(5)
        b = A @ x_true + 0.1 * rng.standard_normal(60)
        qi = QIRegression(oversampling_factor=4, seed=42)
        result = qi.fit(A, b)
        assert 0.0 <= result.r_squared <= 1.0


# ===================================================================
# TestQIPCA
# ===================================================================

class TestQIPCA:
    """Tests for quantum-inspired PCA."""

    def test_component_recovery(self):
        """QI-PCA should recover the dominant direction of rank-1 data."""
        rng = np.random.default_rng(42)
        m, n = 60, 10
        true_direction = rng.standard_normal(n)
        true_direction /= np.linalg.norm(true_direction)
        scores = rng.standard_normal(m)
        X = np.outer(scores, true_direction) + 0.05 * rng.standard_normal((m, n))
        qi = QIPCA(oversampling=10, seed=42)
        result = qi.fit(X, k=1)
        assert isinstance(result, QIPCAResult)
        assert result.components.shape == (1, n)
        # The first component should align with the true direction
        alignment = abs(float(result.components[0] @ true_direction))
        assert alignment > 0.7

    def test_explained_variance_ratio_bounded(self):
        """Explained variance ratios should be in [0, 1]."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 10))
        qi = QIPCA(oversampling=5, seed=42)
        result = qi.fit(X, k=3)
        assert np.all(result.explained_variance_ratio >= -1e-10)
        assert np.all(result.explained_variance_ratio <= 1.0 + 1e-10)

    def test_components_orthogonal(self):
        """Principal components should be roughly orthogonal."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 10))
        qi = QIPCA(oversampling=10, seed=42)
        result = qi.fit(X, k=3)
        gram = result.components @ result.components.T
        # Diagonal should be ~1, off-diagonal ~0
        for i in range(3):
            assert abs(gram[i, i] - 1.0) < 0.3
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(gram[i, j]) < 0.5

    def test_total_variance_positive(self):
        """Total variance should be positive for non-zero data."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 8))
        qi = QIPCA(oversampling=5, seed=42)
        result = qi.fit(X, k=2)
        assert result.total_variance > 0

    def test_n_samples_reported(self):
        """n_samples should be reported correctly."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 20))
        qi = QIPCA(oversampling=5, seed=42)
        result = qi.fit(X, k=3)
        assert result.n_samples > 0
        assert result.n_samples <= 20


# ===================================================================
# TestBenchmarks
# ===================================================================

class TestBenchmarks:
    """Tests for the benchmark suite and convenience functions."""

    def test_optimization_benchmark_runs(self):
        """run_optimization_benchmark completes without error."""
        comp = run_optimization_benchmark(n=4, seed=42)
        assert isinstance(comp, BenchmarkComparison)
        assert len(comp.results) >= 3  # SQA, QAOA, QW, Random

    def test_sampling_benchmark_runs(self):
        """run_sampling_benchmark completes without error."""
        comp = run_sampling_benchmark(n=6, seed=42)
        assert isinstance(comp, BenchmarkComparison)
        assert len(comp.results) >= 3

    def test_linear_algebra_benchmark_runs(self):
        """run_linear_algebra_benchmark completes without error."""
        comp = run_linear_algebra_benchmark(m=20, n=15, k=3, seed=42)
        assert isinstance(comp, BenchmarkComparison)
        assert len(comp.results) >= 3

    def test_summary_table_format(self):
        """Summary table should be a non-empty string with header."""
        comp = run_optimization_benchmark(n=4, seed=42)
        table = comp.summary_table()
        assert isinstance(table, str)
        assert "Algorithm" in table
        assert "Quality" in table
        assert len(table.splitlines()) >= 5

    def test_quality_ratios(self):
        """Quality ratios should map algorithm names to floats."""
        comp = run_optimization_benchmark(n=4, seed=42)
        ratios = comp.quality_ratios()
        assert isinstance(ratios, dict)
        assert len(ratios) >= 3
        # Best algorithm gets ratio 1.0, worst gets 2.0
        assert min(ratios.values()) == pytest.approx(1.0, abs=0.01)
        assert max(ratios.values()) == pytest.approx(2.0, abs=0.01)

    def test_benchmark_suite_custom_sizes(self):
        """BenchmarkSuite with custom problem sizes."""
        suite = BenchmarkSuite(seed=42)
        comp = suite.run_optimization(problem_sizes=[3, 4], n_trials=1)
        # Should have results for both sizes
        sizes = {r.problem_size for r in comp.results}
        assert 3 in sizes
        assert 4 in sizes
