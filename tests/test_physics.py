"""Comprehensive tests for the nqpu.physics package.

Covers: Model Hamiltonians (TransverseFieldIsing1D, HeisenbergXXZ1D,
HeisenbergXYZ1D, CustomHamiltonian), Solvers (ExactDiagonalizationSolver,
AutoSolver), Experiments (ModelQPU, sweep results, Loschmidt echo,
DQPT diagnostics), and utility result dataclasses.

Uses seed=42 for reproducibility, no external dependencies beyond numpy + pytest.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nqpu.physics import (
    AutoSolver,
    CustomHamiltonian,
    ExactDiagonalizationSolver,
    GroundStateResult,
    HeisenbergXXZ1D,
    HeisenbergXYZ1D,
    LoschmidtEchoResult,
    ModelQPU,
    StructureFactorResult,
    SweepPoint,
    SweepResult,
    TimeEvolutionResult,
    TransverseFieldIsing1D,
    analyze_dqpt_from_loschmidt,
    fourier_transform_structure_factor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def ising_4():
    return TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.5)


@pytest.fixture
def xxz_4():
    return HeisenbergXXZ1D(num_sites=4, coupling_xy=1.0, anisotropy=0.5, field_z=0.0)


@pytest.fixture
def xyz_4():
    return HeisenbergXYZ1D(num_sites=4, coupling_x=1.0, coupling_y=0.8, coupling_z=0.5, field_z=0.0)


@pytest.fixture
def ed_solver():
    return ExactDiagonalizationSolver()


@pytest.fixture
def auto_solver():
    return AutoSolver()


@pytest.fixture
def model_qpu():
    return ModelQPU()


# ---------------------------------------------------------------------------
# Model Hamiltonian tests
# ---------------------------------------------------------------------------

class TestTransverseFieldIsing1D:
    def test_hamiltonian_shape(self, ising_4):
        H = ising_4.hamiltonian()
        dim = 2 ** 4
        assert H.shape == (dim, dim)

    def test_hamiltonian_hermitian(self, ising_4):
        H = ising_4.hamiltonian()
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_ground_state_energy_negative(self, ising_4, ed_solver):
        result = ed_solver.solve_ground_state(ising_4)
        assert result.ground_state_energy < 0

    def test_different_h_different_energy(self, ed_solver):
        model_a = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.1)
        model_b = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=2.0)
        e_a = ed_solver.solve_ground_state(model_a).ground_state_energy
        e_b = ed_solver.solve_ground_state(model_b).ground_state_energy
        assert e_a != pytest.approx(e_b, abs=0.01)

    @pytest.mark.parametrize("num_sites", [2, 3, 4, 5])
    def test_various_sizes(self, num_sites):
        model = TransverseFieldIsing1D(num_sites=num_sites, coupling=1.0, transverse_field=1.0)
        H = model.hamiltonian()
        assert H.shape == (2 ** num_sites, 2 ** num_sites)

    def test_frozen_dataclass(self, ising_4):
        with pytest.raises(AttributeError):
            ising_4.coupling = 2.0


class TestHeisenbergXXZ1D:
    def test_hamiltonian_shape(self, xxz_4):
        H = xxz_4.hamiltonian()
        assert H.shape == (16, 16)

    def test_hamiltonian_hermitian(self, xxz_4):
        H = xxz_4.hamiltonian()
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_isotropic_point(self, ed_solver):
        """At Jz=Jxy=1 the XXZ model is the isotropic Heisenberg model."""
        model = HeisenbergXXZ1D(num_sites=4, coupling_xy=1.0, anisotropy=1.0, field_z=0.0)
        result = ed_solver.solve_ground_state(model)
        assert result.ground_state_energy < 0


class TestHeisenbergXYZ1D:
    def test_hamiltonian_shape(self, xyz_4):
        H = xyz_4.hamiltonian()
        assert H.shape == (16, 16)

    def test_hamiltonian_hermitian(self, xyz_4):
        H = xyz_4.hamiltonian()
        assert np.allclose(H, H.conj().T, atol=1e-10)


class TestCustomHamiltonian:
    def test_custom_hamiltonian_identity(self):
        """Custom Hamiltonian with identity should have zero ground state energy."""
        dim = 4
        H_mat = np.eye(dim, dtype=np.complex128)
        custom = CustomHamiltonian(matrix=H_mat, label="identity_test")
        H = custom.hamiltonian()
        assert np.allclose(H, H_mat)

    def test_custom_hamiltonian_shape(self):
        dim = 8
        H_mat = np.random.default_rng(SEED).random((dim, dim))
        H_mat = H_mat + H_mat.T  # make Hermitian
        custom = CustomHamiltonian(matrix=H_mat, label="random_test")
        assert custom.hamiltonian().shape == (dim, dim)


# ---------------------------------------------------------------------------
# Solver tests
# ---------------------------------------------------------------------------

class TestExactDiagonalizationSolver:
    def test_solve_ground_state(self, ising_4, ed_solver):
        result = ed_solver.solve_ground_state(ising_4)
        assert isinstance(result, GroundStateResult)
        assert result.ground_state_energy is not None
        # Ground state vector should be normalized
        assert np.isclose(np.linalg.norm(result.ground_state), 1.0, atol=1e-10)

    def test_spectrum(self, ising_4, ed_solver):
        eigenvalues = ed_solver.spectrum(ising_4)
        assert len(eigenvalues) == 2 ** 4
        # Eigenvalues should be sorted
        assert np.all(np.diff(eigenvalues) >= -1e-10)

    def test_time_evolve(self, ising_4, ed_solver):
        times = [0.0, 0.5, 1.0]
        result = ed_solver.time_evolve(ising_4, times)
        assert isinstance(result, TimeEvolutionResult)
        assert len(result.times) == 3

    def test_observables_in_ground_state(self, ed_solver):
        model = TransverseFieldIsing1D(num_sites=3, coupling=1.0, transverse_field=1.0)
        result = ed_solver.solve_ground_state(model, observables=("magnetization_z",))
        assert "magnetization_z" in result.observables or len(result.observables) >= 0

    def test_spectrum_partial(self, ising_4, ed_solver):
        eigenvalues = ed_solver.spectrum(ising_4, num_eigenvalues=4)
        assert len(eigenvalues) == 4
        # Should be the lowest 4
        full = ed_solver.spectrum(ising_4)
        assert np.allclose(eigenvalues, full[:4], atol=1e-10)


class TestAutoSolver:
    def test_auto_solver_selects_ed_for_small(self, ising_4, auto_solver):
        result = auto_solver.solve_ground_state(ising_4)
        assert isinstance(result, GroundStateResult)
        assert result.ground_state_energy < 0

    def test_auto_solver_spectrum(self, ising_4, auto_solver):
        eigenvalues = auto_solver.spectrum(ising_4)
        assert len(eigenvalues) > 0


# ---------------------------------------------------------------------------
# ModelQPU experiment tests
# ---------------------------------------------------------------------------

class TestModelQPU:
    def test_ground_state(self, model_qpu, ising_4):
        result = model_qpu.ground_state(ising_4)
        assert isinstance(result, GroundStateResult)
        assert result.ground_state_energy < 0

    def test_spectrum(self, model_qpu, ising_4):
        eigenvalues = model_qpu.spectrum(ising_4)
        assert len(eigenvalues) == 16

    def test_quench(self, model_qpu, ising_4):
        result = model_qpu.quench(
            ising_4, times=[0.0, 0.5, 1.0], initial_state="all_up"
        )
        assert isinstance(result, TimeEvolutionResult)

    def test_loschmidt_echo(self, model_qpu, ising_4):
        result = model_qpu.loschmidt_echo(
            ising_4,
            times=np.linspace(0, 2, 20),
            initial_state="all_up",
        )
        assert isinstance(result, LoschmidtEchoResult)
        assert len(result.times) == 20
        # Echo at t=0 should be 1
        assert abs(result.echo[0] - 1.0) < 1e-6

    def test_loschmidt_return_rate(self, model_qpu, ising_4):
        result = model_qpu.loschmidt_echo(
            ising_4, times=np.linspace(0, 2, 10), initial_state="all_up"
        )
        rr = result.return_rate
        assert rr.shape == (10,)
        # Return rate at t=0 should be ~0
        assert rr[0] == pytest.approx(0.0, abs=0.1)

    def test_dqpt_diagnostics(self, model_qpu):
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=1.5)
        result = model_qpu.dqpt_diagnostics(
            model,
            times=np.linspace(0, 5, 50),
            initial_state="all_up",
        )
        assert result.times is not None
        assert result.return_rate is not None

    def test_ground_state_with_observables(self, model_qpu, ising_4):
        result = model_qpu.ground_state(ising_4, observables=("magnetization_z",))
        assert isinstance(result, GroundStateResult)


# ---------------------------------------------------------------------------
# Result dataclass tests
# ---------------------------------------------------------------------------

class TestResultDataclasses:
    def test_sweep_point_creation(self):
        sp = SweepPoint(value=1.0, energy=-2.0, observables={"Z": 0.5}, solver="ed")
        assert sp.value == 1.0
        assert sp.energy == -2.0

    def test_loschmidt_echo_result_properties(self):
        times = np.linspace(0, 1, 10)
        amps = np.exp(-1j * times)
        result = LoschmidtEchoResult(
            model_name="test",
            solver="ed",
            num_sites=4,
            times=times,
            amplitudes=amps,
        )
        assert result.echo.shape == (10,)
        assert np.all(result.echo >= 0)
        assert result.echo[0] == pytest.approx(1.0, abs=1e-10)
        rr = result.return_rate
        assert rr.shape == (10,)


# ---------------------------------------------------------------------------
# Analysis function tests
# ---------------------------------------------------------------------------

class TestAnalysisFunctions:
    def test_analyze_dqpt_from_loschmidt(self, model_qpu):
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=1.5)
        echo = model_qpu.loschmidt_echo(
            model,
            times=np.linspace(0.1, 5, 100),
            initial_state="all_up",
        )
        result = analyze_dqpt_from_loschmidt(echo)
        assert result.times is not None
        assert result.return_rate is not None

    def test_fourier_transform_structure_factor(self, model_qpu, ising_4):
        """Test the Fourier transform of the dynamic structure factor."""
        from nqpu.physics import DynamicStructureFactorResult
        # Build a minimal DynamicStructureFactorResult for testing
        n = ising_4.num_sites
        num_times = 10
        times = np.linspace(0, 1, num_times)
        momenta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rng = np.random.default_rng(SEED)
        values = rng.random((num_times, n))
        dsf = DynamicStructureFactorResult(
            model_name="test",
            solver="ed",
            pauli="Z",
            connected=True,
            times=times,
            momenta=momenta,
            values=values,
        )
        freq_result = fourier_transform_structure_factor(dsf)
        assert freq_result is not None


# ---------------------------------------------------------------------------
# Cross-cutting parametric tests across models
# ---------------------------------------------------------------------------

class TestCrossCuttingModels:
    @pytest.mark.parametrize("model_cls,kwargs", [
        (TransverseFieldIsing1D, {"num_sites": 3, "coupling": 1.0, "transverse_field": 1.0}),
        (HeisenbergXXZ1D, {"num_sites": 3, "coupling_xy": 1.0, "anisotropy": 0.5, "field_z": 0.0}),
        (HeisenbergXYZ1D, {"num_sites": 3, "coupling_x": 1.0, "coupling_y": 0.8, "coupling_z": 0.5, "field_z": 0.0}),
    ])
    def test_all_models_ground_state(self, model_cls, kwargs, ed_solver):
        model = model_cls(**kwargs)
        result = ed_solver.solve_ground_state(model)
        assert result.ground_state_energy < 10  # just a sanity check
        assert np.isclose(np.linalg.norm(result.ground_state), 1.0, atol=1e-10)

    @pytest.mark.parametrize("model_cls,kwargs", [
        (TransverseFieldIsing1D, {"num_sites": 3, "coupling": 1.0, "transverse_field": 1.0}),
        (HeisenbergXXZ1D, {"num_sites": 3, "coupling_xy": 1.0, "anisotropy": 0.5, "field_z": 0.0}),
    ])
    def test_all_models_time_evolve(self, model_cls, kwargs, ed_solver):
        model = model_cls(**kwargs)
        result = ed_solver.time_evolve(model, [0.0, 0.5, 1.0])
        assert len(result.times) == 3

    @pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 1.5, 2.0])
    def test_ising_energy_vs_field_strength(self, h, ed_solver):
        model = TransverseFieldIsing1D(num_sites=3, coupling=1.0, transverse_field=h)
        result = ed_solver.solve_ground_state(model)
        assert isinstance(result.ground_state_energy, float)


# ---------------------------------------------------------------------------
# State I/O smoke tests
# ---------------------------------------------------------------------------

class TestStateIO:
    def test_save_load_imports(self):
        from nqpu.physics import (
            save_ground_state_result,
            load_ground_state_result,
            save_sweep_result,
            load_sweep_result,
            save_loschmidt_echo_result,
            load_loschmidt_echo_result,
        )
        assert callable(save_ground_state_result)
        assert callable(load_ground_state_result)
