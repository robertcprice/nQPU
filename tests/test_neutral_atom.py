"""Comprehensive tests for the nqpu.neutral_atom package.

Tests cover: AtomSpecies physics, AtomArray geometry and loading,
NeutralAtomGateSet compilation, NeutralAtomNoiseModel fidelity,
DevicePresets and DeviceSpec, NeutralAtomSimulator gate operations
and measurement.
"""

import math

import numpy as np
import pytest

from nqpu.neutral_atom import (
    ALL_SPECIES,
    ArrayConfig,
    ArrayGeometry,
    AtomArray,
    AtomSpecies,
    CircuitStats,
    DevicePresets,
    DeviceSpec,
    GateInstruction,
    NativeGateType,
    NeutralAtomGateSet,
    NeutralAtomNoiseModel,
    NeutralAtomSimulator,
    Zone,
    ZoneConfig,
)


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def rb87():
    return AtomSpecies.RB87


@pytest.fixture
def ideal_sim():
    """Small ideal-mode simulator for fast tests."""
    config = ArrayConfig(n_atoms=3, species=AtomSpecies.RB87)
    return NeutralAtomSimulator(config, execution_mode="ideal")


@pytest.fixture
def gate_set():
    return NeutralAtomGateSet()


@pytest.fixture
def small_array():
    return AtomArray(n_sites=9, species=AtomSpecies.RB87, spacing_um=4.0)


# ====================================================================
# AtomSpecies physics tests
# ====================================================================


class TestAtomSpecies:
    def test_all_species_has_four_entries(self):
        assert len(ALL_SPECIES) == 4

    def test_rb87_name_and_mass(self, rb87):
        assert rb87.name == "87Rb"
        assert rb87.mass_amu == pytest.approx(86.909180527)

    def test_mass_kg_positive(self, rb87):
        assert rb87.mass_kg > 0

    def test_blockade_radius_positive(self, rb87):
        r_b = rb87.blockade_radius_um(1.5)
        assert r_b > 0

    def test_blockade_radius_raises_on_negative_rabi(self, rb87):
        with pytest.raises(ValueError, match="positive"):
            rb87.blockade_radius_um(-1.0)

    def test_vdw_interaction_positive(self, rb87):
        v = rb87.vdw_interaction_hz(4.0)
        assert v > 0

    def test_vdw_interaction_raises_on_zero_distance(self, rb87):
        with pytest.raises(ValueError, match="positive"):
            rb87.vdw_interaction_hz(0.0)

    def test_blockade_fidelity_range(self, rb87):
        f = rb87.blockade_fidelity(4.0, 1.5)
        assert 0.0 <= f <= 1.0

    def test_thermal_dephasing_rate_positive(self, rb87):
        rate = rb87.thermal_dephasing_rate_hz(1.5)
        assert rate >= 0


# ====================================================================
# AtomArray tests
# ====================================================================


class TestAtomArray:
    def test_linear_geometry(self):
        arr = AtomArray(n_sites=5, geometry=ArrayGeometry.LINEAR, spacing_um=4.0)
        positions = arr.positions
        assert positions.shape == (5, 2)
        # All y-coordinates should be zero for linear
        assert np.allclose(positions[:, 1], 0.0)

    def test_rectangular_positions_shape(self, small_array):
        assert small_array.positions.shape == (9, 2)

    def test_distance_symmetric(self, small_array):
        d01 = small_array.distance(0, 1)
        d10 = small_array.distance(1, 0)
        assert d01 == pytest.approx(d10)

    def test_deterministic_load_fills_all(self, small_array):
        n = small_array.deterministic_load()
        assert n == 9
        assert small_array.n_atoms == 9

    def test_stochastic_load_returns_valid_count(self, small_array):
        rng = np.random.default_rng(42)
        n = small_array.stochastic_load(rng)
        assert 0 <= n <= 9

    def test_neighbours_returns_list(self, small_array):
        small_array.deterministic_load()
        nbrs = small_array.neighbours(0)
        assert isinstance(nbrs, list)

    def test_zone_assignment(self):
        zones = [
            ZoneConfig(zone_type=Zone.ENTANGLING, center_um=(0.0, 0.0), radius_um=100.0),
        ]
        arr = AtomArray(n_sites=4, geometry=ArrayGeometry.LINEAR, spacing_um=4.0, zones=zones)
        arr.assign_zones()
        sites = arr.sites_in_zone(0)
        assert len(sites) == 4

    def test_distance_matrix_shape(self, small_array):
        dm = small_array.distance_matrix()
        assert dm.shape == (9, 9)
        assert np.allclose(np.diag(dm), 0.0)

    def test_interaction_graph_empty_without_atoms(self, small_array):
        # No atoms loaded, so no interactions
        graph = small_array.interaction_graph(rabi_freq_mhz=1.0)
        assert len(graph) == 0


# ====================================================================
# NeutralAtomGateSet tests
# ====================================================================


class TestGateSet:
    def test_rz_matrix_unitary(self, gate_set):
        rz = gate_set.rz_matrix(0.5)
        identity = rz @ rz.conj().T
        assert np.allclose(identity, np.eye(2))

    def test_cz_matrix_diagonal(self, gate_set):
        cz = gate_set.cz_matrix()
        expected = np.diag([1.0, 1.0, 1.0, -1.0])
        assert np.allclose(cz, expected)

    def test_ccz_matrix_shape_and_last_element(self, gate_set):
        ccz = gate_set.ccz_matrix()
        assert ccz.shape == (8, 8)
        assert ccz[7, 7] == pytest.approx(-1.0)

    def test_compile_cnot_contains_cz(self, gate_set):
        instructions = gate_set.compile_cnot(0, 1)
        cz_count = sum(1 for i in instructions if i.gate_type == NativeGateType.CZ)
        assert cz_count == 1

    def test_compile_toffoli_contains_ccz(self, gate_set):
        instructions = gate_set.compile_toffoli(0, 1, 2)
        ccz_count = sum(1 for i in instructions if i.gate_type == NativeGateType.CCZ)
        assert ccz_count == 1

    def test_gate_count_functions(self, gate_set):
        instructions = gate_set.compile_swap(0, 1)
        assert gate_set.cz_gate_count(instructions) == 3
        assert gate_set.entangling_gate_count(instructions) == 3

    @pytest.mark.parametrize("theta", [0.0, math.pi / 4, math.pi, 2 * math.pi])
    def test_rx_matrix_unitary(self, gate_set, theta):
        rx = gate_set.rx_matrix(theta)
        assert np.allclose(rx @ rx.conj().T, np.eye(2), atol=1e-12)


# ====================================================================
# NeutralAtomSimulator tests
# ====================================================================


class TestSimulator:
    def test_initial_state_is_zero(self, ideal_sim):
        sv = ideal_sim.statevector()
        assert sv[0] == pytest.approx(1.0)
        assert np.linalg.norm(sv) == pytest.approx(1.0)

    def test_x_gate_flips_state(self, ideal_sim):
        ideal_sim.x(0)
        sv = ideal_sim.statevector()
        # After X on qubit 0 of |000>, state should be |100> = index 4 (MSB)
        assert abs(sv[4]) == pytest.approx(1.0, abs=1e-10)

    def test_hadamard_creates_superposition(self, ideal_sim):
        ideal_sim.h(0)
        sv = ideal_sim.statevector()
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)

    def test_cnot_creates_bell_state(self):
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        sim.h(0)
        sim.cnot(0, 1)
        sv = sim.statevector()
        # Bell state: (|00> + |11>) / sqrt(2)
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)
        assert abs(sv[3]) == pytest.approx(1 / math.sqrt(2), abs=1e-10)

    def test_measure_all_returns_correct_keys(self, ideal_sim):
        counts = ideal_sim.measure_all(shots=100)
        for key in counts:
            assert len(key) == 3  # 3 qubits -> 3-char bitstrings

    def test_measure_all_total_shots(self, ideal_sim):
        counts = ideal_sim.measure_all(shots=500)
        assert sum(counts.values()) == 500

    def test_reset_restores_zero_state(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cnot(0, 1)
        ideal_sim.reset()
        sv = ideal_sim.statevector()
        assert sv[0] == pytest.approx(1.0)

    def test_invalid_qubit_raises(self, ideal_sim):
        with pytest.raises(ValueError, match="out of range"):
            ideal_sim.h(10)

    def test_cnot_same_qubit_raises(self, ideal_sim):
        with pytest.raises(ValueError, match="different"):
            ideal_sim.cnot(0, 0)

    def test_run_circuit_bell(self):
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        circuit = [("h", 0), ("cx", 0, 1)]
        counts = sim.run_circuit(circuit, shots=1000)
        # Should see mostly "00" and "11"
        total = sum(counts.values())
        assert total == 1000
        dominant = counts.get("00", 0) + counts.get("11", 0)
        assert dominant > 900

    def test_circuit_stats_after_gates(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cnot(0, 1)
        stats = ideal_sim.circuit_stats()
        assert isinstance(stats, CircuitStats)
        assert stats.total_gates > 0
        assert stats.cz_gate_count >= 1

    def test_density_matrix_from_ideal(self, ideal_sim):
        dm = ideal_sim.density_matrix()
        assert dm.shape == (8, 8)
        assert dm[0, 0] == pytest.approx(1.0)

    def test_invalid_execution_mode_raises(self):
        with pytest.raises(ValueError, match="execution_mode"):
            NeutralAtomSimulator(ArrayConfig(n_atoms=2), execution_mode="invalid")

    def test_noisy_mode_uses_density_matrix(self):
        sim = NeutralAtomSimulator(
            ArrayConfig(n_atoms=2), execution_mode="noisy"
        )
        with pytest.raises(RuntimeError, match="noisy"):
            sim.statevector()

    def test_fidelity_estimate_ideal_is_one(self, ideal_sim):
        ideal_sim.h(0)
        assert ideal_sim.fidelity_estimate() == pytest.approx(1.0)


# ====================================================================
# NeutralAtomNoiseModel tests
# ====================================================================


class TestNoiseModel:
    def test_single_qubit_fidelity_range(self, rb87):
        nm = NeutralAtomNoiseModel(species=rb87, rabi_freq_mhz=1.5)
        f = nm.single_qubit_gate_fidelity()
        assert 0.0 < f <= 1.0

    def test_two_qubit_fidelity_less_than_single(self, rb87):
        nm = NeutralAtomNoiseModel(species=rb87, rabi_freq_mhz=1.5)
        f1 = nm.single_qubit_gate_fidelity()
        f2 = nm.two_qubit_gate_fidelity()
        assert f2 < f1

    def test_error_budget_has_required_keys(self, rb87):
        nm = NeutralAtomNoiseModel(species=rb87, rabi_freq_mhz=1.5)
        budget = nm.error_budget()
        assert "1q_fidelity" in budget
        assert "2q_fidelity" in budget
        assert "readout_error" in budget


# ====================================================================
# DevicePresets tests
# ====================================================================


class TestDevicePresets:
    def test_list_all_returns_four_devices(self):
        devices = DevicePresets.list_all()
        assert len(devices) == 4

    def test_quera_aquila_max_atoms(self):
        dev = DevicePresets.quera_aquila()
        assert dev.max_atoms == 256

    def test_device_create_array(self):
        dev = DevicePresets.quera_aquila()
        arr = dev.create_array(n_atoms=10)
        assert arr.n_sites == 10

    def test_device_create_array_exceeds_max_raises(self):
        dev = DevicePresets.pasqal_fresnel()
        with pytest.raises(ValueError, match="max"):
            dev.create_array(n_atoms=200)

    def test_device_info_keys(self):
        dev = DevicePresets.atom_computing_1225()
        info = dev.info()
        assert "name" in info
        assert "1q_fidelity" in info

    def test_device_noise_model(self):
        dev = DevicePresets.quera_aquila()
        nm = dev.create_noise_model()
        assert isinstance(nm, NeutralAtomNoiseModel)


# ====================================================================
# ArrayConfig validation tests
# ====================================================================


class TestArrayConfig:
    def test_valid_config(self):
        cfg = ArrayConfig(n_atoms=5)
        assert cfg.n_atoms == 5

    def test_zero_atoms_raises(self):
        with pytest.raises(ValueError, match="n_atoms"):
            ArrayConfig(n_atoms=0)

    def test_negative_spacing_raises(self):
        with pytest.raises(ValueError, match="spacing"):
            ArrayConfig(n_atoms=2, spacing_um=-1.0)

    def test_negative_rabi_raises(self):
        with pytest.raises(ValueError, match="rabi_freq"):
            ArrayConfig(n_atoms=2, rabi_freq_mhz=-0.5)
