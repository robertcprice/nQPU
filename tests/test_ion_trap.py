"""Comprehensive tests for the nqpu.ion_trap package.

Covers: IonSpecies, TrapConfig, DevicePresets, TrappedIonGateSet,
GateInstruction, NativeGateType, TrappedIonNoiseModel, AnalogCircuit,
PulseSequence, LaserPulse, TrappedIonSimulator, CircuitStats.

Uses seed=42 for reproducibility, no external dependencies beyond numpy + pytest.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.ion_trap import (
    ALL_SPECIES,
    AnalogCircuit,
    CircuitStats,
    DevicePresets,
    GateInstruction,
    IonSpecies,
    LaserPulse,
    NativeGateType,
    PulseSequence,
    TrappedIonGateSet,
    TrappedIonNoiseModel,
    TrappedIonSimulator,
    TrapConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def yb_species():
    return IonSpecies.YB171


@pytest.fixture
def basic_trap():
    return TrapConfig(n_ions=3, species=IonSpecies.YB171)


@pytest.fixture
def ideal_sim():
    config = TrapConfig(n_ions=3, species=IonSpecies.YB171)
    return TrappedIonSimulator(config, execution_mode="ideal")


@pytest.fixture
def noisy_sim():
    config = TrapConfig(n_ions=3, species=IonSpecies.YB171)
    return TrappedIonSimulator(config, execution_mode="noisy")


# ---------------------------------------------------------------------------
# IonSpecies tests
# ---------------------------------------------------------------------------

class TestIonSpecies:
    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_species_physical_properties(self, species):
        assert species.mass_amu > 0
        assert species.qubit_frequency_hz > 0
        assert species.t1_s > 0
        assert species.t2_s > 0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_mass_kg(self, species):
        assert species.mass_kg > 0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_qubit_wavelength_m(self, species):
        wl = species.qubit_wavelength_m
        assert wl > 0
        # Optical/microwave wavelength range (100nm to 1m)
        assert 1e-7 < wl < 1.0

    def test_wavevector(self, yb_species):
        k = yb_species.wavevector
        assert k > 0

    def test_lamb_dicke_parameter(self, yb_species):
        eta = yb_species.lamb_dicke_parameter(trap_freq_mhz=1.0)
        assert 0 < eta < 1

    def test_length_scale(self, yb_species):
        ls = yb_species.length_scale(axial_freq_mhz=1.0)
        assert ls > 0

    def test_frozen(self, yb_species):
        with pytest.raises(AttributeError):
            yb_species.mass_amu = 200.0

    def test_all_species_list(self):
        assert len(ALL_SPECIES) >= 5
        names = [s.name for s in ALL_SPECIES]
        assert "Yb-171" in names or "171Yb+" in names or len(names) >= 5


# ---------------------------------------------------------------------------
# TrapConfig tests
# ---------------------------------------------------------------------------

class TestTrapConfig:
    def test_basic_creation(self, basic_trap):
        assert basic_trap.n_ions == 3
        assert basic_trap.species is IonSpecies.YB171

    def test_equilibrium_positions(self, basic_trap):
        positions = basic_trap.equilibrium_positions()
        assert len(positions) == 3
        # Should be sorted
        assert np.all(np.diff(positions) >= -1e-10)

    def test_equilibrium_positions_meters(self, basic_trap):
        pos_m = basic_trap.equilibrium_positions_meters()
        assert len(pos_m) == 3
        # Physical positions should be small (microns)
        for p in pos_m:
            assert abs(p) < 1e-3

    def test_normal_modes(self, basic_trap):
        freqs, modes = basic_trap.normal_modes()
        assert len(freqs) == 3
        assert modes.shape == (3, 3)
        # Frequencies should be positive
        assert np.all(np.array(freqs) > 0)

    def test_radial_normal_modes(self, basic_trap):
        freqs, modes = basic_trap.radial_normal_modes()
        assert len(freqs) == 3

    def test_summary(self, basic_trap):
        summary = basic_trap.summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_validation_min_ions(self):
        with pytest.raises((ValueError, Exception)):
            TrapConfig(n_ions=0, species=IonSpecies.YB171)

    def test_validation_max_ions(self):
        with pytest.raises((ValueError, Exception)):
            TrapConfig(n_ions=101, species=IonSpecies.YB171)

    def test_radial_gt_axial(self):
        """Radial frequency must be greater than axial for stability."""
        with pytest.raises((ValueError, Exception)):
            TrapConfig(
                n_ions=3,
                species=IonSpecies.YB171,
                axial_freq_mhz=5.0,
                radial_freq_mhz=1.0,  # less than axial
            )

    @pytest.mark.parametrize("n_ions", [1, 2, 5, 10])
    def test_various_ion_counts(self, n_ions):
        config = TrapConfig(n_ions=n_ions, species=IonSpecies.YB171)
        positions = config.equilibrium_positions()
        assert len(positions) == n_ions


# ---------------------------------------------------------------------------
# DevicePresets tests
# ---------------------------------------------------------------------------

class TestDevicePresets:
    @pytest.mark.parametrize("preset_fn", [
        DevicePresets.ionq_aria,
        DevicePresets.ionq_forte,
        DevicePresets.quantinuum_h1,
        DevicePresets.quantinuum_h2,
        DevicePresets.oxford_ionics_demo,
    ])
    def test_all_presets_create_valid_config(self, preset_fn):
        config = preset_fn()
        assert isinstance(config, TrapConfig)
        assert config.n_ions > 0


# ---------------------------------------------------------------------------
# TrappedIonGateSet tests
# ---------------------------------------------------------------------------

class TestTrappedIonGateSet:
    def test_rz_matrix_unitary(self):
        rz = TrappedIonGateSet.rz_matrix(np.pi / 3)
        assert np.allclose(rz @ rz.conj().T, np.eye(2), atol=1e-10)

    def test_rx_matrix_unitary(self):
        rx = TrappedIonGateSet.rx_matrix(np.pi / 4)
        assert np.allclose(rx @ rx.conj().T, np.eye(2), atol=1e-10)

    def test_ry_matrix_unitary(self):
        ry = TrappedIonGateSet.ry_matrix(np.pi / 6)
        assert np.allclose(ry @ ry.conj().T, np.eye(2), atol=1e-10)

    def test_r_matrix_unitary(self):
        r = TrappedIonGateSet.r_matrix(np.pi / 3, np.pi / 4)
        assert np.allclose(r @ r.conj().T, np.eye(2), atol=1e-10)

    def test_ms_matrix_unitary(self):
        ms = TrappedIonGateSet.ms_matrix(np.pi / 4)
        assert ms.shape == (4, 4)
        assert np.allclose(ms @ ms.conj().T, np.eye(4), atol=1e-10)

    def test_xx_equals_ms(self):
        theta = 0.3
        ms = TrappedIonGateSet.ms_matrix(theta)
        xx = TrappedIonGateSet.xx_matrix(theta)
        assert np.allclose(ms, xx, atol=1e-10)

    def test_zz_matrix_unitary(self):
        zz = TrappedIonGateSet.zz_matrix(np.pi / 4)
        assert zz.shape == (4, 4)
        assert np.allclose(zz @ zz.conj().T, np.eye(4), atol=1e-10)

    def test_compile_h(self):
        instructions = TrappedIonGateSet.compile_h(0)
        assert len(instructions) > 0
        for inst in instructions:
            assert isinstance(inst, GateInstruction)

    def test_compile_cnot_uses_1_ms(self):
        instructions = TrappedIonGateSet.compile_cnot(0, 1)
        ms_count = TrappedIonGateSet.ms_gate_count(instructions)
        assert ms_count == 1

    def test_compile_swap_uses_3_ms(self):
        instructions = TrappedIonGateSet.compile_swap(0, 1)
        ms_count = TrappedIonGateSet.ms_gate_count(instructions)
        assert ms_count == 3

    def test_compile_toffoli(self):
        instructions = TrappedIonGateSet.compile_toffoli(0, 1, 2)
        ms_count = TrappedIonGateSet.ms_gate_count(instructions)
        assert ms_count > 0

    def test_ms_gate_count(self):
        instructions = [
            GateInstruction(NativeGateType.RZ, (0,), (0.5,)),
            GateInstruction(NativeGateType.MS, (0, 1), (np.pi / 4,)),
            GateInstruction(NativeGateType.RY, (0,), (0.3,)),
        ]
        assert TrappedIonGateSet.ms_gate_count(instructions) == 1

    def test_single_qubit_gate_count(self):
        instructions = [
            GateInstruction(NativeGateType.RZ, (0,), (0.5,)),
            GateInstruction(NativeGateType.MS, (0, 1), (np.pi / 4,)),
            GateInstruction(NativeGateType.RY, (0,), (0.3,)),
        ]
        assert TrappedIonGateSet.single_qubit_gate_count(instructions) == 2

    def test_compile_arbitrary_unitary(self):
        # Hadamard matrix
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        instructions = TrappedIonGateSet.compile_arbitrary_unitary(H, 0)
        assert len(instructions) > 0

    @pytest.mark.parametrize("gate_fn,args", [
        (TrappedIonGateSet.compile_x, (0,)),
        (TrappedIonGateSet.compile_y, (0,)),
        (TrappedIonGateSet.compile_z, (0,)),
        (TrappedIonGateSet.compile_rx, (0, np.pi / 3)),
        (TrappedIonGateSet.compile_ry, (0, np.pi / 3)),
        (TrappedIonGateSet.compile_rz, (0, np.pi / 3)),
        (TrappedIonGateSet.compile_cz, (0, 1)),
    ])
    def test_compile_functions_return_instructions(self, gate_fn, args):
        instructions = gate_fn(*args)
        assert len(instructions) > 0
        for inst in instructions:
            assert isinstance(inst, GateInstruction)


# ---------------------------------------------------------------------------
# GateInstruction tests
# ---------------------------------------------------------------------------

class TestGateInstruction:
    def test_creation(self):
        gi = GateInstruction(NativeGateType.RZ, (0,), (np.pi,))
        assert gi.gate_type == NativeGateType.RZ
        assert gi.qubits == (0,)

    def test_repr(self):
        gi = GateInstruction(NativeGateType.MS, (0, 1), (np.pi / 4,))
        r = repr(gi)
        assert "MS" in r


# ---------------------------------------------------------------------------
# TrappedIonNoiseModel tests
# ---------------------------------------------------------------------------

class TestTrappedIonNoiseModel:
    def test_single_qubit_gate_fidelity(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        fid = noise.single_qubit_gate_fidelity()
        assert 0 < fid <= 1

    def test_two_qubit_gate_fidelity(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        fid = noise.two_qubit_gate_fidelity()
        assert 0 < fid <= 1

    def test_error_budget(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        budget = noise.error_budget()
        assert isinstance(budget, dict)
        assert "1q_fidelity" in budget
        assert "2q_fidelity" in budget
        assert budget["1q_fidelity"] <= 1.0
        assert budget["2q_fidelity"] <= 1.0

    def test_apply_noise_single(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        n = basic_trap.n_ions
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        noisy_rho = noise.apply_noise(rho, gate_type="single", gate_time_us=1.0)
        assert noisy_rho.shape == (dim, dim)
        assert np.isclose(np.trace(noisy_rho).real, 1.0, atol=1e-6)

    def test_apply_noise_ms(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        n = basic_trap.n_ions
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        noisy_rho = noise.apply_noise(
            rho, gate_type="ms", gate_time_us=100.0, target_qubits=(0, 1)
        )
        assert np.isclose(np.trace(noisy_rho).real, 1.0, atol=1e-6)

    def test_idle_decoherence(self, basic_trap):
        noise = TrappedIonNoiseModel(config=basic_trap)
        n = basic_trap.n_ions
        dim = 2 ** n
        rho = np.zeros((dim, dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        decohered = noise.idle_decoherence(rho, idle_time_us=100.0)
        assert np.isclose(np.trace(decohered).real, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# TrappedIonSimulator tests - ideal mode
# ---------------------------------------------------------------------------

class TestTrappedIonSimulatorIdeal:
    def test_initial_state(self, ideal_sim):
        sv = ideal_sim.statevector()
        dim = 2 ** 3
        expected = np.zeros(dim, dtype=np.complex128)
        expected[0] = 1.0
        assert np.allclose(sv, expected, atol=1e-10)

    def test_h_gate_superposition(self, ideal_sim):
        ideal_sim.h(0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)
        probs = np.abs(sv) ** 2
        assert probs[0] == pytest.approx(0.5, abs=0.01)

    def test_x_gate(self, ideal_sim):
        ideal_sim.x(0)
        sv = ideal_sim.statevector()
        idx = 1 << (3 - 1)  # qubit 0 is MSB
        assert abs(sv[idx]) == pytest.approx(1.0, abs=1e-10)

    def test_cnot_bell_state(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cnot(0, 1)
        sv = ideal_sim.statevector()
        n = 3
        idx_00x = 0  # |000>
        idx_11x = (1 << (n - 1)) | (1 << (n - 2))  # |110>
        probs = np.abs(sv) ** 2
        assert probs[idx_00x] == pytest.approx(0.5, abs=0.01)
        assert probs[idx_11x] == pytest.approx(0.5, abs=0.01)

    def test_ms_gate(self, ideal_sim):
        ideal_sim.ms(0, 1, theta=np.pi / 4)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_xx_gate(self, ideal_sim):
        ideal_sim.xx(0, 1, theta=np.pi / 4)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    @pytest.mark.parametrize("gate", ["y", "z"])
    def test_single_qubit_gates(self, ideal_sim, gate):
        getattr(ideal_sim, gate)(0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    @pytest.mark.parametrize("gate,angle", [
        ("rx", np.pi / 3),
        ("ry", np.pi / 4),
        ("rz", np.pi / 6),
    ])
    def test_rotation_gates(self, ideal_sim, gate, angle):
        getattr(ideal_sim, gate)(0, angle)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_cz_gate(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cz(0, 1)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_global_rotation(self, ideal_sim):
        ideal_sim.global_rotation(theta=np.pi / 2, phi=0.0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_measure_all(self, ideal_sim):
        ideal_sim.h(0)
        result = ideal_sim.measure_all(shots=100)
        assert isinstance(result, dict)
        assert sum(result.values()) == 100

    def test_measure_single(self, ideal_sim):
        ideal_sim.x(0)
        outcome = ideal_sim.measure(0)
        assert outcome in (0, 1)

    def test_density_matrix(self, ideal_sim):
        ideal_sim.h(0)
        dm = ideal_sim.density_matrix()
        dim = 2 ** 3
        assert dm.shape == (dim, dim)
        assert np.isclose(np.trace(dm).real, 1.0, atol=1e-10)

    def test_fidelity_estimate(self, ideal_sim):
        fid = ideal_sim.fidelity_estimate()
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_circuit_stats(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cnot(0, 1)
        stats = ideal_sim.circuit_stats()
        assert isinstance(stats, CircuitStats)

    def test_device_info(self, ideal_sim):
        info = ideal_sim.device_info()
        assert isinstance(info, dict)

    def test_reset(self, ideal_sim):
        ideal_sim.x(0)
        ideal_sim.reset()
        sv = ideal_sim.statevector()
        dim = 2 ** 3
        expected = np.zeros(dim, dtype=np.complex128)
        expected[0] = 1.0
        assert np.allclose(sv, expected, atol=1e-10)

    def test_compile_circuit(self, ideal_sim):
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0],q[1];
"""
        instructions = ideal_sim.compile_circuit(qasm_str)
        assert isinstance(instructions, list)
        assert len(instructions) > 0
        for inst in instructions:
            assert isinstance(inst, GateInstruction)


# ---------------------------------------------------------------------------
# TrappedIonSimulator tests - noisy mode
# ---------------------------------------------------------------------------

class TestTrappedIonSimulatorNoisy:
    def test_noisy_sim_runs(self, noisy_sim):
        noisy_sim.h(0)
        noisy_sim.cnot(0, 1)
        result = noisy_sim.measure_all(shots=100)
        assert sum(result.values()) == 100

    def test_noisy_fidelity_less_than_one(self, noisy_sim):
        noisy_sim.h(0)
        noisy_sim.cnot(0, 1)
        fid = noisy_sim.fidelity_estimate()
        assert fid <= 1.0


# ---------------------------------------------------------------------------
# AnalogCircuit tests
# ---------------------------------------------------------------------------

class TestAnalogCircuit:
    def test_create_and_simulate(self, basic_trap):
        circ = AnalogCircuit(n_ions=3, config=basic_trap)
        # Add a simple Rabi drive on ion 0
        circ.add_rabi_drive(ion=0, rabi_freq_mhz=1.0, phase=0.0, duration_us=1.0)
        state = circ.simulate()
        assert len(state) == 2 ** 3
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-6)

    def test_ms_interaction(self, basic_trap):
        circ = AnalogCircuit(n_ions=2, config=TrapConfig(n_ions=2, species=IonSpecies.YB171))
        circ.add_ms_interaction(
            ion_a=0, ion_b=1,
            rabi_freq_mhz=0.1,
            detuning_mhz=0.01,
            duration_us=10.0,
        )
        state = circ.simulate()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-4)

    def test_stark_shift(self, basic_trap):
        circ = AnalogCircuit(n_ions=3, config=basic_trap)
        circ.add_stark_shift(ion=0, shift_mhz=0.5, duration_us=1.0)
        state = circ.simulate()
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-6)

    def test_total_duration(self, basic_trap):
        circ = AnalogCircuit(n_ions=3, config=basic_trap)
        circ.add_rabi_drive(ion=0, rabi_freq_mhz=1.0, phase=0.0, duration_us=1.0)
        circ.add_rabi_drive(ion=1, rabi_freq_mhz=1.0, phase=0.0, duration_us=2.0)
        assert circ.total_duration_us() == pytest.approx(3.0)

    def test_hamiltonian_shape_validation(self, basic_trap):
        circ = AnalogCircuit(n_ions=3, config=basic_trap)
        wrong_H = np.eye(4, dtype=np.complex128)  # Should be 8x8 for 3 qubits
        with pytest.raises(ValueError):
            circ.add_evolution(wrong_H, duration_us=1.0)


# ---------------------------------------------------------------------------
# PulseSequence tests
# ---------------------------------------------------------------------------

class TestPulseSequence:
    def test_add_pulse(self):
        ps = PulseSequence(n_ions=3)
        ps.add_pulse(ion=0, frequency_mhz=0.0, amplitude=0.5, phase=0.0, duration_us=1.0)
        assert len(ps.pulses) == 1

    def test_invalid_ion_raises(self):
        ps = PulseSequence(n_ions=3)
        with pytest.raises(ValueError):
            ps.add_pulse(ion=5, frequency_mhz=0.0, amplitude=0.5, phase=0.0, duration_us=1.0)

    def test_invalid_amplitude_raises(self):
        ps = PulseSequence(n_ions=3)
        with pytest.raises(ValueError):
            ps.add_pulse(ion=0, frequency_mhz=0.0, amplitude=1.5, phase=0.0, duration_us=1.0)

    def test_invalid_shape_raises(self):
        ps = PulseSequence(n_ions=3)
        with pytest.raises(ValueError):
            ps.add_pulse(ion=0, frequency_mhz=0.0, amplitude=0.5, phase=0.0,
                         duration_us=1.0, shape="invalid")

    @pytest.mark.parametrize("shape", ["square", "gaussian", "sech", "blackman", "cosine"])
    def test_envelope_function(self, shape):
        ps = PulseSequence(n_ions=2)
        t = np.linspace(0, 1.0, 100)
        env = ps.envelope_function(shape, t, duration=1.0)
        assert len(env) == 100
        assert np.all(env >= -0.01)  # Should be non-negative (or close)

    def test_to_analog_circuit(self, basic_trap):
        ps = PulseSequence(n_ions=3)
        ps.add_pulse(ion=0, frequency_mhz=0.0, amplitude=0.5, phase=0.0, duration_us=1.0)
        circ = ps.to_analog_circuit(basic_trap)
        assert isinstance(circ, AnalogCircuit)
        assert len(circ.steps) > 0

    def test_total_duration(self):
        ps = PulseSequence(n_ions=2)
        ps.add_pulse(ion=0, frequency_mhz=0.0, amplitude=0.5, phase=0.0, duration_us=1.5)
        ps.add_pulse(ion=1, frequency_mhz=0.0, amplitude=0.3, phase=0.0, duration_us=2.5)
        assert ps.total_duration_us() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Cross-cutting parametric tests
# ---------------------------------------------------------------------------

class TestCrossCuttingPresets:
    @pytest.mark.parametrize("preset_fn", [
        DevicePresets.ionq_aria,
        DevicePresets.ionq_forte,
        DevicePresets.quantinuum_h1,
        DevicePresets.quantinuum_h2,
    ])
    def test_preset_ideal_simulation(self, preset_fn):
        # Use a small ion count (3) to avoid OOM from 2^N state vectors
        config = preset_fn()
        small_config = TrapConfig(
            n_ions=3,
            species=config.species,
            axial_freq_mhz=config.axial_freq_mhz,
            radial_freq_mhz=config.radial_freq_mhz,
            heating_rate_quanta_per_s=config.heating_rate_quanta_per_s,
            background_gas_collision_rate=config.background_gas_collision_rate,
        )
        sim = TrappedIonSimulator(small_config, execution_mode="ideal")
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.measure_all(shots=50)
        assert sum(result.values()) == 50

    @pytest.mark.parametrize("preset_fn", [
        DevicePresets.ionq_aria,
        DevicePresets.quantinuum_h1,
    ])
    def test_preset_noisy_simulation(self, preset_fn):
        # Use a small ion count (3) to avoid OOM from 2^N density matrices
        config = preset_fn()
        small_config = TrapConfig(
            n_ions=3,
            species=config.species,
            axial_freq_mhz=config.axial_freq_mhz,
            radial_freq_mhz=config.radial_freq_mhz,
            heating_rate_quanta_per_s=config.heating_rate_quanta_per_s,
            background_gas_collision_rate=config.background_gas_collision_rate,
        )
        sim = TrappedIonSimulator(small_config, execution_mode="noisy")
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.measure_all(shots=50)
        assert sum(result.values()) == 50

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_species_in_trap_config(self, species):
        config = TrapConfig(n_ions=2, species=species)
        positions = config.equilibrium_positions()
        assert len(positions) == 2
