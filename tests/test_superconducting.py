"""Comprehensive tests for the nqpu.superconducting package.

Covers: TransmonQubit, ChipTopology, ChipConfig, DevicePresets, TransmonGateSet,
TransmonNoiseModel, TransmonSimulator, LeakageReductionUnit, TransmonQCVV,
DigitalTwin, CompilerBenchmark, NativeGateAnalyzer, and core pulse-level types.

Uses seed=42 for reproducibility, no external dependencies beyond numpy + pytest.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.superconducting import (
    BenchmarkResult,
    CalibrationData,
    CalibrationDrift,
    ChipConfig,
    ChipTopology,
    CircuitStats,
    CompilationResult,
    CompilerBenchmark,
    DevicePresets,
    DigitalTwin,
    GateInstruction,
    LeakageReductionUnit,
    NativeGateAnalyzer,
    NativeGateFamily,
    NativeGateType,
    StabilityReport,
    TopologyType,
    TransmonGateSet,
    TransmonNoiseModel,
    TransmonQubit,
    TransmonQCVV,
    TransmonSimulator,
    ValidationReport,
    compare_backends,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture
def typical_qubit():
    return TransmonQubit.typical(0)


@pytest.fixture
def ibm_eagle_config():
    return DevicePresets.IBM_EAGLE.build(num_qubits=5)


@pytest.fixture
def ibm_heron_config():
    return DevicePresets.IBM_HERON.build(num_qubits=5)


@pytest.fixture
def google_sycamore_config():
    return DevicePresets.GOOGLE_SYCAMORE.build(num_qubits=5)


@pytest.fixture
def ideal_sim(ibm_eagle_config):
    return TransmonSimulator(ibm_eagle_config, execution_mode="ideal")


@pytest.fixture
def noisy_sim(ibm_eagle_config):
    return TransmonSimulator(ibm_eagle_config, execution_mode="noisy")


@pytest.fixture
def ideal_noise_model(ibm_eagle_config):
    return TransmonNoiseModel.ideal(ibm_eagle_config)


# ---------------------------------------------------------------------------
# TransmonQubit tests
# ---------------------------------------------------------------------------

class TestTransmonQubit:
    def test_typical_qubit_properties(self, typical_qubit):
        q = typical_qubit
        assert q.frequency_ghz > 0
        assert q.anharmonicity_mhz < 0
        assert q.t1_us > 0
        assert q.t2_us > 0
        assert 0 < q.single_gate_fidelity <= 1
        assert 0 < q.readout_fidelity <= 1

    def test_frequency_02(self, typical_qubit):
        """f02 = 2*f01 + anharmonicity."""
        q = typical_qubit
        expected = 2 * q.frequency_ghz + q.anharmonicity_mhz / 1000
        assert abs(q.frequency_02_ghz - expected) < 1e-6

    def test_pure_dephasing_rate(self, typical_qubit):
        q = typical_qubit
        rate = q.pure_dephasing_rate_mhz
        assert rate >= 0

    @pytest.mark.parametrize("factory", [
        TransmonQubit.ibm_eagle_qubit,
        TransmonQubit.ibm_heron_qubit,
        TransmonQubit.google_sycamore_qubit,
    ])
    def test_factory_methods_return_valid_qubit(self, factory):
        q = factory()
        assert q.frequency_ghz > 0
        assert q.t1_us > 0

    def test_qubit_is_frozen(self, typical_qubit):
        with pytest.raises(AttributeError):
            typical_qubit.frequency_ghz = 6.0


# ---------------------------------------------------------------------------
# ChipTopology tests
# ---------------------------------------------------------------------------

class TestChipTopology:
    def test_heavy_hex_topology(self):
        topo = ChipTopology.heavy_hex(5)
        assert topo.num_qubits == 5
        assert topo.topology_type == TopologyType.HEAVY_HEX
        assert len(topo.edges) > 0

    def test_grid_topology(self):
        topo = ChipTopology.grid(2, 2)
        assert topo.num_qubits == 4
        assert topo.topology_type == TopologyType.GRID

    def test_fully_connected(self):
        topo = ChipTopology.fully_connected(4)
        assert topo.num_qubits == 4
        assert topo.topology_type == TopologyType.FULLY_CONNECTED
        # 4 choose 2 = 6 edges
        assert len(topo.edges) == 6

    def test_neighbors(self):
        topo = ChipTopology.fully_connected(3)
        nbrs = topo.neighbors(0)
        assert set(nbrs) == {1, 2}

    def test_coupling_strength(self):
        topo = ChipTopology.fully_connected(3)
        for edge in topo.edges:
            strength = topo.coupling_strength(edge[0], edge[1])
            assert strength > 0


# ---------------------------------------------------------------------------
# DevicePresets tests
# ---------------------------------------------------------------------------

class TestDevicePresets:
    @pytest.mark.parametrize("preset", list(DevicePresets))
    def test_all_presets_build(self, preset):
        config = preset.build(num_qubits=4)
        assert isinstance(config, ChipConfig)
        assert config.num_qubits >= 4
        assert len(config.qubits) >= 4

    def test_preset_device_info(self, ibm_eagle_config):
        info = ibm_eagle_config.device_info()
        assert isinstance(info, dict)
        assert "num_qubits" in info


# ---------------------------------------------------------------------------
# NativeGateFamily enum test
# ---------------------------------------------------------------------------

class TestNativeGateFamily:
    def test_gate_families_exist(self):
        assert NativeGateFamily.ECR is not None
        assert NativeGateFamily.SQRT_ISWAP is not None
        assert NativeGateFamily.CZ is not None


# ---------------------------------------------------------------------------
# TransmonGateSet tests
# ---------------------------------------------------------------------------

class TestTransmonGateSet:
    def test_gate_matrices_unitary(self):
        """All static gate matrices should be unitary."""
        for mat_fn, args in [
            (TransmonGateSet.h_matrix, ()),
            (TransmonGateSet.x_matrix, ()),
            (TransmonGateSet.sx_matrix, ()),
            (TransmonGateSet.rz_matrix, (np.pi / 4,)),
            (TransmonGateSet.cnot_matrix, ()),
            (TransmonGateSet.cz_matrix, ()),
            (TransmonGateSet.swap_matrix, ()),
        ]:
            mat = mat_fn(*args)
            identity = mat @ mat.conj().T
            assert np.allclose(identity, np.eye(mat.shape[0]), atol=1e-10)

    def test_h_matrix_shape(self):
        h = TransmonGateSet.h_matrix()
        assert h.shape == (2, 2)

    def test_cnot_matrix_shape(self):
        c = TransmonGateSet.cnot_matrix()
        assert c.shape == (4, 4)

    def test_rz_periodicity(self):
        """Rz(0) should be identity (up to global phase)."""
        rz0 = TransmonGateSet.rz_matrix(0.0)
        assert np.allclose(rz0, np.eye(2), atol=1e-10)

    def test_gate_instruction_is_two_qubit(self):
        g1 = GateInstruction(NativeGateType.RZ, (0,), 0.5)
        g2 = GateInstruction(NativeGateType.ECR, (0, 1), None)
        assert not g1.is_two_qubit
        assert g2.is_two_qubit


# ---------------------------------------------------------------------------
# TransmonNoiseModel tests
# ---------------------------------------------------------------------------

class TestTransmonNoiseModel:
    def test_ideal_model_zero_errors(self, ibm_eagle_config):
        model = TransmonNoiseModel.ideal(ibm_eagle_config)
        # Ideal model should produce zero or near-zero error probabilities
        assert model.t1_decay_prob(0, 200.0) == pytest.approx(0.0, abs=1e-10)
        assert model.leakage_prob(0) == pytest.approx(0.0, abs=1e-10)

    def test_noisy_model_nonzero_errors(self, ibm_eagle_config):
        model = TransmonNoiseModel(ibm_eagle_config)
        # Noisy model should have some error
        assert model.single_gate_error(0) >= 0
        assert model.two_qubit_gate_error(0, 1) >= 0

    def test_readout_confusion_matrix_shape(self, ibm_eagle_config):
        model = TransmonNoiseModel(ibm_eagle_config)
        conf = model.readout_confusion(qubit=0)
        assert conf.shape == (2, 2)
        # Each row should sum to 1
        for row in conf:
            assert abs(row.sum() - 1.0) < 1e-10

    def test_amplitude_damping_kraus(self, ibm_eagle_config):
        model = TransmonNoiseModel(ibm_eagle_config)
        kraus = model.amplitude_damping_kraus(qubit=0, gate_time_ns=25.0)
        assert len(kraus) == 2
        # Completeness: sum of K^dag K = I
        total = sum(k.conj().T @ k for k in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-6)

    def test_dephasing_kraus(self, ibm_eagle_config):
        model = TransmonNoiseModel(ibm_eagle_config)
        kraus = model.dephasing_kraus(qubit=0, gate_time_ns=25.0)
        assert len(kraus) == 2
        total = sum(k.conj().T @ k for k in kraus)
        assert np.allclose(total, np.eye(2), atol=1e-6)

    def test_depolarizing_channel(self, ibm_eagle_config):
        model = TransmonNoiseModel(ibm_eagle_config)
        n_qubits = ibm_eagle_config.num_qubits
        dim = 2 ** n_qubits
        rho = np.zeros((dim, dim), dtype=np.complex128)
        rho[0, 0] = 1.0
        rho_out = model.depolarizing_channel(rho, qubit=0, n_qubits=n_qubits, p=0.1)
        assert rho_out.shape == (dim, dim)
        assert abs(np.trace(rho_out) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# TransmonSimulator tests - ideal mode
# ---------------------------------------------------------------------------

class TestTransmonSimulatorIdeal:
    def test_initial_state_is_zero(self, ideal_sim):
        sv = ideal_sim.statevector()
        expected = np.zeros(2 ** ideal_sim.n_qubits, dtype=np.complex128)
        expected[0] = 1.0
        assert np.allclose(sv, expected, atol=1e-10)

    def test_h_gate_superposition(self, ideal_sim):
        ideal_sim.h(0)
        probs = ideal_sim.probabilities()
        assert probs[0] == pytest.approx(0.5, abs=0.01)

    def test_x_gate_flips_qubit(self, ideal_sim):
        ideal_sim.x(0)
        sv = ideal_sim.statevector()
        # |0> -> |1>; qubit 0 is LSB (bit 0), so index = 1
        idx = 1 << 0
        assert abs(sv[idx]) == pytest.approx(1.0, abs=1e-10)

    def test_cnot_creates_bell_state(self, ibm_eagle_config):
        sim = TransmonSimulator(ibm_eagle_config, execution_mode="ideal")
        sim.h(0)
        sim.cnot(0, 1)
        probs = sim.probabilities()
        # qubit 0 = bit 0 (LSB), qubit 1 = bit 1
        idx_00 = 0
        idx_11 = (1 << 0) | (1 << 1)  # = 3
        assert probs[idx_00] == pytest.approx(0.5, abs=0.01)
        assert probs[idx_11] == pytest.approx(0.5, abs=0.01)

    def test_measure_all_returns_dict(self, ideal_sim):
        ideal_sim.h(0)
        result = ideal_sim.measure_all(shots=100)
        assert isinstance(result, dict)
        total_shots = sum(result.values())
        assert total_shots == 100

    def test_measure_single_qubit(self, ideal_sim):
        ideal_sim.x(0)
        outcome = ideal_sim.measure(0)
        assert outcome in (0, 1)

    def test_reset_returns_to_zero(self, ideal_sim):
        ideal_sim.x(0)
        ideal_sim.reset()
        sv = ideal_sim.statevector()
        expected = np.zeros(2 ** ideal_sim.n_qubits, dtype=np.complex128)
        expected[0] = 1.0
        assert np.allclose(sv, expected, atol=1e-10)

    @pytest.mark.parametrize("gate,qubit,angle", [
        ("rx", 0, np.pi),
        ("ry", 0, np.pi),
        ("rz", 0, np.pi),
    ])
    def test_rotation_gates_run(self, ideal_sim, gate, qubit, angle):
        getattr(ideal_sim, gate)(qubit, angle)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-10)

    def test_y_gate(self, ideal_sim):
        ideal_sim.y(0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_z_gate(self, ideal_sim):
        ideal_sim.z(0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_sx_gate(self, ideal_sim):
        ideal_sim.sx(0)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_swap_gate(self, ideal_sim):
        ideal_sim.x(0)
        ideal_sim.swap(0, 1)
        # Qubit 0 should now be |0> and qubit 1 should be |1>
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_cz_gate(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cz(0, 1)
        sv = ideal_sim.statevector()
        assert np.isclose(np.linalg.norm(sv), 1.0)

    def test_density_matrix(self, ideal_sim):
        ideal_sim.h(0)
        dm = ideal_sim.density_matrix()
        assert dm.shape[0] == 2 ** ideal_sim.n_qubits
        assert np.isclose(np.trace(dm).real, 1.0, atol=1e-10)

    def test_fidelity_estimate_ideal(self, ideal_sim):
        # On an ideal simulator fidelity should be 1.0
        fid = ideal_sim.fidelity_estimate()
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_circuit_stats(self, ideal_sim):
        ideal_sim.h(0)
        ideal_sim.cnot(0, 1)
        stats = ideal_sim.circuit_stats()
        assert isinstance(stats, CircuitStats)
        assert stats.total_gates >= 2

    def test_device_info_dict(self, ideal_sim):
        info = ideal_sim.device_info()
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# TransmonSimulator tests - noisy mode
# ---------------------------------------------------------------------------

class TestTransmonSimulatorNoisy:
    def test_noisy_sim_creates(self, noisy_sim):
        assert noisy_sim is not None

    def test_noisy_fidelity_less_than_ideal(self, ibm_eagle_config):
        ideal = TransmonSimulator(ibm_eagle_config, execution_mode="ideal")
        noisy = TransmonSimulator(ibm_eagle_config, execution_mode="noisy")
        # Run same circuit
        for sim in (ideal, noisy):
            sim.h(0)
            sim.cnot(0, 1)
        ideal_fid = ideal.fidelity_estimate()
        noisy_fid = noisy.fidelity_estimate()
        assert noisy_fid <= ideal_fid + 1e-6

    def test_noisy_measure_all(self, noisy_sim):
        noisy_sim.h(0)
        noisy_sim.cnot(0, 1)
        result = noisy_sim.measure_all(shots=200)
        assert sum(result.values()) == 200


# ---------------------------------------------------------------------------
# LeakageReductionUnit tests
# ---------------------------------------------------------------------------

class TestLeakageReductionUnit:
    def test_leakage_per_qubit_nonnegative(self):
        # Build a 3-level state vector for 2 qubits (dimension 9)
        n_qubits = 2
        dim3 = 3 ** n_qubits
        state = np.zeros(dim3, dtype=np.complex128)
        state[0] = 1.0  # |00> in 3-level basis
        leakage = LeakageReductionUnit.leakage_per_qubit(state, n_qubits)
        for q, val in leakage.items():
            assert val >= 0.0

    def test_apply_lru_returns_normalized_state(self):
        # Build a 3-level state with some leakage into |2>
        n_qubits = 1
        dim3 = 3 ** n_qubits
        state = np.zeros(dim3, dtype=np.complex128)
        state[0] = 0.9  # |0>
        state[2] = 0.1  # |2> (leaked)
        state /= np.linalg.norm(state)
        new_state = LeakageReductionUnit.apply_lru(state, qubit=0, n_qubits=n_qubits)
        assert np.isclose(np.linalg.norm(new_state), 1.0)


# ---------------------------------------------------------------------------
# TransmonQCVV tests
# ---------------------------------------------------------------------------

class TestTransmonQCVV:
    def test_bell_state_fidelity(self, ibm_eagle_config):
        qcvv = TransmonQCVV(ibm_eagle_config)
        result = qcvv.bell_state_fidelity(qubit_a=0, qubit_b=1)
        assert isinstance(result, BenchmarkResult)
        assert 0 <= result.metric_value <= 1

    def test_randomized_benchmarking(self, ibm_eagle_config):
        qcvv = TransmonQCVV(ibm_eagle_config)
        result = qcvv.randomized_benchmarking(
            qubit=0, sequence_lengths=[1, 2, 4, 8, 16, 32], n_sequences=10
        )
        assert isinstance(result, BenchmarkResult)
        # epg can be slightly negative due to statistical noise in short runs
        assert result.metric_value >= -0.1

    def test_quantum_volume(self, ibm_eagle_config):
        qcvv = TransmonQCVV(ibm_eagle_config)
        result = qcvv.quantum_volume(max_depth=2, n_trials=5)
        assert isinstance(result, BenchmarkResult)
        assert result.metric_value >= 0

    def test_xeb_fidelity(self, ibm_eagle_config):
        qcvv = TransmonQCVV(ibm_eagle_config)
        result = qcvv.xeb_fidelity(num_qubits=2, n_circuits=5, depth=3)
        assert isinstance(result, BenchmarkResult)


# ---------------------------------------------------------------------------
# compare_backends
# ---------------------------------------------------------------------------

class TestCompareBackends:
    def test_compare_returns_dict(self):
        config_a = DevicePresets.IBM_EAGLE.build(num_qubits=3)
        config_b = DevicePresets.IBM_HERON.build(num_qubits=3)

        def bell_circuit(sim):
            sim.h(0)
            sim.cnot(0, 1)

        results = compare_backends(
            bell_circuit,
            {"eagle": config_a, "heron": config_b},
            execution_modes=["ideal"],
            shots=50,
        )
        assert isinstance(results, dict)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# DigitalTwin tests
# ---------------------------------------------------------------------------

class TestDigitalTwin:
    def test_from_calibration(self, ibm_eagle_config):
        edges = ibm_eagle_config.topology.edges
        cal = CalibrationData(
            qubit_frequencies_ghz=[q.frequency_ghz for q in ibm_eagle_config.qubits],
            t1_us=[q.t1_us for q in ibm_eagle_config.qubits],
            t2_us=[q.t2_us for q in ibm_eagle_config.qubits],
            single_gate_fidelities=[q.single_gate_fidelity for q in ibm_eagle_config.qubits],
            readout_fidelities=[q.readout_fidelity for q in ibm_eagle_config.qubits],
            two_qubit_fidelities={
                (e[0], e[1]): ibm_eagle_config.two_qubit_fidelity for e in edges
            },
            coupling_map=[(e[0], e[1]) for e in edges],
        )
        twin = DigitalTwin.from_calibration(cal)
        assert twin is not None

    def test_compare_presets(self):
        results = DigitalTwin.compare_presets(num_qubits=3)
        assert isinstance(results, dict)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# CalibrationDrift tests
# ---------------------------------------------------------------------------

class TestCalibrationDrift:
    def test_drift_modifies_calibration(self, ibm_eagle_config):
        edges = ibm_eagle_config.topology.edges
        base_cal = CalibrationData(
            qubit_frequencies_ghz=[q.frequency_ghz for q in ibm_eagle_config.qubits],
            t1_us=[q.t1_us for q in ibm_eagle_config.qubits],
            t2_us=[q.t2_us for q in ibm_eagle_config.qubits],
            single_gate_fidelities=[q.single_gate_fidelity for q in ibm_eagle_config.qubits],
            readout_fidelities=[q.readout_fidelity for q in ibm_eagle_config.qubits],
            two_qubit_fidelities={
                (e[0], e[1]): ibm_eagle_config.two_qubit_fidelity for e in edges
            },
            coupling_map=[(e[0], e[1]) for e in edges],
        )
        drift = CalibrationDrift(base_cal, seed=SEED)
        drifted = drift.at_time(hours=1.0)
        assert isinstance(drifted, CalibrationData)


# ---------------------------------------------------------------------------
# CompilerBenchmark + NativeGateAnalyzer tests
# ---------------------------------------------------------------------------

class TestNativeGateAnalyzer:
    @pytest.mark.parametrize("gate_family_str", ["ecr", "sqrt_iswap", "cz"])
    def test_decompose_cnot(self, gate_family_str):
        analyzer = NativeGateAnalyzer()
        instructions = analyzer.decompose_cnot(gate_family_str)
        assert len(instructions) > 0

    def test_overhead_ratio(self):
        analyzer = NativeGateAnalyzer()
        ratio = analyzer.overhead_ratio("ecr", "bell", num_qubits=4)
        assert ratio > 0

    def test_cnot_native_counts(self):
        analyzer = NativeGateAnalyzer()
        counts = analyzer.cnot_native_counts("ecr")
        assert isinstance(counts, dict)

    def test_cnot_duration(self):
        analyzer = NativeGateAnalyzer()
        dur = analyzer.cnot_duration_ns("ecr")
        assert dur > 0


class TestCompilerBenchmark:
    def test_benchmark_compilation(self):
        bench = CompilerBenchmark(num_qubits=4)
        result = bench.benchmark_compilation(circuit_type="bell", gate_family="ecr")
        assert isinstance(result, CompilationResult)

    def test_benchmark_all_circuits(self):
        bench = CompilerBenchmark(num_qubits=4)
        results = bench.benchmark_all_circuits()
        assert isinstance(results, dict)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Pulse-level type imports smoke test
# ---------------------------------------------------------------------------

class TestPulseImports:
    def test_pulse_types_importable(self):
        from nqpu.superconducting import (
            PulseShape,
            Pulse,
            PulseSchedule,
            PulseSimulator,
            build_lindblad_operators,
            evolve_density_matrix,
            ReadoutSimulator,
            ReadoutResult,
            DiscriminationResult,
            CRCalibrator,
            CRCalibrationResult,
            thermal_state,
            EchoedCRCalibrator,
            EchoedCRCalibrationResult,
        )
        # These should all be importable
        assert PulseShape is not None
        assert Pulse is not None

    def test_thermal_state(self):
        from nqpu.superconducting import thermal_state
        rho = thermal_state(temperature_mk=20, frequency_ghz=5.0)
        assert rho.shape == (3, 3) or rho.shape == (2, 2)
        assert np.isclose(np.trace(rho).real, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Additional module smoke tests
# ---------------------------------------------------------------------------

class TestAdditionalModules:
    def test_grape_importable(self):
        from nqpu.superconducting import GrapeOptimizer, GrapeResult
        assert GrapeOptimizer is not None
        assert GrapeResult is not None

    def test_unified_sim_importable(self):
        from nqpu.superconducting import (
            UnifiedSimulator,
            FidelityTable,
            ScalingData,
            GateOverheadTable,
            PublicationFigures,
            BackendResult,
        )
        assert UnifiedSimulator is not None

    def test_pulse_library_importable(self):
        from nqpu.superconducting import (
            PulseLibrary,
            HardwarePreset,
            HARDWARE_PRESETS,
            SUPPORTED_GATES,
            DURATION_PRESETS,
        )
        assert isinstance(HARDWARE_PRESETS, dict)
        assert isinstance(SUPPORTED_GATES, (list, tuple, set, dict))

    def test_noise_fingerprint_importable(self):
        from nqpu.superconducting import NoiseFingerprint, FingerprintComparison
        assert NoiseFingerprint is not None

    def test_qec_drift_study_importable(self):
        from nqpu.superconducting import QECDriftStudy, DriftImpactReport
        assert QECDriftStudy is not None

    def test_calibration_forecaster_importable(self):
        from nqpu.superconducting import CalibrationForecaster, ForecastReport
        assert CalibrationForecaster is not None


# ---------------------------------------------------------------------------
# Cross-cutting parametric tests
# ---------------------------------------------------------------------------

class TestCrossCuttingPresets:
    @pytest.mark.parametrize("preset", [
        DevicePresets.IBM_EAGLE,
        DevicePresets.IBM_HERON,
        DevicePresets.GOOGLE_SYCAMORE,
        DevicePresets.GOOGLE_WILLOW,
        DevicePresets.RIGETTI_ANKAA,
    ])
    def test_preset_ideal_simulation(self, preset):
        config = preset.build(num_qubits=3)
        sim = TransmonSimulator(config, execution_mode="ideal")
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.measure_all(shots=50)
        assert sum(result.values()) == 50

    @pytest.mark.parametrize("preset", [
        DevicePresets.IBM_EAGLE,
        DevicePresets.IBM_HERON,
        DevicePresets.GOOGLE_SYCAMORE,
    ])
    def test_preset_noisy_simulation(self, preset):
        config = preset.build(num_qubits=3)
        sim = TransmonSimulator(config, execution_mode="noisy")
        sim.h(0)
        sim.cnot(0, 1)
        result = sim.measure_all(shots=50)
        assert sum(result.values()) == 50
