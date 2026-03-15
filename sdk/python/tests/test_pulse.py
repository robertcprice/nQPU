"""Comprehensive tests for pulse-level transmon simulation with DRAG correction.

Tests cover:
- Pulse envelope shapes (Gaussian, DRAG, Cosine, Flat, Gaussian-Square)
- Hamiltonian construction and Hermiticity
- RK4 solver norm preservation and eigenstate stability
- DRAG vs Gaussian leakage suppression
- PulseSchedule construction and sequential scheduling
- Gate fidelity and computational population extraction
- Boundary conditions and edge cases
"""

import math

import numpy as np
import pytest

from nqpu.superconducting.pulse import (
    ChannelType,
    Pulse,
    PulseSchedule,
    PulseShape,
    PulseSimulator,
    ScheduledPulse,
    TransmonHamiltonian,
    evolve_state,
)
from nqpu.superconducting.qubit import TransmonQubit
from nqpu.superconducting.chip import (
    ChipConfig,
    ChipTopology,
    NativeGateFamily,
    DevicePresets,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def single_qubit():
    """A transmon qubit with default parameters."""
    return TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)


@pytest.fixture
def single_qubit_config():
    """A 1-qubit chip config for pulse simulation."""
    topo = ChipTopology.fully_connected(1, coupling=0.0)
    qubit = TransmonQubit(
        frequency_ghz=5.0,
        anharmonicity_mhz=-330.0,
        gate_time_ns=25.0,
    )
    return ChipConfig(
        topology=topo,
        qubits=[qubit],
        native_2q_gate=NativeGateFamily.ECR,
    )


@pytest.fixture
def two_qubit_config():
    """A 2-qubit chip config for pulse simulation."""
    topo = ChipTopology.fully_connected(2, coupling=3.0)
    q0 = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0, gate_time_ns=25.0)
    q1 = TransmonQubit(frequency_ghz=5.1, anharmonicity_mhz=-320.0, gate_time_ns=25.0)
    return ChipConfig(
        topology=topo,
        qubits=[q0, q1],
        native_2q_gate=NativeGateFamily.ECR,
        two_qubit_gate_time_ns=200.0,
    )


@pytest.fixture
def leaky_config():
    """Config with small anharmonicity and fast gate for visible leakage."""
    topo = ChipTopology.fully_connected(1, coupling=0.0)
    qubit = TransmonQubit(
        frequency_ghz=5.0,
        anharmonicity_mhz=-150.0,
        gate_time_ns=8.0,
    )
    return ChipConfig(
        topology=topo,
        qubits=[qubit],
        native_2q_gate=NativeGateFamily.ECR,
    )


# ======================================================================
# Pulse envelope tests
# ======================================================================


class TestPulseEnvelopes:
    """Tests for pulse envelope shape evaluation."""

    def test_flat_envelope_constant(self):
        """Flat pulse returns constant amplitude at all times."""
        p = Pulse(amplitude=0.1, duration_ns=10.0, shape=PulseShape.FLAT)
        for t in [0.0, 3.0, 5.0, 9.9]:
            env = p.envelope(t)
            assert env.real == pytest.approx(0.1, abs=1e-12)
            assert env.imag == pytest.approx(0.0, abs=1e-12)

    def test_gaussian_peaks_at_center(self):
        """Gaussian envelope has max amplitude at center of pulse."""
        p = Pulse(amplitude=0.1, duration_ns=20.0, shape=PulseShape.GAUSSIAN)
        center = p.envelope(10.0)
        edge = p.envelope(0.0)
        assert center.real == pytest.approx(0.1, abs=1e-12)
        assert center.real > edge.real
        assert center.imag == pytest.approx(0.0, abs=1e-12)

    def test_cosine_zero_at_boundaries(self):
        """Cosine envelope is zero at t=0 and peaks at center."""
        p = Pulse(amplitude=0.1, duration_ns=20.0, shape=PulseShape.COSINE)
        assert p.envelope(0.0).real == pytest.approx(0.0, abs=1e-12)
        assert p.envelope(10.0).real == pytest.approx(0.1, abs=1e-6)

    def test_drag_quadrature_zero_at_center(self):
        """DRAG quadrature (derivative of Gaussian) is zero at pulse center."""
        p = Pulse(
            amplitude=0.1,
            duration_ns=20.0,
            shape=PulseShape.DRAG,
            drag_coefficient=1.0,
            sigma_ns=5.0,
        )
        env_center = p.envelope(10.0)
        assert env_center.imag == pytest.approx(0.0, abs=1e-12)

    def test_drag_quadrature_nonzero_away_from_center(self):
        """DRAG quadrature is nonzero away from the center."""
        p = Pulse(
            amplitude=0.1,
            duration_ns=20.0,
            shape=PulseShape.DRAG,
            drag_coefficient=1.0,
            sigma_ns=5.0,
        )
        env_left = p.envelope(5.0)
        assert abs(env_left.imag) > 1e-6

    def test_gaussian_square_flat_region(self):
        """Gaussian-square has a flat region in the middle."""
        p = Pulse(
            amplitude=0.1,
            duration_ns=30.0,
            shape=PulseShape.GAUSSIAN_SQUARE,
            sigma_ns=5.0,
            flat_duration_ns=10.0,
        )
        # Middle of flat region should be at full amplitude
        env_flat = p.envelope(15.0)
        assert env_flat.real == pytest.approx(0.1, abs=1e-12)

    @pytest.mark.parametrize("shape", list(PulseShape))
    def test_envelope_returns_complex(self, shape):
        """Every pulse shape returns a complex value."""
        kwargs = {"amplitude": 0.05, "duration_ns": 20.0, "shape": shape}
        if shape == PulseShape.DRAG:
            kwargs["drag_coefficient"] = 1.0
            kwargs["sigma_ns"] = 5.0
        if shape == PulseShape.GAUSSIAN_SQUARE:
            kwargs["flat_duration_ns"] = 5.0
        p = Pulse(**kwargs)
        env = p.envelope(10.0)
        assert isinstance(env, complex)

    def test_sigma_defaults_to_duration_over_4(self):
        """When sigma_ns is 0, effective sigma = duration / 4."""
        p = Pulse(amplitude=0.1, duration_ns=20.0, shape=PulseShape.GAUSSIAN)
        assert p.sigma == pytest.approx(5.0, abs=1e-12)


# ======================================================================
# Envelope array tests
# ======================================================================


class TestEnvelopeArray:
    """Tests for vectorised envelope sampling."""

    def test_envelope_array_length(self):
        """Envelope array produces expected number of samples."""
        p = Pulse(amplitude=0.05, duration_ns=25.0, shape=PulseShape.GAUSSIAN)
        times, vals = p.envelope_array(dt_ns=1.0)
        assert len(times) == 25
        assert len(vals) == 25

    def test_envelope_array_dtype(self):
        """Envelope array has complex128 dtype."""
        p = Pulse(amplitude=0.05, duration_ns=25.0, shape=PulseShape.GAUSSIAN)
        _, vals = p.envelope_array(dt_ns=1.0)
        assert vals.dtype == np.complex128

    def test_envelope_array_peak_near_center(self):
        """Gaussian peak appears near the center sample."""
        p = Pulse(amplitude=0.05, duration_ns=25.0, shape=PulseShape.GAUSSIAN)
        _, vals = p.envelope_array(dt_ns=1.0)
        peak_idx = int(np.argmax(np.abs(vals)))
        assert abs(peak_idx - 12) <= 1


# ======================================================================
# Hamiltonian tests
# ======================================================================


class TestTransmonHamiltonian:
    """Tests for Hamiltonian construction."""

    def test_single_qubit_dimension(self, single_qubit):
        """Single-qubit Hamiltonian is 3x3."""
        ham = TransmonHamiltonian([single_qubit])
        assert ham.dim == 3
        H0 = ham.static_hamiltonian(5.0)
        assert H0.shape == (3, 3)

    def test_single_qubit_hermiticity(self, single_qubit):
        """Static Hamiltonian must be Hermitian."""
        ham = TransmonHamiltonian([single_qubit])
        H0 = ham.static_hamiltonian(5.0)
        assert np.allclose(H0, H0.conj().T)

    def test_two_qubit_dimension(self):
        """Two-qubit Hamiltonian is 9x9."""
        q0 = TransmonQubit(frequency_ghz=5.0, anharmonicity_mhz=-330.0)
        q1 = TransmonQubit(frequency_ghz=5.1, anharmonicity_mhz=-320.0)
        ham = TransmonHamiltonian([q0, q1], coupling_mhz=3.0)
        assert ham.dim == 9
        H0 = ham.static_hamiltonian([5.0, 5.1])
        assert H0.shape == (9, 9)
        assert np.allclose(H0, H0.conj().T)

    def test_rejects_three_qubits(self):
        """Hamiltonian only supports 1 or 2 qubits."""
        q = TransmonQubit()
        with pytest.raises(ValueError, match="1 or 2 qubits"):
            TransmonHamiltonian([q, q, q])

    def test_drive_operators_shape(self, single_qubit):
        """Drive operators have correct shape for 1Q system."""
        ham = TransmonHamiltonian([single_qubit])
        dx, dy = ham.drive_operators()
        assert dx.shape == (3, 3)
        assert dy.shape == (3, 3)


# ======================================================================
# RK4 solver tests
# ======================================================================


class TestRK4Solver:
    """Tests for the RK4 time-evolution solver."""

    def test_norm_preservation(self, single_qubit):
        """RK4 evolution preserves state norm."""
        ham = TransmonHamiltonian([single_qubit])
        H0 = ham.static_hamiltonian(5.0)
        psi0 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
        psi_f = evolve_state(psi0, lambda _t: H0, total_time=100.0, dt=0.1)
        norm = np.linalg.norm(psi_f)
        assert norm == pytest.approx(1.0, abs=1e-10)

    def test_eigenstate_stability(self, single_qubit):
        """Energy eigenstate |1> remains stable under free evolution."""
        ham = TransmonHamiltonian([single_qubit])
        H0 = ham.static_hamiltonian(5.0)
        psi0 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
        psi_f = evolve_state(psi0, lambda _t: H0, total_time=100.0, dt=0.1)
        pop1 = abs(psi_f[1]) ** 2
        assert pop1 > 0.99

    def test_ground_state_stays_ground(self, single_qubit):
        """Ground state |0> remains in |0> under zero drive."""
        ham = TransmonHamiltonian([single_qubit])
        H0 = ham.static_hamiltonian(5.0)
        psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        psi_f = evolve_state(psi0, lambda _t: H0, total_time=50.0, dt=0.1)
        pop0 = abs(psi_f[0]) ** 2
        assert pop0 > 0.99


# ======================================================================
# Pulse simulator tests
# ======================================================================


class TestPulseSimulator:
    """Tests for PulseSimulator functionality."""

    def test_drag_pulse_generation(self, single_qubit_config):
        """DRAG pulse has correct shape and nonzero drag coefficient."""
        psim = PulseSimulator(single_qubit_config)
        p = psim.drag_pulse(qubit=0, angle=math.pi, axis="x")
        assert p.shape == PulseShape.DRAG
        assert p.drag_coefficient != 0.0
        assert p.amplitude > 0.0
        assert p.frequency_ghz == pytest.approx(5.0, abs=0.01)

    def test_gaussian_pulse_no_drag(self, single_qubit_config):
        """Gaussian pulse has zero drag coefficient."""
        psim = PulseSimulator(single_qubit_config)
        p = psim.gaussian_pulse(qubit=0, angle=math.pi, axis="x")
        assert p.shape == PulseShape.GAUSSIAN
        assert p.drag_coefficient == 0.0

    def test_y_axis_pulse_phase(self, single_qubit_config):
        """Y-axis rotation has pi/2 phase offset."""
        psim = PulseSimulator(single_qubit_config)
        px = psim.drag_pulse(qubit=0, angle=math.pi, axis="x")
        py = psim.drag_pulse(qubit=0, angle=math.pi, axis="y")
        assert py.phase == pytest.approx(math.pi / 2.0, abs=1e-12)
        assert px.phase == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.slow
    def test_drag_reduces_leakage(self, leaky_config):
        """DRAG pulse has less leakage to |2> than a Gaussian pulse."""
        psim = PulseSimulator(leaky_config, dt_ns=0.01)
        gauss_p = psim.gaussian_pulse(qubit=0, angle=math.pi, axis="x")
        psi_gauss = psim.simulate_pulse(gauss_p, qubit=0)
        leak_gauss = abs(psi_gauss[2]) ** 2

        drag_p = psim.drag_pulse(qubit=0, angle=math.pi, axis="x")
        psi_drag = psim.simulate_pulse(drag_p, qubit=0)
        leak_drag = abs(psi_drag[2]) ** 2

        # Gaussian leakage should be significant
        assert leak_gauss > 0.01
        # DRAG should reduce leakage
        assert leak_drag < leak_gauss

    def test_pi_pulse_population_transfer(self, single_qubit_config):
        """A pi pulse transfers population from |0> to |1>."""
        psim = PulseSimulator(single_qubit_config, dt_ns=0.05)
        drag_p = psim.drag_pulse(qubit=0, angle=math.pi, axis="x")
        psi = psim.simulate_pulse(drag_p, qubit=0)
        pop1 = abs(psi[1]) ** 2
        # Should have high population in |1>
        assert pop1 > 0.85

    def test_simulate_with_custom_initial_state(self, single_qubit_config):
        """Simulation accepts a custom initial state."""
        psim = PulseSimulator(single_qubit_config, dt_ns=0.1)
        p = Pulse(
            amplitude=0.0,
            duration_ns=10.0,
            shape=PulseShape.FLAT,
            frequency_ghz=5.0,
        )
        psi0 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
        psi_f = psim.simulate_pulse(p, qubit=0, initial_state=psi0)
        # Zero-amplitude pulse should not change populations much
        pop1 = abs(psi_f[1]) ** 2
        assert pop1 > 0.95

    def test_computational_populations_single_qubit(self):
        """Computational populations sum to ~1 for a single-qubit state."""
        psi = np.array([0.6, 0.8, 0.0], dtype=np.complex128)
        psi /= np.linalg.norm(psi)
        pops = PulseSimulator.computational_populations(psi)
        assert "|0>" in pops
        assert "|1>" in pops
        assert "|2> (leakage)" in pops
        total = sum(pops.values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_computational_populations_two_qubit(self):
        """Computational populations for 9-level state include leakage."""
        psi = np.zeros(9, dtype=np.complex128)
        psi[0] = 1.0 / math.sqrt(2)  # |00>
        psi[4] = 1.0 / math.sqrt(2)  # |11>
        pops = PulseSimulator.computational_populations(psi)
        assert "|00>" in pops
        assert "|11>" in pops
        assert "leakage" in pops
        assert pops["|00>"] == pytest.approx(0.5, abs=1e-10)
        assert pops["|11>"] == pytest.approx(0.5, abs=1e-10)
        assert pops["leakage"] == pytest.approx(0.0, abs=1e-10)

    def test_gate_fidelity_perfect(self):
        """Perfect state should give fidelity 1.0."""
        psi = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
        target = np.array([0.0, 1.0], dtype=np.complex128)
        fid = PulseSimulator.gate_fidelity(psi, target)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_gate_fidelity_orthogonal(self):
        """Orthogonal state gives fidelity 0.0."""
        psi = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        target = np.array([0.0, 1.0], dtype=np.complex128)
        fid = PulseSimulator.gate_fidelity(psi, target)
        assert fid == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# Pulse schedule tests
# ======================================================================


class TestPulseSchedule:
    """Tests for PulseSchedule construction and timing."""

    def test_empty_schedule_duration(self):
        """Empty schedule has zero duration."""
        sched = PulseSchedule()
        assert sched.duration_ns == 0.0

    def test_sequential_scheduling(self):
        """Sequential adds auto-calculate start times."""
        sched = PulseSchedule()
        p1 = Pulse(duration_ns=10.0, shape=PulseShape.FLAT)
        p2 = Pulse(duration_ns=15.0, shape=PulseShape.FLAT)
        sched.add(p1, ChannelType.DRIVE, qubit=0)
        sched.add(p2, ChannelType.DRIVE, qubit=0)
        assert sched.duration_ns == pytest.approx(25.0, abs=1e-10)

    def test_explicit_start_time(self):
        """Pulses can be placed at explicit start times."""
        sched = PulseSchedule()
        p = Pulse(duration_ns=10.0, shape=PulseShape.FLAT)
        sched.add(p, ChannelType.DRIVE, qubit=0, start_ns=5.0)
        assert sched.entries[0].start_ns == 5.0
        assert sched.entries[0].end_ns == 15.0

    def test_sorted_entries(self):
        """Sorted entries returns pulses in chronological order."""
        sched = PulseSchedule()
        p1 = Pulse(duration_ns=10.0, shape=PulseShape.FLAT)
        p2 = Pulse(duration_ns=10.0, shape=PulseShape.FLAT)
        sched.add(p1, ChannelType.DRIVE, qubit=0, start_ns=20.0)
        sched.add(p2, ChannelType.DRIVE, qubit=0, start_ns=5.0)
        sorted_e = sched.sorted_entries()
        assert sorted_e[0].start_ns < sorted_e[1].start_ns

    def test_schedule_simulation_two_half_pi(self, single_qubit_config):
        """Two pi/2 pulses approximate a pi rotation."""
        psim = PulseSimulator(single_qubit_config, dt_ns=0.05)
        half_pi = psim.drag_pulse(qubit=0, angle=math.pi / 2.0, axis="x")
        sched = PulseSchedule()
        sched.add(half_pi, ChannelType.DRIVE, qubit=0)
        sched.add(half_pi, ChannelType.DRIVE, qubit=0)
        psi = psim.simulate_schedule(sched, qubits=[0])
        pops = PulseSimulator.computational_populations(psi)
        assert pops["|1>"] > 0.80

    def test_schedule_simulation_rejects_empty(self, single_qubit_config):
        """Simulating an empty schedule raises ValueError."""
        psim = PulseSimulator(single_qubit_config)
        sched = PulseSchedule()
        with pytest.raises(ValueError, match="no pulses"):
            psim.simulate_schedule(sched)

    def test_schedule_simulation_rejects_too_many_qubits(self, single_qubit_config):
        """More than 2 qubits in a schedule raises ValueError."""
        # Build 3-qubit config
        topo = ChipTopology.fully_connected(3, coupling=3.0)
        qubits = [TransmonQubit() for _ in range(3)]
        config = ChipConfig(topology=topo, qubits=qubits, native_2q_gate=NativeGateFamily.ECR)
        psim = PulseSimulator(config)

        sched = PulseSchedule()
        p = Pulse(duration_ns=10.0, shape=PulseShape.FLAT)
        sched.add(p, ChannelType.DRIVE, qubit=0)
        sched.add(p, ChannelType.DRIVE, qubit=1)
        sched.add(p, ChannelType.DRIVE, qubit=2)

        with pytest.raises(ValueError, match="at most 2 qubits"):
            psim.simulate_schedule(sched)


# ======================================================================
# Cross-resonance tests
# ======================================================================


class TestCrossResonancePulse:
    """Tests for CR pulse generation."""

    def test_cr_schedule_has_four_segments(self, two_qubit_config):
        """Echoed CR schedule has 4 drive pulses on the control channel."""
        psim = PulseSimulator(two_qubit_config)
        sched = psim.cr_pulse(control=0, target=1)
        assert len(sched.entries) == 4
        assert sched.duration_ns > 0

    def test_cr_schedule_total_duration(self, two_qubit_config):
        """CR schedule total duration is positive and reasonable."""
        psim = PulseSimulator(two_qubit_config)
        sched = psim.cr_pulse(control=0, target=1)
        # 4 segments, each with some duration
        assert sched.duration_ns > 50.0
