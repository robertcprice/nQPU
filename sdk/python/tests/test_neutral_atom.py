"""Comprehensive tests for the neutral-atom Rydberg blockade backend.

Tests cover all six modules of the neutral_atom package:
  - physics.py: Atom species physical parameters and derived quantities
  - array.py: Optical tweezer array geometries, loading, and zone management
  - gates.py: Native Rydberg gate set and compilation
  - noise.py: Physics-based noise model (depolarising, dephasing, atom loss)
  - devices.py: Device presets for QuEra, Atom Computing, Pasqal
  - simulator.py: End-to-end quantum circuit simulation (ideal, noisy, pulse)

Cross-backend comparison tests verify consistency against trapped-ion and
superconducting transmon simulators.
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


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rb87() -> AtomSpecies:
    """Rubidium-87 species singleton."""
    return AtomSpecies.RB87


@pytest.fixture
def sim_ideal_2q() -> NeutralAtomSimulator:
    """2-qubit ideal neutral-atom simulator."""
    return NeutralAtomSimulator(ArrayConfig(n_atoms=2))


@pytest.fixture
def sim_ideal_3q() -> NeutralAtomSimulator:
    """3-qubit ideal neutral-atom simulator."""
    return NeutralAtomSimulator(ArrayConfig(n_atoms=3))


@pytest.fixture
def sim_noisy_2q() -> NeutralAtomSimulator:
    """2-qubit noisy neutral-atom simulator."""
    return NeutralAtomSimulator(
        ArrayConfig(n_atoms=2), execution_mode="noisy"
    )


@pytest.fixture
def sim_noisy_3q() -> NeutralAtomSimulator:
    """3-qubit noisy neutral-atom simulator."""
    return NeutralAtomSimulator(
        ArrayConfig(n_atoms=3), execution_mode="noisy"
    )


@pytest.fixture
def noise_model_rb87() -> NeutralAtomNoiseModel:
    """Default noise model for Rb87."""
    return NeutralAtomNoiseModel(species=AtomSpecies.RB87)


# ======================================================================
# Physics tests
# ======================================================================


class TestAtomSpeciesPhysics:
    """Tests for Rydberg atom physics parameters and derived quantities."""

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_positive_mass(self, species: AtomSpecies):
        """Every species must have a positive atomic mass."""
        assert species.mass_amu > 0
        assert species.mass_kg > 0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_positive_c6_coefficient(self, species: AtomSpecies):
        """C6 dispersion coefficient must be positive (attractive VdW)."""
        assert species.c6_hz_um6 > 0
        assert species.c6_joule_m6 > 0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_blockade_radius_positive(self, species: AtomSpecies):
        """Blockade radius must be positive at a typical Rabi frequency."""
        rb = species.blockade_radius_um(rabi_freq_mhz=1.0)
        assert rb > 0
        # Typical values: 5-15 um for MHz-scale Rabi frequencies
        assert rb < 50.0, "Blockade radius unreasonably large"

    def test_blockade_radius_decreases_with_rabi(self, rb87: AtomSpecies):
        """Higher Rabi frequency yields a smaller blockade radius.

        R_b = (C6 / Omega)^(1/6), so increasing Omega shrinks R_b.
        """
        r_low = rb87.blockade_radius_um(rabi_freq_mhz=0.5)
        r_high = rb87.blockade_radius_um(rabi_freq_mhz=5.0)
        assert r_low > r_high

    def test_blockade_radius_rejects_non_positive(self, rb87: AtomSpecies):
        """Blockade radius raises ValueError for non-positive Rabi frequency."""
        with pytest.raises(ValueError, match="positive"):
            rb87.blockade_radius_um(rabi_freq_mhz=0.0)
        with pytest.raises(ValueError, match="positive"):
            rb87.blockade_radius_um(rabi_freq_mhz=-1.0)

    def test_vdw_interaction_r_minus_6(self, rb87: AtomSpecies):
        """VdW interaction scales as r^{-6}: doubling distance reduces it by 64x."""
        v_near = rb87.vdw_interaction_hz(distance_um=4.0)
        v_far = rb87.vdw_interaction_hz(distance_um=8.0)
        ratio = v_near / v_far
        np.testing.assert_allclose(ratio, 2.0**6, rtol=1e-10)

    def test_vdw_interaction_rejects_non_positive(self, rb87: AtomSpecies):
        """VdW interaction raises ValueError for non-positive distance."""
        with pytest.raises(ValueError, match="positive"):
            rb87.vdw_interaction_hz(distance_um=0.0)

    def test_blockade_fidelity_near_unity_at_typical_spacing(
        self, rb87: AtomSpecies
    ):
        """Blockade fidelity should be near 1 at typical operational spacings.

        At 4 um spacing with 1.5 MHz Rabi freq, the ratio Omega/V is very
        small, so the blockade error (Omega/V)^2 is negligible.
        """
        fid = rb87.blockade_fidelity(distance_um=4.0, rabi_freq_mhz=1.5)
        assert fid > 0.99

    def test_blockade_fidelity_degrades_at_large_distance(
        self, rb87: AtomSpecies
    ):
        """Blockade fidelity should degrade when atoms are far apart.

        At large distance, V -> 0 and blockade breaks down.
        """
        fid_close = rb87.blockade_fidelity(distance_um=4.0, rabi_freq_mhz=1.5)
        fid_far = rb87.blockade_fidelity(distance_um=20.0, rabi_freq_mhz=1.5)
        assert fid_close > fid_far

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_thermal_dephasing_rate_positive(self, species: AtomSpecies):
        """Thermal dephasing rate must be positive (atoms have finite temperature)."""
        rate = species.thermal_dephasing_rate_hz(rabi_freq_mhz=1.0)
        assert rate > 0

    @pytest.mark.parametrize("species", ALL_SPECIES)
    def test_coherence_times_positive(self, species: AtomSpecies):
        """T1 and T2 must be positive for all species."""
        assert species.t1_s > 0
        assert species.t2_s > 0
        # T2 <= 2*T1 is a physical constraint
        assert species.t2_s <= 2 * species.t1_s + 1e-6


# ======================================================================
# Array tests
# ======================================================================


class TestAtomArray:
    """Tests for optical tweezer array geometry and management."""

    @pytest.mark.parametrize(
        "geometry,n_sites",
        [
            (ArrayGeometry.LINEAR, 10),
            (ArrayGeometry.RECTANGULAR, 16),
            (ArrayGeometry.TRIANGULAR, 12),
            (ArrayGeometry.HONEYCOMB, 8),
            (ArrayGeometry.KAGOME, 9),
        ],
    )
    def test_geometry_creates_correct_sites(
        self, geometry: ArrayGeometry, n_sites: int
    ):
        """Each geometry type creates the requested number of site positions."""
        arr = AtomArray(n_sites=n_sites, geometry=geometry)
        assert arr.positions.shape == (n_sites, 2)

    def test_custom_geometry(self):
        """Custom geometry uses the provided positions directly."""
        custom = np.array([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]])
        arr = AtomArray(
            n_sites=3,
            geometry=ArrayGeometry.CUSTOM,
            custom_positions=custom,
        )
        np.testing.assert_allclose(arr.positions, custom)

    def test_custom_geometry_requires_positions(self):
        """CUSTOM geometry without positions raises ValueError."""
        with pytest.raises(ValueError, match="custom_positions"):
            AtomArray(n_sites=3, geometry=ArrayGeometry.CUSTOM)

    def test_distance_matrix_symmetric(self):
        """Distance matrix must be symmetric: D[i,j] == D[j,i]."""
        arr = AtomArray(n_sites=9, geometry=ArrayGeometry.RECTANGULAR)
        dm = arr.distance_matrix()
        np.testing.assert_allclose(dm, dm.T, atol=1e-12)

    def test_distance_matrix_diagonal_zero(self):
        """Distance matrix diagonal must be exactly zero."""
        arr = AtomArray(n_sites=9, geometry=ArrayGeometry.RECTANGULAR)
        dm = arr.distance_matrix()
        np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-12)

    def test_distance_between_sites(self):
        """Distance between two sites matches the expected spacing."""
        arr = AtomArray(
            n_sites=4, geometry=ArrayGeometry.LINEAR, spacing_um=5.0
        )
        # Adjacent sites in a linear chain should be spacing apart
        d01 = arr.distance(0, 1)
        np.testing.assert_allclose(d01, 5.0, atol=1e-10)

    def test_neighbours_rectangular_interior(self):
        """Interior site in a rectangular grid has 4 nearest neighbours.

        Default cutoff is 1.5*spacing = 6.0 um, which includes diagonal
        neighbours at 4*sqrt(2) = 5.66 um.  To get only the 4 axial NN,
        use a tighter cutoff of 1.1*spacing.
        """
        arr = AtomArray(
            n_sites=25, geometry=ArrayGeometry.RECTANGULAR, rows=5, cols=5
        )
        # Centre site (index 12 in a 5x5 grid)
        # Use tight cutoff to exclude diagonal neighbours
        nbrs = arr.neighbours(12, max_distance_um=1.1 * arr.spacing_um)
        assert len(nbrs) == 4

    def test_neighbours_rectangular_default_cutoff_includes_diagonals(self):
        """Default cutoff (1.5*spacing) includes diagonal neighbours in a grid.

        At 1.5*spacing = 6.0 um, diagonals at sqrt(2)*spacing = 5.66 um
        are within range, giving 8 neighbours for an interior site.
        """
        arr = AtomArray(
            n_sites=25, geometry=ArrayGeometry.RECTANGULAR, rows=5, cols=5
        )
        nbrs = arr.neighbours(12)
        assert len(nbrs) == 8

    def test_stochastic_load_partial(self):
        """Stochastic loading fills a fraction of sites (not all, not none)."""
        arr = AtomArray(n_sites=100, geometry=ArrayGeometry.RECTANGULAR)
        rng = np.random.default_rng(42)
        n_loaded = arr.stochastic_load(rng=rng)
        # With p=0.5, expecting ~50 out of 100
        assert 0 < n_loaded < 100
        assert n_loaded == arr.n_atoms

    def test_deterministic_load_fills_all(self):
        """Deterministic loading fills every site."""
        arr = AtomArray(n_sites=16, geometry=ArrayGeometry.RECTANGULAR)
        n_loaded = arr.deterministic_load()
        assert n_loaded == 16
        assert arr.n_atoms == 16

    def test_rearrange_fills_target_sites(self):
        """Rearrangement moves atoms from random sites to target sites."""
        arr = AtomArray(n_sites=20, geometry=ArrayGeometry.LINEAR)
        rng = np.random.default_rng(123)
        arr.stochastic_load(rng=rng)
        n_before = arr.n_atoms

        # Target: fill the first 5 sites
        target = list(range(5))
        filled = arr.rearrange(target_sites=target)
        # Should fill up to min(available atoms, target count)
        assert filled <= len(target)
        assert filled <= n_before

    def test_zone_management(self):
        """Zone assignment puts sites near zone centers into the correct zone."""
        zones = [
            ZoneConfig(zone_type=Zone.ENTANGLING, center_um=(0.0, 0.0), radius_um=10.0),
            ZoneConfig(zone_type=Zone.STORAGE, center_um=(50.0, 0.0), radius_um=10.0),
        ]
        arr = AtomArray(
            n_sites=4,
            geometry=ArrayGeometry.LINEAR,
            spacing_um=2.0,
            zones=zones,
        )
        arr.assign_zones()
        # All 4 sites are near (0,0) due to centering, so should be in zone 0
        in_zone0 = arr.sites_in_zone(0)
        assert len(in_zone0) > 0

    def test_interaction_graph_requires_loaded_atoms(self):
        """Interaction graph is empty if no atoms are loaded."""
        arr = AtomArray(n_sites=4, geometry=ArrayGeometry.LINEAR)
        # Not loaded -- all sites are empty
        graph = arr.interaction_graph(rabi_freq_mhz=1.0)
        assert len(graph) == 0

    def test_interaction_graph_with_loaded_atoms(self):
        """Interaction graph has edges between loaded atoms within blockade radius."""
        arr = AtomArray(n_sites=4, geometry=ArrayGeometry.LINEAR, spacing_um=4.0)
        arr.deterministic_load()
        graph = arr.interaction_graph(rabi_freq_mhz=1.0)
        # At 4 um spacing, blockade radius for Rb87 at 1 MHz ~ 9 um,
        # so nearest neighbours should be interacting
        assert len(graph) > 0
        for (i, j), strength in graph.items():
            assert i < j
            assert strength > 0

    def test_invalid_site_raises(self):
        """Accessing an out-of-range site raises ValueError."""
        arr = AtomArray(n_sites=5, geometry=ArrayGeometry.LINEAR)
        with pytest.raises(ValueError):
            arr.distance(0, 10)


# ======================================================================
# Gate tests
# ======================================================================


class TestNeutralAtomGates:
    """Tests for native Rydberg gate matrices and compilation."""

    def test_cz_matrix_diagonal(self):
        """CZ matrix is diag(1, 1, 1, -1)."""
        cz = NeutralAtomGateSet.cz_matrix()
        expected = np.diag([1.0, 1.0, 1.0, -1.0])
        np.testing.assert_allclose(cz, expected, atol=1e-12)

    def test_cz_matrix_unitary(self):
        """CZ matrix is unitary."""
        cz = NeutralAtomGateSet.cz_matrix()
        product = cz @ cz.conj().T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-12)

    def test_ccz_matrix_diagonal(self):
        """CCZ matrix is diag(1,1,1,1,1,1,1,-1) -- only |111> gets -1 phase."""
        ccz = NeutralAtomGateSet.ccz_matrix()
        assert ccz.shape == (8, 8)
        for i in range(7):
            np.testing.assert_allclose(ccz[i, i], 1.0, atol=1e-12)
        np.testing.assert_allclose(ccz[7, 7], -1.0, atol=1e-12)
        # Off-diagonal should be zero
        off_diag = ccz - np.diag(np.diag(ccz))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-12)

    def test_compile_cnot_has_one_cz(self):
        """CNOT compilation produces exactly 1 CZ entangling gate."""
        instructions = NeutralAtomGateSet.compile_cnot(0, 1)
        cz_count = NeutralAtomGateSet.cz_gate_count(instructions)
        assert cz_count == 1

    def test_compile_toffoli_has_one_ccz(self):
        """Toffoli compilation uses a single native CCZ gate.

        This is the key neutral-atom advantage: Toffoli = H + CCZ + H,
        requiring only 1 three-qubit entangling gate instead of 6 CZ gates.
        """
        instructions = NeutralAtomGateSet.compile_toffoli(0, 1, 2)
        ccz_count = NeutralAtomGateSet.ccz_gate_count(instructions)
        assert ccz_count == 1
        cz_count = NeutralAtomGateSet.cz_gate_count(instructions)
        assert cz_count == 0

    def test_compile_h_produces_ry_rz(self):
        """Hadamard decomposes into Ry(pi/2) followed by Rz(pi)."""
        instructions = NeutralAtomGateSet.compile_h(0)
        assert len(instructions) == 2
        assert instructions[0].gate_type == NativeGateType.RY
        np.testing.assert_allclose(instructions[0].params[0], math.pi / 2)
        assert instructions[1].gate_type == NativeGateType.RZ
        np.testing.assert_allclose(instructions[1].params[0], math.pi)

    def test_compile_x_produces_rxy_pi(self):
        """X gate decomposes into Rxy(pi, 0)."""
        instructions = NeutralAtomGateSet.compile_x(0)
        assert len(instructions) == 1
        assert instructions[0].gate_type == NativeGateType.RXY
        np.testing.assert_allclose(instructions[0].params[0], math.pi)

    def test_compile_z_produces_virtual_rz(self):
        """Z gate is a virtual Rz(pi) -- zero physical error."""
        instructions = NeutralAtomGateSet.compile_z(0)
        assert len(instructions) == 1
        assert instructions[0].gate_type == NativeGateType.RZ
        np.testing.assert_allclose(instructions[0].params[0], math.pi)

    def test_arbitrary_unitary_zyz_decomposition(self):
        """ZYZ decomposition of a known unitary produces the correct gate sequence."""
        # Hadamard matrix as the target unitary
        h_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        instructions = NeutralAtomGateSet.compile_arbitrary_unitary(h_mat, 0)
        # Should produce Rz and/or Ry instructions
        for inst in instructions:
            assert inst.gate_type in (NativeGateType.RZ, NativeGateType.RY)
        # Verify: multiply the native gate matrices and compare to H
        result = np.eye(2, dtype=np.complex128)
        for inst in instructions:
            if inst.gate_type == NativeGateType.RZ:
                result = NeutralAtomGateSet.rz_matrix(inst.params[0]) @ result
            elif inst.gate_type == NativeGateType.RY:
                result = NeutralAtomGateSet.ry_matrix(inst.params[0]) @ result
        # Up to global phase, result should match H
        # Check that result * H_dag is proportional to identity
        product = result @ h_mat.conj().T
        phase = product[0, 0]
        np.testing.assert_allclose(
            product, phase * np.eye(2), atol=1e-8
        )

    def test_compile_swap_has_three_cz(self):
        """SWAP = 3 CNOTs, each with 1 CZ, so 3 CZ gates total."""
        instructions = NeutralAtomGateSet.compile_swap(0, 1)
        cz_count = NeutralAtomGateSet.cz_gate_count(instructions)
        assert cz_count == 3

    def test_rotation_matrices_unitary(self):
        """All single-qubit rotation matrices should be unitary."""
        for theta in [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]:
            for matrix_fn in [
                NeutralAtomGateSet.rx_matrix,
                NeutralAtomGateSet.ry_matrix,
                NeutralAtomGateSet.rz_matrix,
            ]:
                m = matrix_fn(theta)
                product = m @ m.conj().T
                np.testing.assert_allclose(product, np.eye(2), atol=1e-12)


# ======================================================================
# Noise model tests
# ======================================================================


class TestNoiseModel:
    """Tests for the physics-based neutral-atom noise model."""

    def test_fidelities_in_valid_range(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """All gate fidelities must be in (0, 1]."""
        f1q = noise_model_rb87.single_qubit_gate_fidelity()
        f2q = noise_model_rb87.two_qubit_gate_fidelity()
        f3q = noise_model_rb87.three_qubit_gate_fidelity()
        for f, label in [(f1q, "1Q"), (f2q, "2Q"), (f3q, "3Q")]:
            assert 0.0 < f <= 1.0, f"{label} fidelity {f} out of range"

    def test_fidelity_ordering(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """1Q fidelity > 2Q fidelity > 3Q fidelity (more qubits = more error)."""
        f1q = noise_model_rb87.single_qubit_gate_fidelity()
        f2q = noise_model_rb87.two_qubit_gate_fidelity()
        f3q = noise_model_rb87.three_qubit_gate_fidelity()
        assert f1q > f2q, f"Expected 1Q ({f1q}) > 2Q ({f2q})"
        assert f2q > f3q, f"Expected 2Q ({f2q}) > 3Q ({f3q})"

    def test_gate_times_positive(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """All gate times must be positive."""
        assert noise_model_rb87.single_qubit_gate_time_us > 0
        assert noise_model_rb87.two_qubit_gate_time_us > 0
        assert noise_model_rb87.three_qubit_gate_time_us > 0

    def test_gate_time_ordering(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """1Q gate < 2Q gate < 3Q gate in duration."""
        t1q = noise_model_rb87.single_qubit_gate_time_us
        t2q = noise_model_rb87.two_qubit_gate_time_us
        t3q = noise_model_rb87.three_qubit_gate_time_us
        assert t1q < t2q < t3q

    def test_error_budget_keys(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """Error budget must contain all expected noise source categories."""
        budget = noise_model_rb87.error_budget()
        expected_keys = {
            "1q_intensity_noise",
            "1q_thermal_dephasing",
            "1q_atom_loss",
            "1q_trap_scattering",
            "2q_rydberg_decay",
            "2q_blockade_leakage",
            "2q_intensity_noise",
            "2q_atom_loss",
            "2q_crosstalk",
            "readout_error",
            "1q_total",
            "2q_total",
            "1q_fidelity",
            "2q_fidelity",
        }
        assert expected_keys.issubset(
            set(budget.keys())
        ), f"Missing keys: {expected_keys - set(budget.keys())}"

    def test_error_budget_values_non_negative(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """Individual error budget entries must be non-negative."""
        budget = noise_model_rb87.error_budget()
        for key, val in budget.items():
            if "fidelity" not in key:
                assert val >= 0, f"{key} = {val} is negative"

    def test_apply_noise_preserves_trace(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """apply_noise must preserve Tr(rho) = 1 for density matrices."""
        # Create a 2-qubit |00> density matrix
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho[0, 0] = 1.0

        rho_noisy = noise_model_rb87.apply_noise(
            rho, gate_type="cz", gate_time_us=1.0, target_qubits=(0, 1)
        )
        trace = np.real(np.trace(rho_noisy))
        np.testing.assert_allclose(trace, 1.0, atol=1e-6)

    def test_idle_decoherence_preserves_trace(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """idle_decoherence must preserve Tr(rho) = 1."""
        rho = np.zeros((4, 4), dtype=np.complex128)
        rho[0, 0] = 1.0

        rho_idle = noise_model_rb87.idle_decoherence(rho, idle_time_us=10.0)
        trace = np.real(np.trace(rho_idle))
        np.testing.assert_allclose(trace, 1.0, atol=1e-6)

    def test_apply_noise_single_qubit(
        self, noise_model_rb87: NeutralAtomNoiseModel
    ):
        """Single-qubit noise channel produces a valid density matrix."""
        rho = np.zeros((2, 2), dtype=np.complex128)
        rho[0, 0] = 1.0

        rho_noisy = noise_model_rb87.apply_noise(
            rho, gate_type="single", gate_time_us=0.5, target_qubits=(0,)
        )
        trace = np.real(np.trace(rho_noisy))
        np.testing.assert_allclose(trace, 1.0, atol=1e-6)
        # Eigenvalues of a valid density matrix must be non-negative
        eigenvalues = np.linalg.eigvalsh(rho_noisy)
        assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues}"


# ======================================================================
# Device preset tests
# ======================================================================


class TestDevicePresets:
    """Tests for commercial neutral-atom device presets."""

    def test_all_presets_load(self):
        """All device presets can be instantiated without error."""
        presets = DevicePresets.list_all()
        assert len(presets) >= 3

    @pytest.mark.parametrize(
        "preset_fn",
        [
            DevicePresets.quera_aquila,
            DevicePresets.atom_computing_1225,
            DevicePresets.pasqal_fresnel,
            DevicePresets.harvard_quera_48_logical,
        ],
    )
    def test_preset_has_valid_config(self, preset_fn):
        """Each preset has valid species, fidelities, and array config."""
        spec = preset_fn()
        assert isinstance(spec, DeviceSpec)
        assert spec.max_atoms > 0
        assert spec.max_sites >= spec.max_atoms
        assert 0.9 < spec.single_qubit_fidelity <= 1.0
        assert 0.9 < spec.two_qubit_fidelity <= 1.0
        assert 0.9 < spec.readout_fidelity <= 1.0

    def test_quera_aquila_sites(self):
        """QuEra Aquila has 256 sites as published."""
        spec = DevicePresets.quera_aquila()
        assert spec.max_sites == 256
        assert spec.vendor == "QuEra"

    def test_preset_creates_array(self):
        """Device preset can create a matching atom array."""
        spec = DevicePresets.quera_aquila()
        arr = spec.create_array(n_atoms=10)
        assert arr.n_sites == 10

    def test_preset_creates_noise_model(self):
        """Device preset can create a calibrated noise model."""
        spec = DevicePresets.quera_aquila()
        nm = spec.create_noise_model()
        assert isinstance(nm, NeutralAtomNoiseModel)
        f = nm.single_qubit_gate_fidelity()
        assert 0.9 < f <= 1.0

    def test_preset_info_dict(self):
        """Device info dict has expected keys."""
        spec = DevicePresets.quera_aquila()
        info = spec.info()
        for key in [
            "name",
            "vendor",
            "max_atoms",
            "species",
            "1q_fidelity",
            "2q_fidelity",
            "blockade_radius_um",
        ]:
            assert key in info, f"Missing key: {key}"


# ======================================================================
# Simulator tests -- ideal mode
# ======================================================================


class TestSimulatorIdeal:
    """Tests for ideal (state-vector) neutral-atom simulation."""

    def test_initial_state_is_ground(self, sim_ideal_2q: NeutralAtomSimulator):
        """Simulator starts in |00> state."""
        sv = sim_ideal_2q.statevector()
        np.testing.assert_allclose(sv[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(sv), 1.0, atol=1e-12)

    def test_bell_state(self, sim_ideal_2q: NeutralAtomSimulator):
        """H(0) + CNOT(0,1) produces Bell state (|00> + |11>)/sqrt(2)."""
        sim_ideal_2q.h(0)
        sim_ideal_2q.cnot(0, 1)
        sv = sim_ideal_2q.statevector()
        probs = np.abs(sv) ** 2

        # |00> and |11> should have probability 0.5 each
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-6)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-6)
        # |01> and |10> should have zero probability
        np.testing.assert_allclose(probs[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(probs[2], 0.0, atol=1e-6)

    def test_ghz_state(self, sim_ideal_3q: NeutralAtomSimulator):
        """H(0) + CNOT(0,1) + CNOT(0,2) produces GHZ state."""
        sim_ideal_3q.h(0)
        sim_ideal_3q.cnot(0, 1)
        sim_ideal_3q.cnot(0, 2)
        sv = sim_ideal_3q.statevector()
        probs = np.abs(sv) ** 2

        # |000> = index 0 and |111> = index 7 should have prob 0.5
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-6)
        np.testing.assert_allclose(probs[7], 0.5, atol=1e-6)
        # All other states should be zero
        for i in range(1, 7):
            np.testing.assert_allclose(probs[i], 0.0, atol=1e-6)

    def test_toffoli_flips_target(self, sim_ideal_3q: NeutralAtomSimulator):
        """Toffoli on |110> produces |111> (both controls are 1)."""
        sim_ideal_3q.x(0)  # |100>
        sim_ideal_3q.x(1)  # |110>
        sim_ideal_3q.toffoli(0, 1, 2)  # should flip qubit 2 -> |111>
        sv = sim_ideal_3q.statevector()
        probs = np.abs(sv) ** 2
        # |111> = index 7
        np.testing.assert_allclose(probs[7], 1.0, atol=1e-6)

    def test_toffoli_no_flip(self, sim_ideal_3q: NeutralAtomSimulator):
        """Toffoli on |100> leaves state unchanged (only one control is 1)."""
        sim_ideal_3q.x(0)  # |100>
        sim_ideal_3q.toffoli(0, 1, 2)
        sv = sim_ideal_3q.statevector()
        probs = np.abs(sv) ** 2
        # |100> = index 4
        np.testing.assert_allclose(probs[4], 1.0, atol=1e-6)

    def test_ccz_phase_on_111(self, sim_ideal_3q: NeutralAtomSimulator):
        """CCZ applies -1 relative phase to |111>.

        The X gate compiled as Rxy(pi, 0) = -i*X introduces a -i phase
        per application.  We verify CCZ by comparing the state with and
        without CCZ: the |111> amplitude should differ by a factor of -1.
        """
        # Reference: prepare |111> without CCZ
        ref = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        ref.x(0)
        ref.x(1)
        ref.x(2)
        sv_ref = ref.statevector()

        # Test: prepare |111> then apply CCZ
        sim_ideal_3q.x(0)
        sim_ideal_3q.x(1)
        sim_ideal_3q.x(2)
        sim_ideal_3q.ccz(0, 1, 2)
        sv_ccz = sim_ideal_3q.statevector()

        # CCZ should flip the sign of |111>
        ratio = sv_ccz[7] / sv_ref[7]
        np.testing.assert_allclose(ratio, -1.0, atol=1e-6)
        # Magnitude should be unchanged
        np.testing.assert_allclose(np.abs(sv_ccz[7]), 1.0, atol=1e-6)

    def test_ccz_no_phase_on_011(self, sim_ideal_3q: NeutralAtomSimulator):
        """CCZ does not change the phase of |011> (only |111> gets -1)."""
        # Reference: prepare |011> without CCZ
        ref = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        ref.x(1)
        ref.x(2)
        sv_ref = ref.statevector()

        # Test: prepare |011> then apply CCZ
        sim_ideal_3q.x(1)
        sim_ideal_3q.x(2)
        sim_ideal_3q.ccz(0, 1, 2)
        sv_ccz = sim_ideal_3q.statevector()

        # CCZ should not change the phase of |011>
        ratio = sv_ccz[3] / sv_ref[3]
        np.testing.assert_allclose(ratio, 1.0, atol=1e-6)

    def test_measure_all_returns_dict(self, sim_ideal_2q: NeutralAtomSimulator):
        """measure_all returns a dict of bitstrings to counts."""
        sim_ideal_2q.h(0)
        sim_ideal_2q.cnot(0, 1)
        shots = 1000
        counts = sim_ideal_2q.measure_all(shots=shots)
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == shots

    def test_measure_all_bell_distribution(
        self, sim_ideal_2q: NeutralAtomSimulator
    ):
        """Bell state measurement yields ~50/50 split of |00> and |11>."""
        sim_ideal_2q.h(0)
        sim_ideal_2q.cnot(0, 1)
        counts = sim_ideal_2q.measure_all(shots=5000)
        p00 = counts.get("00", 0) / 5000
        p11 = counts.get("11", 0) / 5000
        np.testing.assert_allclose(p00, 0.5, atol=0.05)
        np.testing.assert_allclose(p11, 0.5, atol=0.05)
        # |01> and |10> should be absent or negligible
        p01 = counts.get("01", 0) / 5000
        p10 = counts.get("10", 0) / 5000
        assert p01 < 0.01
        assert p10 < 0.01

    def test_reset_returns_ground_state(
        self, sim_ideal_2q: NeutralAtomSimulator
    ):
        """reset() restores the simulator to |00...0>."""
        sim_ideal_2q.h(0)
        sim_ideal_2q.cnot(0, 1)
        sim_ideal_2q.reset()
        sv = sim_ideal_2q.statevector()
        np.testing.assert_allclose(sv[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(np.abs(sv[1:]) ** 2), 0.0, atol=1e-12)

    def test_circuit_stats_gate_counts(
        self, sim_ideal_3q: NeutralAtomSimulator
    ):
        """circuit_stats correctly tracks gate counts."""
        sim_ideal_3q.h(0)
        sim_ideal_3q.cnot(0, 1)
        sim_ideal_3q.toffoli(0, 1, 2)
        stats = sim_ideal_3q.circuit_stats()
        assert isinstance(stats, CircuitStats)
        assert stats.total_gates > 0
        assert stats.cz_gate_count >= 1
        assert stats.ccz_gate_count >= 1

    def test_device_info(self, sim_ideal_2q: NeutralAtomSimulator):
        """device_info returns dict with expected keys."""
        info = sim_ideal_2q.device_info()
        assert isinstance(info, dict)
        for key in [
            "n_qubits",
            "species",
            "execution_mode",
            "spacing_um",
            "rabi_freq_mhz",
            "blockade_radius_um",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_pauli_x_gate(self, sim_ideal_2q: NeutralAtomSimulator):
        """X gate flips qubit 0 from |0> to |1>."""
        sim_ideal_2q.x(0)
        sv = sim_ideal_2q.statevector()
        probs = np.abs(sv) ** 2
        # |10> = index 2
        np.testing.assert_allclose(probs[2], 1.0, atol=1e-6)

    def test_pauli_z_gate(self, sim_ideal_2q: NeutralAtomSimulator):
        """Z gate on |0> leaves it unchanged (phase only)."""
        sim_ideal_2q.z(0)
        sv = sim_ideal_2q.statevector()
        # |00> should still have amplitude 1 (Z|0> = |0>)
        np.testing.assert_allclose(np.abs(sv[0]) ** 2, 1.0, atol=1e-6)

    def test_swap_gate(self, sim_ideal_2q: NeutralAtomSimulator):
        """SWAP on |10> produces |01>."""
        config = ArrayConfig(n_atoms=2)
        sim = NeutralAtomSimulator(config)
        sim.x(0)  # |10>
        # Use run_circuit to test SWAP via compile_swap pathway
        native_instrs = NeutralAtomGateSet.compile_swap(0, 1)
        for inst in native_instrs:
            matrix = sim._instruction_to_matrix(inst)
            if len(inst.qubits) == 1:
                sim._apply_single_qubit_gate(matrix, inst.qubits[0])
            elif len(inst.qubits) == 2:
                sim._apply_two_qubit_gate(matrix, inst.qubits[0], inst.qubits[1])
        sv = sim.statevector()
        probs = np.abs(sv) ** 2
        # |01> = index 1
        np.testing.assert_allclose(probs[1], 1.0, atol=1e-6)


# ======================================================================
# Simulator tests -- error handling
# ======================================================================


class TestSimulatorErrors:
    """Tests for simulator error handling."""

    def test_invalid_qubit_index(self, sim_ideal_2q: NeutralAtomSimulator):
        """Accessing a qubit beyond the array size raises ValueError."""
        with pytest.raises(ValueError):
            sim_ideal_2q.h(5)

    def test_negative_qubit_index(self, sim_ideal_2q: NeutralAtomSimulator):
        """Negative qubit index raises ValueError."""
        with pytest.raises(ValueError):
            sim_ideal_2q.x(-1)

    def test_same_qubit_cnot(self, sim_ideal_2q: NeutralAtomSimulator):
        """CNOT with control == target raises ValueError."""
        with pytest.raises(ValueError, match="different"):
            sim_ideal_2q.cnot(0, 0)

    def test_same_qubit_cz(self, sim_ideal_2q: NeutralAtomSimulator):
        """CZ with control == target raises ValueError."""
        with pytest.raises(ValueError, match="different"):
            sim_ideal_2q.cz(0, 0)

    def test_ccz_non_distinct_qubits(self):
        """CCZ with non-distinct qubits raises ValueError."""
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        with pytest.raises(ValueError, match="distinct"):
            sim.ccz(0, 0, 1)

    def test_invalid_execution_mode(self):
        """Invalid execution mode raises ValueError."""
        with pytest.raises(ValueError, match="execution_mode"):
            NeutralAtomSimulator(ArrayConfig(n_atoms=2), execution_mode="invalid")

    def test_statevector_unavailable_in_noisy(self):
        """statevector() raises RuntimeError in noisy mode."""
        sim = NeutralAtomSimulator(
            ArrayConfig(n_atoms=2), execution_mode="noisy"
        )
        with pytest.raises(RuntimeError, match="noisy"):
            sim.statevector()


# ======================================================================
# Simulator tests -- noisy mode
# ======================================================================


class TestSimulatorNoisy:
    """Tests for noisy (density-matrix) neutral-atom simulation."""

    def test_noisy_bell_fidelity_below_one(
        self, sim_noisy_2q: NeutralAtomSimulator
    ):
        """Noisy Bell state has fidelity < 1 but > 0.5."""
        sim_noisy_2q.h(0)
        sim_noisy_2q.cnot(0, 1)
        dm = sim_noisy_2q.density_matrix()
        # Ideal Bell state |Phi+> = (|00> + |11>)/sqrt(2)
        bell = np.zeros(4, dtype=np.complex128)
        bell[0] = bell[3] = 1.0 / math.sqrt(2)
        bell_dm = np.outer(bell, bell.conj())
        fidelity = np.real(np.trace(bell_dm @ dm))
        assert fidelity < 1.0, "Noisy fidelity should be < 1"
        assert fidelity > 0.5, f"Noisy fidelity too low: {fidelity}"

    def test_noisy_density_matrix_valid(
        self, sim_noisy_2q: NeutralAtomSimulator
    ):
        """Noisy density matrix must have Tr=1 and non-negative eigenvalues."""
        sim_noisy_2q.h(0)
        sim_noisy_2q.cnot(0, 1)
        dm = sim_noisy_2q.density_matrix()
        trace = np.real(np.trace(dm))
        np.testing.assert_allclose(trace, 1.0, atol=1e-6)
        eigenvalues = np.linalg.eigvalsh(dm)
        assert np.all(eigenvalues >= -1e-8), f"Negative eigenvalue: {min(eigenvalues)}"

    def test_noisy_measure_all(self, sim_noisy_2q: NeutralAtomSimulator):
        """measure_all in noisy mode returns valid shot counts."""
        sim_noisy_2q.h(0)
        sim_noisy_2q.cnot(0, 1)
        counts = sim_noisy_2q.measure_all(shots=1000)
        total = sum(counts.values())
        assert total == 1000

    def test_noisy_fidelity_estimate(
        self, sim_noisy_3q: NeutralAtomSimulator
    ):
        """fidelity_estimate returns a value < 1 after entangling gates."""
        sim_noisy_3q.h(0)
        sim_noisy_3q.cnot(0, 1)
        sim_noisy_3q.cnot(0, 2)
        f = sim_noisy_3q.fidelity_estimate()
        assert 0.0 < f < 1.0

    def test_noisy_device_info_has_error_budget(
        self, sim_noisy_2q: NeutralAtomSimulator
    ):
        """device_info in noisy mode includes the error budget."""
        info = sim_noisy_2q.device_info()
        assert "1q_gate_fidelity" in info
        assert "2q_gate_fidelity" in info
        assert "error_budget" in info


# ======================================================================
# Simulator tests -- pulse mode
# ======================================================================


class TestSimulatorPulse:
    """Tests for pulse-level Rydberg Hamiltonian simulation."""

    def test_pulse_mode_bell_state(self):
        """Pulse mode produces a valid Bell state."""
        sim = NeutralAtomSimulator(
            ArrayConfig(n_atoms=2), execution_mode="pulse"
        )
        sim.h(0)
        sim.cnot(0, 1)
        sv = sim.statevector()
        probs = np.abs(sv) ** 2
        # Should approximate Bell state
        np.testing.assert_allclose(probs[0], 0.5, atol=1e-6)
        np.testing.assert_allclose(probs[3], 0.5, atol=1e-6)

    def test_pulse_mode_toffoli(self):
        """Pulse mode Toffoli on |110> produces |111>."""
        sim = NeutralAtomSimulator(
            ArrayConfig(n_atoms=3), execution_mode="pulse"
        )
        sim.x(0)
        sim.x(1)
        sim.toffoli(0, 1, 2)
        sv = sim.statevector()
        probs = np.abs(sv) ** 2
        np.testing.assert_allclose(probs[7], 1.0, atol=1e-6)


# ======================================================================
# run_circuit interface tests
# ======================================================================


class TestRunCircuit:
    """Tests for the run_circuit tuple-based circuit interface."""

    def test_run_circuit_bell(self):
        """run_circuit with H+CNOT produces Bell state distribution."""
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        circuit = [("h", 0), ("cx", 0, 1)]
        counts = sim.run_circuit(circuit, shots=2000)
        total = sum(counts.values())
        assert total == 2000
        p00 = counts.get("00", 0) / 2000
        p11 = counts.get("11", 0) / 2000
        np.testing.assert_allclose(p00, 0.5, atol=0.05)
        np.testing.assert_allclose(p11, 0.5, atol=0.05)

    def test_run_circuit_unknown_gate(self):
        """Unknown gate name raises ValueError."""
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        with pytest.raises(ValueError, match="Unknown gate"):
            sim.run_circuit([("foobar", 0)], shots=100)

    def test_run_circuit_toffoli(self):
        """run_circuit Toffoli gate works."""
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        circuit = [("x", 0), ("x", 1), ("toffoli", 0, 1, 2)]
        counts = sim.run_circuit(circuit, shots=100)
        # Should be 100% |111>
        assert counts.get("111", 0) == 100


# ======================================================================
# Cross-backend comparison tests
# ======================================================================


class TestCrossBackendComparison:
    """Compare neutral-atom backend with trapped-ion and superconducting.

    All three backends should produce equivalent ideal results for standard
    quantum circuits.  Noisy fidelities will differ due to different
    physics-based noise models.
    """

    def test_bell_state_all_backends_ideal(self):
        """All three backends produce the same ideal Bell state probabilities."""
        from nqpu.ion_trap import TrapConfig, TrappedIonSimulator
        from nqpu.superconducting import DevicePresets as SCPresets
        from nqpu.superconducting import TransmonSimulator

        # Neutral atom
        na_sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        na_sim.h(0)
        na_sim.cnot(0, 1)
        na_probs = np.abs(na_sim.statevector()) ** 2

        # Trapped ion
        ion_sim = TrappedIonSimulator(TrapConfig(n_ions=2))
        ion_sim.h(0)
        ion_sim.cnot(0, 1)
        ion_probs = np.abs(ion_sim.statevector()) ** 2

        # Superconducting
        sc_config = SCPresets.IBM_HERON.build(num_qubits=2)
        sc_sim = TransmonSimulator(sc_config)
        sc_sim.h(0)
        sc_sim.cnot(0, 1)
        sc_probs = np.abs(sc_sim.statevector()) ** 2

        # All should match ideal Bell state
        expected = np.array([0.5, 0.0, 0.0, 0.5])
        np.testing.assert_allclose(na_probs, expected, atol=1e-6)
        np.testing.assert_allclose(ion_probs, expected, atol=1e-6)
        np.testing.assert_allclose(sc_probs, expected, atol=1e-6)

    def test_ideal_fidelity_is_one_all_backends(self):
        """All backends report fidelity ~1.0 in ideal mode."""
        from nqpu.ion_trap import TrapConfig, TrappedIonSimulator
        from nqpu.superconducting import DevicePresets as SCPresets
        from nqpu.superconducting import TransmonSimulator

        na_sim = NeutralAtomSimulator(ArrayConfig(n_atoms=2))
        na_sim.h(0)
        na_sim.cnot(0, 1)

        ion_sim = TrappedIonSimulator(TrapConfig(n_ions=2))
        ion_sim.h(0)
        ion_sim.cnot(0, 1)

        sc_sim = TransmonSimulator(SCPresets.IBM_HERON.build(num_qubits=2))
        sc_sim.h(0)
        sc_sim.cnot(0, 1)

        np.testing.assert_allclose(na_sim.fidelity_estimate(), 1.0, atol=1e-6)
        np.testing.assert_allclose(ion_sim.fidelity_estimate(), 1.0, atol=1e-6)
        np.testing.assert_allclose(sc_sim.fidelity_estimate(), 1.0, atol=1e-6)

    def test_noisy_fidelities_differ_between_backends(self):
        """Different backends have different noisy fidelities due to
        different physical noise processes."""
        from nqpu.ion_trap import TrapConfig, TrappedIonSimulator
        from nqpu.superconducting import DevicePresets as SCPresets
        from nqpu.superconducting import TransmonSimulator

        na_sim = NeutralAtomSimulator(
            ArrayConfig(n_atoms=2), execution_mode="noisy"
        )
        na_sim.h(0)
        na_sim.cnot(0, 1)
        na_fid = na_sim.fidelity_estimate()

        ion_sim = TrappedIonSimulator(
            TrapConfig(n_ions=2), execution_mode="noisy"
        )
        ion_sim.h(0)
        ion_sim.cnot(0, 1)
        ion_fid = ion_sim.fidelity_estimate()

        sc_sim = TransmonSimulator(
            SCPresets.IBM_HERON.build(num_qubits=2), execution_mode="noisy"
        )
        sc_sim.h(0)
        sc_sim.cnot(0, 1)
        sc_fid = sc_sim.fidelity_estimate()

        # All should be valid but differ
        for fid, name in [(na_fid, "neutral_atom"), (ion_fid, "ion"), (sc_fid, "sc")]:
            assert 0.0 < fid < 1.0, f"{name} fidelity {fid} out of (0,1)"

        # At least two backends should produce different fidelities
        fids = [na_fid, ion_fid, sc_fid]
        assert len(set(round(f, 6) for f in fids)) >= 2, (
            f"All fidelities identical: {fids}"
        )

    def test_ghz_state_all_backends_ideal(self):
        """All backends produce the same ideal GHZ state for 3 qubits."""
        from nqpu.ion_trap import TrapConfig, TrappedIonSimulator
        from nqpu.superconducting import DevicePresets as SCPresets
        from nqpu.superconducting import TransmonSimulator

        expected = np.zeros(8)
        expected[0] = 0.5
        expected[7] = 0.5

        # Neutral atom
        na_sim = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        na_sim.h(0)
        na_sim.cnot(0, 1)
        na_sim.cnot(0, 2)
        na_probs = np.abs(na_sim.statevector()) ** 2

        # Trapped ion
        ion_sim = TrappedIonSimulator(TrapConfig(n_ions=3))
        ion_sim.h(0)
        ion_sim.cnot(0, 1)
        ion_sim.cnot(0, 2)
        ion_probs = np.abs(ion_sim.statevector()) ** 2

        # Superconducting
        sc_sim = TransmonSimulator(SCPresets.IBM_HERON.build(num_qubits=3))
        sc_sim.h(0)
        sc_sim.cnot(0, 1)
        sc_sim.cnot(0, 2)
        sc_probs = np.abs(sc_sim.statevector()) ** 2

        np.testing.assert_allclose(na_probs, expected, atol=1e-6)
        np.testing.assert_allclose(ion_probs, expected, atol=1e-6)
        np.testing.assert_allclose(sc_probs, expected, atol=1e-6)

    def test_neutral_atom_toffoli_advantage(self):
        """Neutral atom Toffoli uses 1 CCZ gate vs 6+ CZ gates on other platforms.

        This demonstrates the key architectural advantage of neutral atoms
        for multi-qubit operations.
        """
        sim = NeutralAtomSimulator(ArrayConfig(n_atoms=3))
        sim.toffoli(0, 1, 2)
        stats = sim.circuit_stats()
        # Native CCZ: only 1 three-qubit entangling gate
        assert stats.ccz_gate_count == 1
        assert stats.cz_gate_count == 0
        # Total entangling gates = 1
        total_entangling = stats.cz_gate_count + stats.ccz_gate_count
        assert total_entangling == 1


# ======================================================================
# ArrayConfig validation tests
# ======================================================================


class TestArrayConfig:
    """Tests for ArrayConfig validation."""

    def test_rejects_zero_atoms(self):
        """n_atoms must be >= 1."""
        with pytest.raises(ValueError, match="n_atoms"):
            ArrayConfig(n_atoms=0)

    def test_rejects_negative_spacing(self):
        """spacing_um must be positive."""
        with pytest.raises(ValueError, match="spacing"):
            ArrayConfig(n_atoms=2, spacing_um=-1.0)

    def test_rejects_non_positive_rabi(self):
        """rabi_freq_mhz must be positive."""
        with pytest.raises(ValueError, match="rabi_freq"):
            ArrayConfig(n_atoms=2, rabi_freq_mhz=0.0)

    def test_all_species_accepted(self):
        """ArrayConfig works with all available species."""
        for species in ALL_SPECIES:
            config = ArrayConfig(n_atoms=2, species=species)
            sim = NeutralAtomSimulator(config)
            sim.h(0)
            sv = sim.statevector()
            assert np.isfinite(sv).all()
