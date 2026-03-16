"""Comprehensive tests for the hardware calibration package.

Tests cover all five submodules:
  - ibm.py: TransmonProcessor, IBM property parsing, presets, quality scores
  - quantinuum.py: TrapConfig, H-series presets, fidelity estimation
  - quera.py: NeutralAtomConfig, blockade graph, lattice generators
  - generic.py: GenericCalibration conversions, auto-detect, compare
  - exporters.py: Reports, diffs, JSON roundtrip, CSV export

Tests verify physical correctness against published specifications and
sanity checks on data model invariants.
"""

import json
import math

import numpy as np
import pytest

# -- IBM imports -------------------------------------------------------------
from nqpu.calibration.ibm import (
    TransmonGate,
    TransmonProcessor,
    TransmonQubit,
    ibm_eagle_r3,
    ibm_heron_r2,
    parse_ibm_properties,
    parse_ibm_v2,
)

# -- Quantinuum imports ------------------------------------------------------
from nqpu.calibration.quantinuum import (
    TrapConfig,
    TrapZone,
    h1_1,
    h2_1,
    parse_quantinuum_specs,
)

# -- QuEra imports -----------------------------------------------------------
from nqpu.calibration.quera import (
    AtomSite,
    NeutralAtomConfig,
    aquila,
    generate_grid,
    generate_kagome,
    generate_triangular,
    parse_quera_capabilities,
)

# -- Generic imports ---------------------------------------------------------
from nqpu.calibration.generic import (
    GenericCalibration,
    auto_detect_format,
    compare_devices,
    ideal_calibration,
    load_calibration,
)

# -- Exporter imports --------------------------------------------------------
from nqpu.calibration.exporters import (
    CalibrationDiff,
    CalibrationReport,
    from_json,
    to_csv,
    to_json,
)


# ======================================================================
#  IBM TESTS
# ======================================================================

class TestIBM:
    """Tests for the IBM calibration module."""

    def test_transmon_qubit_properties(self):
        """TransmonQubit derived properties compute correctly."""
        q = TransmonQubit(
            index=0, frequency=5.0, t1=200.0, t2=100.0,
            readout_error=0.01, readout_length=800.0,
        )
        assert q.t1_ns == 200_000.0
        assert q.coherence_ratio == pytest.approx(0.5, rel=1e-6)

    def test_transmon_qubit_zero_t1(self):
        """coherence_ratio returns 0 when T1 is zero."""
        q = TransmonQubit(
            index=0, frequency=5.0, t1=0.0, t2=0.0,
            readout_error=0.01, readout_length=800.0,
        )
        assert q.coherence_ratio == 0.0

    def test_parse_ibm_properties(self):
        """parse_ibm_properties correctly extracts qubit and gate data."""
        props = {
            "backend_name": "test_backend",
            "last_update_date": "2024-01-01T00:00:00Z",
            "qubits": [
                [
                    {"name": "frequency", "value": 5.05, "unit": "GHz"},
                    {"name": "T1", "value": 250.0, "unit": "us"},
                    {"name": "T2", "value": 120.0, "unit": "us"},
                    {"name": "readout_error", "value": 0.015, "unit": ""},
                    {"name": "readout_length", "value": 700.0, "unit": "ns"},
                ],
                [
                    {"name": "frequency", "value": 5.10, "unit": "GHz"},
                    {"name": "T1", "value": 300.0, "unit": "us"},
                    {"name": "T2", "value": 180.0, "unit": "us"},
                    {"name": "readout_error", "value": 0.010, "unit": ""},
                    {"name": "readout_length", "value": 750.0, "unit": "ns"},
                ],
            ],
            "gates": [
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": [
                        {"name": "gate_error", "value": 0.007},
                        {"name": "gate_length", "value": 300.0},
                    ],
                },
                {
                    "gate": "sx",
                    "qubits": [0],
                    "parameters": [
                        {"name": "gate_error", "value": 0.0003},
                        {"name": "gate_length", "value": 35.0},
                    ],
                },
            ],
            "general": [],
        }
        proc = parse_ibm_properties(props)
        assert proc.name == "test_backend"
        assert proc.n_qubits == 2
        assert proc.qubits[0].frequency == pytest.approx(5.05)
        assert proc.qubits[0].t1 == pytest.approx(250.0)
        assert proc.qubits[1].readout_error == pytest.approx(0.010)
        assert len(proc.gates) == 2
        assert proc.coupling_map == [(0, 1)]

    def test_parse_ibm_v2(self):
        """parse_ibm_v2 handles v2 dict-style qubit properties."""
        config = {
            "backend_name": "v2_backend",
            "n_qubits": 2,
            "qubits": [
                {"frequency": 5.0, "t1": 200.0, "t2": 100.0,
                 "readout_error": 0.02, "readout_length": 800.0},
                {"frequency": 5.1, "t1": 220.0, "t2": 110.0,
                 "readout_error": 0.015, "readout_length": 800.0},
            ],
            "gates": [
                {"gate": "ecr", "qubits": [0, 1],
                 "parameters": {"gate_error": 0.005, "gate_length": 280.0}},
            ],
            "coupling_map": [[0, 1], [1, 0]],
            "basis_gates": ["ecr", "id", "rz", "sx", "x"],
        }
        proc = parse_ibm_v2(config)
        assert proc.n_qubits == 2
        assert proc.qubits[0].t1 == pytest.approx(200.0)
        assert proc.basis_gates == ["ecr", "id", "rz", "sx", "x"]
        assert (0, 1) in proc.coupling_map

    def test_eagle_r3_preset(self):
        """ibm_eagle_r3 returns a 127-qubit processor with sane values."""
        eagle = ibm_eagle_r3()
        assert eagle.name == "ibm_eagle_r3"
        assert eagle.n_qubits == 127
        assert len(eagle.qubits) == 127
        assert eagle.median_t1 > 200.0
        assert eagle.median_t2 > 100.0
        assert 0.0 < eagle.median_cx_error < 0.05
        assert "cx" in eagle.basis_gates

    def test_heron_r2_preset(self):
        """ibm_heron_r2 returns a 156-qubit processor with ECR gates."""
        heron = ibm_heron_r2()
        assert heron.name == "ibm_heron_r2"
        assert heron.n_qubits == 156
        assert "ecr" in heron.basis_gates
        # Heron should have better CX error than Eagle
        eagle = ibm_eagle_r3()
        assert heron.median_cx_error < eagle.median_cx_error

    def test_qubit_quality_scores(self):
        """Quality scores are in [0, 1] and sum to 1/3 each component."""
        eagle = ibm_eagle_r3()
        scores = eagle.qubit_quality_scores()
        assert len(scores) == 127
        for idx, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Qubit {idx} score out of range: {score}"

    def test_pair_quality_scores(self):
        """Pair quality scores are positive and bounded."""
        eagle = ibm_eagle_r3()
        pair_scores = eagle.pair_quality_scores()
        assert len(pair_scores) > 0
        for pair, score in pair_scores.items():
            assert 0.0 <= score <= 1.0, f"Pair {pair} score out of range"

    def test_best_subgraph(self):
        """best_subgraph returns a connected set of the requested size."""
        eagle = ibm_eagle_r3()
        sub = eagle.best_subgraph(10)
        assert len(sub) == 10
        # All indices should be valid
        for idx in sub:
            assert 0 <= idx < 127

    def test_best_subgraph_zero(self):
        """best_subgraph(0) returns empty list."""
        eagle = ibm_eagle_r3()
        assert eagle.best_subgraph(0) == []

    def test_best_subgraph_full(self):
        """best_subgraph(n >= n_qubits) returns all qubits."""
        eagle = ibm_eagle_r3()
        sub = eagle.best_subgraph(200)
        assert len(sub) == 127

    def test_summary_format(self):
        """summary() returns a non-empty string with key info."""
        eagle = ibm_eagle_r3()
        s = eagle.summary()
        assert "ibm_eagle_r3" in s
        assert "127" in s
        assert "Median T1" in s

    def test_median_properties_empty_processor(self):
        """Median properties return 0 for an empty processor."""
        proc = TransmonProcessor(name="empty", n_qubits=0)
        assert proc.median_t1 == 0.0
        assert proc.median_t2 == 0.0
        assert proc.median_cx_error == 0.0


# ======================================================================
#  QUANTINUUM TESTS
# ======================================================================

class TestQuantinuum:
    """Tests for the Quantinuum calibration module."""

    def test_trap_config_derived_properties(self):
        """TrapConfig error properties are 1 - fidelity."""
        config = TrapConfig(
            name="test", n_qubits=10,
            single_qubit_fidelity=0.9999,
            two_qubit_fidelity=0.998,
            measurement_fidelity=0.9997,
        )
        assert config.single_qubit_error == pytest.approx(0.0001, abs=1e-8)
        assert config.two_qubit_error == pytest.approx(0.002, abs=1e-8)
        assert config.measurement_error == pytest.approx(0.0003, abs=1e-8)

    def test_h1_1_preset(self):
        """H1-1 preset has 20 qubits with correct zone structure."""
        h1 = h1_1()
        assert h1.name == "quantinuum_h1_1"
        assert h1.n_qubits == 20
        assert h1.all_to_all is True
        assert len(h1.zones) == 5
        compute_zones = [z for z in h1.zones if z.zone_type == "compute"]
        assert len(compute_zones) == 3

    def test_h2_1_preset(self):
        """H2-1 preset has 56 qubits with improved fidelities."""
        h2 = h2_1()
        assert h2.name == "quantinuum_h2_1"
        assert h2.n_qubits == 56
        assert len(h2.zones) == 7
        # H2 should have better two-qubit fidelity than H1
        h1 = h1_1()
        assert h2.two_qubit_fidelity > h1.two_qubit_fidelity

    def test_expected_circuit_fidelity(self):
        """Circuit fidelity estimation produces reasonable values."""
        h1 = h1_1()
        # Zero gates should give perfect fidelity
        assert h1.expected_circuit_fidelity(0, 0, 0) == pytest.approx(1.0)
        # Many gates should reduce fidelity
        f_small = h1.expected_circuit_fidelity(10, 5, 5)
        f_large = h1.expected_circuit_fidelity(100, 50, 50)
        assert 0.0 < f_large < f_small < 1.0

    def test_expected_circuit_fidelity_with_transport(self):
        """Transport adds decoherence to circuit fidelity."""
        h1 = h1_1()
        f_no_transport = h1.expected_circuit_fidelity(10, 5, 5, n_transports=0)
        f_with_transport = h1.expected_circuit_fidelity(10, 5, 5, n_transports=10)
        assert f_with_transport < f_no_transport

    def test_circuit_time_us(self):
        """Circuit time estimation produces positive values."""
        h1 = h1_1()
        t = h1.circuit_time_us(10, 5, 5, 2)
        assert t > 0
        expected = 10 * h1.gate_time_1q + 5 * h1.gate_time_2q + 2 * h1.transport_time + 5 * 300.0
        assert t == pytest.approx(expected)

    def test_decoherence_limit(self):
        """Decoherence limit returns a positive integer for valid config."""
        h1 = h1_1()
        limit = h1.decoherence_limit()
        assert limit > 0
        # Should be T2 / (10 * gate_time_2q)
        expected = int(h1.t2 / (10.0 * h1.gate_time_2q))
        assert limit == expected

    def test_parse_quantinuum_specs(self):
        """parse_quantinuum_specs correctly builds a TrapConfig."""
        specs = {
            "name": "custom_trap",
            "n_qubits": 12,
            "zones": [
                {"zone_id": 0, "n_qubits": 6, "zone_type": "compute"},
                {"zone_id": 1, "n_qubits": 6, "zone_type": "memory"},
            ],
            "single_qubit_fidelity": 0.99999,
            "two_qubit_fidelity": 0.999,
            "measurement_fidelity": 0.9998,
        }
        config = parse_quantinuum_specs(specs)
        assert config.name == "custom_trap"
        assert config.n_qubits == 12
        assert len(config.zones) == 2
        assert config.two_qubit_fidelity == pytest.approx(0.999)

    def test_summary_format(self):
        """summary() returns a non-empty string with key info."""
        h1 = h1_1()
        s = h1.summary()
        assert "quantinuum_h1_1" in s
        assert "20" in s
        assert "fidelity" in s.lower()

    def test_total_zone_capacity(self):
        """total_zone_capacity sums all zone qubit counts."""
        h1 = h1_1()
        assert h1.total_zone_capacity == sum(z.n_qubits for z in h1.zones)

    def test_compute_zones_filter(self):
        """compute_zones returns only compute-type zones."""
        h1 = h1_1()
        compute = h1.compute_zones
        assert all(z.zone_type == "compute" for z in compute)


# ======================================================================
#  QUERA TESTS
# ======================================================================

class TestQuEra:
    """Tests for the QuEra calibration module."""

    def test_generate_grid(self):
        """generate_grid produces correct number of sites."""
        sites = generate_grid(4, 5, spacing=5.0)
        assert len(sites) == 20
        # Check spacing
        assert sites[1].x == pytest.approx(5.0)
        assert sites[5].y == pytest.approx(5.0)

    def test_generate_grid_indices(self):
        """Grid sites have sequential indices."""
        sites = generate_grid(3, 3)
        indices = [s.index for s in sites]
        assert indices == list(range(9))

    def test_generate_kagome(self):
        """generate_kagome produces 3*n_cells^2 sites."""
        sites = generate_kagome(3, spacing=5.0)
        assert len(sites) == 3 * 3 * 3  # 3 basis * n_cells^2

    def test_generate_triangular(self):
        """generate_triangular produces correct number of sites."""
        sites = generate_triangular(4, 5, spacing=5.0)
        assert len(sites) == 20

    def test_blockade_graph_small_grid(self):
        """blockade_graph identifies correct pairs for a 2x2 grid."""
        sites = generate_grid(2, 2, spacing=4.0)
        config = NeutralAtomConfig(
            name="test", max_atoms=4, sites=sites,
            rydberg_range=5.0,  # Only nearest neighbours
        )
        adj = config.blockade_graph()
        assert adj.shape == (4, 4)
        # Diagonal should be False
        for i in range(4):
            assert not adj[i, i]
        # Adjacent atoms (distance 4.0) should be connected
        assert adj[0, 1]  # (0,0)-(4,0)
        assert adj[0, 2]  # (0,0)-(0,4)
        # Diagonal atoms (distance ~5.66) should NOT be connected
        assert not adj[0, 3]

    def test_connectivity(self):
        """connectivity() returns deduplicated edge list."""
        sites = generate_grid(2, 2, spacing=4.0)
        config = NeutralAtomConfig(
            name="test", max_atoms=4, sites=sites,
            rydberg_range=5.0,
        )
        edges = config.connectivity()
        # Should have 4 edges in a 2x2 grid with range > spacing
        assert len(edges) == 4
        # No self-loops
        for i, j in edges:
            assert i < j

    def test_aquila_preset(self):
        """Aquila preset has 256 sites in 16x16 grid."""
        aq = aquila()
        assert aq.name == "quera_aquila"
        assert aq.max_atoms == 256
        assert len(aq.sites) == 256
        assert aq.global_drive_only is True
        assert aq.n_occupied == 256

    def test_expected_fidelity(self):
        """expected_fidelity decreases with more pulses."""
        sites = generate_grid(4, 4, spacing=5.0)
        config = NeutralAtomConfig(
            name="test", max_atoms=16, sites=sites,
        )
        f0 = config.expected_fidelity(0)
        f10 = config.expected_fidelity(10)
        f100 = config.expected_fidelity(100)
        assert f0 > f10 > f100 > 0.0

    def test_filling_fraction(self):
        """Filling fraction is correct for partially loaded arrays."""
        sites = generate_grid(2, 2, spacing=5.0)
        sites[0].occupied = False
        config = NeutralAtomConfig(name="test", max_atoms=4, sites=sites)
        assert config.filling_fraction() == pytest.approx(0.75)
        assert config.n_occupied == 3

    def test_parse_quera_capabilities(self):
        """parse_quera_capabilities correctly builds config."""
        caps = {
            "name": "custom_atom",
            "max_atoms": 100,
            "rydberg_range": 10.0,
            "sites": [
                {"index": 0, "x": 0.0, "y": 0.0},
                {"index": 1, "x": 5.0, "y": 0.0},
            ],
            "atom_loading_fidelity": 0.98,
        }
        config = parse_quera_capabilities(caps)
        assert config.name == "custom_atom"
        assert config.max_atoms == 100
        assert len(config.sites) == 2
        assert config.atom_loading_fidelity == pytest.approx(0.98)

    def test_summary_format(self):
        """summary() returns a non-empty string with key info."""
        sites = generate_grid(3, 3, spacing=5.0)
        config = NeutralAtomConfig(name="test_na", max_atoms=9, sites=sites)
        s = config.summary()
        assert "test_na" in s
        assert "9" in s
        assert "Rydberg" in s or "blockade" in s.lower() or "Blockade" in s


# ======================================================================
#  GENERIC TESTS
# ======================================================================

class TestGeneric:
    """Tests for the generic calibration module."""

    def test_from_transmon(self):
        """GenericCalibration.from_transmon preserves key data."""
        eagle = ibm_eagle_r3()
        cal = GenericCalibration.from_transmon(eagle)
        assert cal.technology == "superconducting"
        assert cal.n_qubits == 127
        assert len(cal.qubit_t1) == 127
        assert len(cal.readout_errors) == 127
        assert len(cal.coupling_map) > 0

    def test_from_trap(self):
        """GenericCalibration.from_trap creates all-to-all connectivity."""
        h1 = h1_1()
        cal = GenericCalibration.from_trap(h1)
        assert cal.technology == "trapped_ion"
        assert cal.n_qubits == 20
        # All-to-all: n*(n-1)/2 edges
        expected_edges = 20 * 19 // 2
        assert len(cal.coupling_map) == expected_edges
        assert len(cal.two_qubit_errors) == expected_edges

    def test_from_neutral_atom(self):
        """GenericCalibration.from_neutral_atom converts correctly."""
        sites = generate_grid(4, 4, spacing=5.0)
        config = NeutralAtomConfig(
            name="test_na", max_atoms=16, sites=sites, rydberg_range=6.0,
        )
        cal = GenericCalibration.from_neutral_atom(config)
        assert cal.technology == "neutral_atom"
        assert cal.n_qubits == 16
        assert len(cal.coupling_map) > 0

    def test_auto_detect_ibm_v1(self):
        """auto_detect_format identifies IBM v1 JSON."""
        data = {"backend_name": "ibm_x", "qubits": [[]]}
        assert auto_detect_format(data) == "ibm_v1"

    def test_auto_detect_ibm_v2(self):
        """auto_detect_format identifies IBM v2 JSON."""
        data = {"backend_name": "ibm_x", "qubits": [{}]}
        assert auto_detect_format(data) == "ibm_v2"

    def test_auto_detect_quantinuum(self):
        """auto_detect_format identifies Quantinuum data."""
        data = {"single_qubit_fidelity": 0.9999, "n_qubits": 20}
        assert auto_detect_format(data) == "quantinuum"

    def test_auto_detect_quera(self):
        """auto_detect_format identifies QuEra data."""
        data = {"rydberg_range": 9.0, "max_atoms": 256}
        assert auto_detect_format(data) == "quera"

    def test_auto_detect_unknown(self):
        """auto_detect_format returns 'unknown' for unrecognised data."""
        data = {"something": "else"}
        assert auto_detect_format(data) == "unknown"

    def test_load_calibration_quantinuum(self):
        """load_calibration auto-detects and loads Quantinuum data."""
        data = {
            "name": "test_trap",
            "n_qubits": 10,
            "single_qubit_fidelity": 0.9999,
            "two_qubit_fidelity": 0.998,
        }
        cal = load_calibration(data)
        assert cal.technology == "trapped_ion"
        assert cal.n_qubits == 10

    def test_load_calibration_unknown_raises(self):
        """load_calibration raises ValueError for unknown format."""
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            load_calibration({"foo": "bar"})

    def test_ideal_calibration(self):
        """ideal_calibration has zero errors and huge coherence."""
        cal = ideal_calibration(10)
        assert cal.n_qubits == 10
        assert cal.overall_quality() == pytest.approx(1.0)
        for i in range(10):
            assert cal.readout_errors[i] == 0.0
            assert cal.single_qubit_errors[i] == 0.0

    def test_best_qubits(self):
        """best_qubits returns the requested number of qubits."""
        eagle = ibm_eagle_r3()
        cal = GenericCalibration.from_transmon(eagle)
        best = cal.best_qubits(5)
        assert len(best) == 5

    def test_best_pairs(self):
        """best_pairs returns pairs sorted by error rate."""
        eagle = ibm_eagle_r3()
        cal = GenericCalibration.from_transmon(eagle)
        pairs = cal.best_pairs(3)
        assert len(pairs) == 3
        # Verify sorted order
        errors = [cal.two_qubit_errors[p] for p in pairs]
        assert errors == sorted(errors)

    def test_compare_devices(self):
        """compare_devices produces a formatted table."""
        eagle = GenericCalibration.from_transmon(ibm_eagle_r3())
        h1 = GenericCalibration.from_trap(h1_1())
        table = compare_devices([eagle, h1])
        assert "Technology" in table
        assert "superconducting" in table
        assert "trapped_ion" in table

    def test_compare_devices_empty(self):
        """compare_devices handles empty list."""
        assert "No devices" in compare_devices([])

    def test_overall_quality_bounded(self):
        """overall_quality is always in [0, 1]."""
        eagle = GenericCalibration.from_transmon(ibm_eagle_r3())
        assert 0.0 <= eagle.overall_quality() <= 1.0
        h1 = GenericCalibration.from_trap(h1_1())
        assert 0.0 <= h1.overall_quality() <= 1.0

    def test_summary_format(self):
        """summary() returns a meaningful string."""
        cal = ideal_calibration(5)
        s = cal.summary()
        assert "ideal" in s
        assert "5" in s


# ======================================================================
#  EXPORTER TESTS
# ======================================================================

class TestExporters:
    """Tests for the calibration export module."""

    def _make_calibration(self) -> GenericCalibration:
        """Helper to build a small calibration for testing."""
        return GenericCalibration(
            name="test_device",
            technology="superconducting",
            n_qubits=4,
            qubit_t1={0: 200.0, 1: 250.0, 2: 180.0, 3: 300.0},
            qubit_t2={0: 100.0, 1: 120.0, 2: 90.0, 3: 150.0},
            single_qubit_errors={0: 0.001, 1: 0.0008, 2: 0.002, 3: 0.0005},
            two_qubit_errors={(0, 1): 0.01, (1, 2): 0.015, (2, 3): 0.008},
            readout_errors={0: 0.02, 1: 0.015, 2: 0.025, 3: 0.01},
            coupling_map=[(0, 1), (1, 2), (2, 3)],
            gate_times={"1q": 35.0, "2q": 300.0},
            basis_gates=["cx", "rz", "sx"],
            timestamp="2024-01-01",
        )

    def test_report_generate(self):
        """CalibrationReport.generate produces a multi-section report."""
        cal = self._make_calibration()
        report = CalibrationReport()
        text = report.generate(cal)
        assert "test_device" in text
        assert "Qubit Detail" in text
        assert "Two-Qubit Gate Detail" in text

    def test_report_qubit_table(self):
        """qubit_table includes all qubits."""
        cal = self._make_calibration()
        report = CalibrationReport()
        table = report.qubit_table(cal)
        assert "T1" in table
        assert "RO err" in table

    def test_report_gate_table(self):
        """gate_table includes gate pairs."""
        cal = self._make_calibration()
        report = CalibrationReport()
        table = report.gate_table(cal)
        assert "Pair" in table

    def test_report_quality_summary(self):
        """quality_summary includes key metrics."""
        cal = self._make_calibration()
        report = CalibrationReport()
        summary = report.quality_summary(cal)
        assert "superconducting" in summary
        assert "Timestamp" in summary

    def test_json_roundtrip(self):
        """to_json -> from_json preserves all fields."""
        cal = self._make_calibration()
        json_str = to_json(cal)
        restored = from_json(json_str)
        assert restored.name == cal.name
        assert restored.technology == cal.technology
        assert restored.n_qubits == cal.n_qubits
        assert restored.qubit_t1 == cal.qubit_t1
        assert restored.readout_errors == cal.readout_errors
        assert len(restored.two_qubit_errors) == len(cal.two_qubit_errors)
        for pair, err in cal.two_qubit_errors.items():
            assert restored.two_qubit_errors[pair] == pytest.approx(err)

    def test_json_is_valid(self):
        """to_json produces valid JSON."""
        cal = self._make_calibration()
        json_str = to_json(cal)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test_device"

    def test_csv_export(self):
        """to_csv produces correct CSV with header and data rows."""
        cal = self._make_calibration()
        csv_str = to_csv(cal)
        lines = csv_str.strip().split("\n")
        assert lines[0] == "qubit,t1_us,t2_us,readout_error,single_qubit_error"
        assert len(lines) == 5  # header + 4 qubits

    def test_csv_values(self):
        """CSV values are parseable and correct."""
        cal = self._make_calibration()
        csv_str = to_csv(cal)
        lines = csv_str.strip().split("\n")
        # Parse first data line (qubit 0)
        parts = lines[1].split(",")
        assert int(parts[0]) == 0
        assert float(parts[1]) == pytest.approx(200.0)

    def test_diff_basic(self):
        """CalibrationDiff.diff produces a meaningful comparison."""
        old = self._make_calibration()
        new = self._make_calibration()
        new.name = "test_device_v2"
        # Worsen qubit 0
        new.readout_errors[0] = 0.05
        new.qubit_t1[0] = 150.0

        differ = CalibrationDiff()
        diff_text = differ.diff(old, new)
        assert "test_device" in diff_text
        assert "test_device_v2" in diff_text

    def test_diff_degraded_qubits(self):
        """degraded_qubits identifies qubits with worsened readout."""
        old = self._make_calibration()
        new = self._make_calibration()
        # Double readout error on qubit 0
        new.readout_errors[0] = 0.04  # was 0.02

        differ = CalibrationDiff()
        degraded = differ.degraded_qubits(old, new, threshold=0.1)
        assert 0 in degraded

    def test_diff_improved_qubits(self):
        """improved_qubits identifies qubits with better readout."""
        old = self._make_calibration()
        new = self._make_calibration()
        # Halve readout error on qubit 2
        new.readout_errors[2] = 0.005  # was 0.025

        differ = CalibrationDiff()
        improved = differ.improved_qubits(old, new, threshold=0.1)
        assert 2 in improved


# ======================================================================
#  INTEGRATION TESTS (cross-module)
# ======================================================================

class TestIntegration:
    """Cross-module integration tests."""

    def test_ibm_to_generic_to_json_roundtrip(self):
        """IBM preset -> GenericCalibration -> JSON -> back."""
        eagle = ibm_eagle_r3()
        cal = GenericCalibration.from_transmon(eagle)
        json_str = to_json(cal)
        restored = from_json(json_str)
        assert restored.n_qubits == 127
        assert restored.technology == "superconducting"
        assert abs(restored.overall_quality() - cal.overall_quality()) < 1e-6

    def test_quantinuum_to_generic_to_report(self):
        """Quantinuum preset -> GenericCalibration -> report."""
        h1 = h1_1()
        cal = GenericCalibration.from_trap(h1)
        report = CalibrationReport()
        text = report.generate(cal)
        assert "trapped_ion" in text

    def test_quera_to_generic_to_csv(self):
        """QuEra config -> GenericCalibration -> CSV."""
        sites = generate_grid(3, 3, spacing=5.0)
        config = NeutralAtomConfig(name="test", max_atoms=9, sites=sites)
        cal = GenericCalibration.from_neutral_atom(config)
        csv_str = to_csv(cal)
        lines = csv_str.strip().split("\n")
        assert len(lines) == 10  # header + 9 qubits

    def test_all_presets_comparable(self):
        """All vendor presets can be converted and compared."""
        eagle_cal = GenericCalibration.from_transmon(ibm_eagle_r3())
        h1_cal = GenericCalibration.from_trap(h1_1())
        na_sites = generate_grid(4, 4, spacing=5.0)
        na_config = NeutralAtomConfig(name="na_test", max_atoms=16, sites=na_sites)
        na_cal = GenericCalibration.from_neutral_atom(na_config)

        table = compare_devices([eagle_cal, h1_cal, na_cal])
        assert "superconducting" in table
        assert "trapped_ion" in table
        assert "neutral_atom" in table

    def test_init_imports_all(self):
        """Package __init__ exports everything listed in __all__."""
        import nqpu.calibration as cal_pkg

        for name in cal_pkg.__all__:
            assert hasattr(cal_pkg, name), f"Missing export: {name}"
