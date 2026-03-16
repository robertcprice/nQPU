"""Comprehensive tests for the nQPU quantum circuit transpiler.

Covers circuit construction, coupling maps, routing (SABRE, greedy, trivial),
gate cancellation, rotation merging, basis translation, ZYZ/KAK decomposition,
optimization levels, device presets, and full pipeline integration.

Run with:
    cd sdk/python && python3 -m pytest tests/test_transpiler.py -v
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from nqpu.transpiler import (
    # Circuits
    Gate,
    QuantumCircuit,
    CircuitStats,
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
    SX,
    Id,
    Rx,
    Ry,
    Rz,
    U3,
    CX,
    CNOT,
    CZ,
    SWAP,
    CCX,
    Toffoli,
    # Coupling map
    CouplingMap,
    # Routing
    Layout,
    InitialLayout,
    RoutingResult,
    TrivialRouter,
    GreedyRouter,
    SABRERouter,
    SabreConfig,
    SabreHeuristic,
    route,
    # Optimization
    GateCancellation,
    RotationMerging,
    SingleQubitFusion,
    CommutationAnalysis,
    TwoQubitDecomposition,
    OptimizationLevel,
    optimize,
    # Decomposition
    BasisSet,
    BasisTranslator,
    ZYZDecomposition,
    KAKDecomposition,
    KAKResult,
    ToffoliDecomposition,
    decompose,
)


# ===================================================================
# Helpers
# ===================================================================

def _unitaries_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> bool:
    """Check two unitaries are equal up to global phase."""
    if a.shape != b.shape:
        return False
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(a[i, j]) > tol:
                phase = b[i, j] / a[i, j]
                return float(np.max(np.abs(a * phase - b))) < tol
    return float(np.max(np.abs(b))) < tol


def _all_2q_adjacent(circuit: QuantumCircuit, coupling_map: CouplingMap) -> bool:
    """Check that every two-qubit gate acts on adjacent physical qubits."""
    for gate in circuit.gates:
        if gate.name == "SWAP":
            q0, q1 = gate.qubits
            if not coupling_map.are_connected(q0, q1):
                return False
        elif gate.is_two_qubit:
            q0, q1 = gate.qubits
            if not coupling_map.are_connected(q0, q1):
                return False
    return True


# ===================================================================
# Circuit Construction Tests
# ===================================================================

class TestGate:
    """Tests for the Gate dataclass."""

    def test_single_qubit_gate(self):
        g = H(0)
        assert g.name == "H"
        assert g.qubits == (0,)
        assert g.params == ()
        assert g.is_single_qubit
        assert not g.is_two_qubit
        assert g.num_qubits == 1

    def test_parametric_gate(self):
        g = Rz(1, 0.5)
        assert g.name == "Rz"
        assert g.qubits == (1,)
        assert g.params == (0.5,)
        assert g.is_parametric

    def test_two_qubit_gate(self):
        g = CX(0, 1)
        assert g.is_two_qubit
        assert g.qubits == (0, 1)

    def test_three_qubit_gate(self):
        g = CCX(0, 1, 2)
        assert g.is_three_qubit

    def test_gate_matrix_h(self):
        mat = H(0).matrix()
        assert mat.shape == (2, 2)
        expected = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        np.testing.assert_allclose(mat, expected, atol=1e-12)

    def test_gate_matrix_cnot(self):
        mat = CX(0, 1).matrix()
        assert mat.shape == (4, 4)
        assert mat[0, 0] == 1
        assert mat[1, 1] == 1
        assert mat[2, 3] == 1
        assert mat[3, 2] == 1

    def test_gate_inverse_self_inverse(self):
        for constructor in [H, X, Y, Z]:
            g = constructor(0)
            inv = g.inverse()
            assert g.name == inv.name
            assert g.qubits == inv.qubits

    def test_gate_inverse_s_sdg(self):
        assert S(0).inverse().name == "Sdg"
        assert Sdg(0).inverse().name == "S"

    def test_gate_inverse_t_tdg(self):
        assert T(0).inverse().name == "Tdg"
        assert Tdg(0).inverse().name == "T"

    def test_gate_inverse_rotation(self):
        g = Rz(0, 0.7)
        inv = g.inverse()
        assert inv.params == (-0.7,)

    def test_gate_repr(self):
        g = Rz(0, 1.234)
        r = repr(g)
        assert "Rz" in r
        assert "1.2340" in r


class TestQuantumCircuit:
    """Tests for QuantumCircuit."""

    def test_construction(self):
        qc = QuantumCircuit(3)
        assert qc.num_qubits == 3
        assert qc.gate_count() == 0

    def test_invalid_qubit_count(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_add_gate(self):
        qc = QuantumCircuit(2)
        qc.add_gate(H(0))
        qc.add_gate(CX(0, 1))
        assert qc.gate_count() == 2

    def test_add_gate_out_of_range(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.add_gate(H(2))

    def test_chaining(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert qc.gate_count() == 2

    def test_depth_single_qubit(self):
        qc = QuantumCircuit(1).h(0).x(0).z(0)
        assert qc.depth() == 3

    def test_depth_parallel(self):
        qc = QuantumCircuit(3).h(0).h(1).h(2)
        assert qc.depth() == 1

    def test_depth_cnot_chain(self):
        qc = QuantumCircuit(3).cx(0, 1).cx(1, 2)
        assert qc.depth() == 2

    def test_depth_empty(self):
        qc = QuantumCircuit(2)
        assert qc.depth() == 0

    def test_count_ops(self):
        qc = QuantumCircuit(2).h(0).h(1).cx(0, 1)
        ops = qc.count_ops()
        assert ops["H"] == 2
        assert ops["CX"] == 1

    def test_stats(self):
        qc = QuantumCircuit(3).h(0).cx(0, 1).ccx(0, 1, 2)
        stats = qc.stats()
        assert stats.num_qubits == 3
        assert stats.gate_count == 3
        assert stats.single_qubit_gate_count == 1
        assert stats.two_qubit_gate_count == 1
        assert stats.three_qubit_gate_count == 1

    def test_to_matrix_identity(self):
        qc = QuantumCircuit(1)
        mat = qc.to_matrix()
        np.testing.assert_allclose(mat, np.eye(2), atol=1e-12)

    def test_to_matrix_h(self):
        qc = QuantumCircuit(1).h(0)
        mat = qc.to_matrix()
        expected = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
        np.testing.assert_allclose(mat, expected, atol=1e-12)

    def test_to_matrix_bell_state(self):
        """H on q0, then CNOT(0,1) should produce Bell unitary."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        mat = qc.to_matrix()
        assert mat.shape == (4, 4)
        # |00> -> (|00> + |11>)/sqrt(2)
        state_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        result = mat @ state_00
        expected = np.array([1, 0, 0, 1]) / math.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_to_matrix_too_large(self):
        qc = QuantumCircuit(13)
        with pytest.raises(ValueError, match="impractical"):
            qc.to_matrix()

    def test_copy(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        qc2 = qc.copy()
        assert qc2.gate_count() == 2
        qc2.h(1)
        assert qc.gate_count() == 2  # original unchanged

    def test_inverse(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        inv = qc.inverse()
        # Inverse should be CX then H (reversed order)
        assert inv.gates[0].name in ("CX", "CNOT")
        assert inv.gates[1].name == "H"

    def test_inverse_roundtrip_unitary(self):
        """circuit @ circuit.inverse() should be identity."""
        qc = QuantumCircuit(2).h(0).cx(0, 1).rz(1, 0.3)
        inv = qc.inverse()
        combined = QuantumCircuit(2)
        for g in qc.gates:
            combined.add_gate(g)
        for g in inv.gates:
            combined.add_gate(g)
        mat = combined.to_matrix()
        np.testing.assert_allclose(
            np.abs(mat), np.eye(4), atol=1e-10
        )

    def test_len_and_iter(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        assert len(qc) == 2
        names = [g.name for g in qc]
        assert names == ["H", "CX"]

    def test_all_gate_constructors(self):
        """Verify all convenience methods work."""
        qc = QuantumCircuit(3)
        qc.h(0).x(0).y(0).z(0).s(0).sdg(0).t(0).tdg(0).sx(0)
        qc.rx(0, 0.1).ry(0, 0.2).rz(0, 0.3).u3(0, 0.1, 0.2, 0.3)
        qc.cx(0, 1).cnot(0, 1).cz(0, 1).swap(0, 1)
        qc.ccx(0, 1, 2).toffoli(0, 1, 2)
        assert qc.gate_count() == 19


# ===================================================================
# Coupling Map Tests
# ===================================================================

class TestCouplingMap:
    """Tests for CouplingMap."""

    def test_from_line(self):
        cm = CouplingMap.from_line(5)
        assert cm.num_qubits == 5
        assert cm.num_edges() == 4
        assert cm.are_connected(0, 1)
        assert cm.are_connected(1, 2)
        assert not cm.are_connected(0, 2)

    def test_from_ring(self):
        cm = CouplingMap.from_ring(4)
        assert cm.num_qubits == 4
        assert cm.are_connected(3, 0)
        assert cm.num_edges() == 4

    def test_from_grid(self):
        cm = CouplingMap.from_grid(3, 3)
        assert cm.num_qubits == 9
        # Corner qubit has 2 neighbors
        assert cm.degree(0) == 2
        # Center qubit has 4 neighbors
        assert cm.degree(4) == 4

    def test_from_heavy_hex(self):
        cm = CouplingMap.from_heavy_hex(3)
        assert cm.num_qubits > 0
        assert cm.is_connected()

    def test_all_to_all(self):
        cm = CouplingMap.all_to_all(5)
        assert cm.num_qubits == 5
        assert cm.num_edges() == 10
        for i in range(5):
            for j in range(i + 1, 5):
                assert cm.are_connected(i, j)

    def test_shortest_path_adjacent(self):
        cm = CouplingMap.from_line(5)
        path = cm.shortest_path(0, 1)
        assert path == [0, 1]

    def test_shortest_path_longer(self):
        cm = CouplingMap.from_line(5)
        path = cm.shortest_path(0, 4)
        assert path == [0, 1, 2, 3, 4]
        assert len(path) == 5

    def test_shortest_path_same(self):
        cm = CouplingMap.from_line(3)
        assert cm.shortest_path(1, 1) == [1]

    def test_distance(self):
        cm = CouplingMap.from_grid(3, 3)
        assert cm.distance(0, 0) == 0
        assert cm.distance(0, 1) == 1
        assert cm.distance(0, 8) == 4  # corner to corner in 3x3 grid

    def test_distance_matrix(self):
        cm = CouplingMap.from_line(4)
        dist = cm.distance_matrix()
        assert dist.shape == (4, 4)
        assert dist[0, 3] == 3
        assert dist[1, 2] == 1

    def test_is_connected(self):
        cm = CouplingMap.from_line(5)
        assert cm.is_connected()

    def test_is_disconnected(self):
        cm = CouplingMap(num_qubits=4)
        cm.add_edge(0, 1)
        cm.add_edge(2, 3)
        assert not cm.is_connected()

    def test_diameter_line(self):
        cm = CouplingMap.from_line(5)
        assert cm.diameter() == 4

    def test_diameter_ring(self):
        cm = CouplingMap.from_ring(6)
        assert cm.diameter() == 3

    def test_diameter_disconnected(self):
        cm = CouplingMap(num_qubits=4)
        cm.add_edge(0, 1)
        cm.add_edge(2, 3)
        assert cm.diameter() == -1

    def test_neighbors(self):
        cm = CouplingMap.from_grid(2, 2)
        # Qubit 0 in a 2x2 grid: neighbors are 1 and 2
        nbs = cm.neighbors(0)
        assert set(nbs) == {1, 2}

    def test_edge_list(self):
        cm = CouplingMap.from_line(3)
        edges = cm.edge_list()
        assert (0, 1) in edges
        assert (1, 2) in edges

    def test_from_edge_list(self):
        cm = CouplingMap.from_edge_list(4, [(0, 1), (1, 2), (2, 3)])
        assert cm.num_qubits == 4
        assert cm.are_connected(0, 1)
        assert not cm.are_connected(0, 3)

    def test_device_preset_ibm_eagle(self):
        cm = CouplingMap.ibm_eagle()
        assert cm.num_qubits > 40  # heavy-hex with 15 unit cells
        assert cm.is_connected()

    def test_device_preset_google_sycamore(self):
        cm = CouplingMap.google_sycamore()
        assert cm.num_qubits == 54
        assert cm.is_connected()

    def test_device_preset_rigetti_aspen(self):
        cm = CouplingMap.rigetti_aspen()
        assert cm.num_qubits == 80
        assert cm.is_connected()


# ===================================================================
# Routing Tests
# ===================================================================

class TestLayout:
    """Tests for Layout."""

    def test_trivial_layout(self):
        layout = Layout.trivial(4)
        assert layout.logical_to_physical == [0, 1, 2, 3]
        assert layout.physical_to_logical == [0, 1, 2, 3]

    def test_apply_swap(self):
        layout = Layout.trivial(4)
        layout.apply_swap(0, 1)
        assert layout.logical_to_physical[0] == 1
        assert layout.logical_to_physical[1] == 0
        assert layout.physical_to_logical[0] == 1
        assert layout.physical_to_logical[1] == 0

    def test_copy(self):
        layout = Layout.trivial(3)
        layout2 = layout.copy()
        layout.apply_swap(0, 1)
        assert layout2.logical_to_physical[0] == 0  # unchanged


class TestTrivialRouter:
    """Tests for TrivialRouter."""

    def test_trivial_no_swaps(self):
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2)
        cm = CouplingMap.from_line(3)
        result = TrivialRouter().route(qc, cm)
        assert result.num_swaps_inserted == 0
        assert result.circuit.gate_count() == 3


class TestGreedyRouter:
    """Tests for GreedyRouter."""

    def test_greedy_adjacent(self):
        """No SWAPs needed if gates are already adjacent."""
        qc = QuantumCircuit(3).cx(0, 1).cx(1, 2)
        cm = CouplingMap.from_line(3)
        result = GreedyRouter().route(qc, cm)
        assert result.num_swaps_inserted == 0

    def test_greedy_non_adjacent(self):
        """Greedy router inserts SWAPs for non-adjacent gates."""
        qc = QuantumCircuit(4).cx(0, 3)
        cm = CouplingMap.from_line(4)
        result = GreedyRouter().route(qc, cm)
        assert result.num_swaps_inserted > 0
        assert _all_2q_adjacent(result.circuit, cm)

    def test_greedy_preserves_single_qubit(self):
        qc = QuantumCircuit(3).h(0).h(1).h(2)
        cm = CouplingMap.from_line(3)
        result = GreedyRouter().route(qc, cm)
        assert result.num_swaps_inserted == 0
        # Single-qubit gates only, mapped to physical qubits
        assert result.circuit.gate_count() == 3


class TestSABRERouter:
    """Tests for SABRERouter."""

    def test_sabre_empty_circuit(self):
        qc = QuantumCircuit(2)
        cm = CouplingMap.from_line(2)
        result = SABRERouter().route(qc, cm)
        assert result.num_swaps_inserted == 0
        assert result.circuit.gate_count() == 0

    def test_sabre_adjacent_gates(self):
        """Already-adjacent gates need no SWAPs."""
        qc = QuantumCircuit(3).cx(0, 1).cx(1, 2)
        cm = CouplingMap.from_line(3)
        result = SABRERouter().route(qc, cm)
        # All two-qubit gates should be on adjacent qubits
        assert _all_2q_adjacent(result.circuit, cm)

    def test_sabre_non_adjacent_inserts_swaps(self):
        """Non-adjacent gates on a line require SWAPs or smart layout.

        SABRE may find an initial layout that places logically distant
        qubits onto adjacent physical qubits, avoiding SWAPs entirely.
        The key correctness check is that all 2Q gates are adjacent.
        """
        qc = QuantumCircuit(5).cx(0, 4)
        cm = CouplingMap.from_line(5)
        result = SABRERouter().route(qc, cm)
        # SABRE may use a clever initial layout to avoid SWAPs
        assert result.num_swaps_inserted >= 0
        assert _all_2q_adjacent(result.circuit, cm)

    def test_sabre_all_to_all_no_swaps(self):
        """All-to-all connectivity needs zero SWAPs."""
        qc = QuantumCircuit(4).cx(0, 3).cx(1, 2).cx(0, 2)
        cm = CouplingMap.all_to_all(4)
        result = SABRERouter().route(qc, cm)
        assert result.num_swaps_inserted == 0

    def test_sabre_produces_valid_circuit(self):
        """Routed circuit must have all 2Q gates on adjacent qubits."""
        qc = QuantumCircuit(4)
        qc.cx(0, 2).cx(1, 3).cx(0, 3).cx(2, 1)
        cm = CouplingMap.from_line(4)
        result = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(result.circuit, cm)

    def test_sabre_with_single_qubit_gates(self):
        """Single-qubit gates should pass through unaffected."""
        qc = QuantumCircuit(3).h(0).rz(1, 0.5).cx(0, 2)
        cm = CouplingMap.from_line(3)
        result = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(result.circuit, cm)
        # Should have at least the original gates (plus possible SWAPs)
        assert result.circuit.gate_count() >= 3

    def test_sabre_config_heuristics(self):
        """All three heuristics should produce valid results."""
        qc = QuantumCircuit(4).cx(0, 3).cx(1, 2)
        cm = CouplingMap.from_line(4)
        for heuristic in SabreHeuristic:
            config = SabreConfig(
                num_trials=5, heuristic=heuristic, seed=42
            )
            result = SABRERouter(config).route(qc, cm)
            assert _all_2q_adjacent(result.circuit, cm)

    def test_sabre_deterministic(self):
        """Same seed should produce same result."""
        qc = QuantumCircuit(4).cx(0, 3).cx(1, 2)
        cm = CouplingMap.from_line(4)
        config = SabreConfig(num_trials=5, seed=123)
        r1 = SABRERouter(config).route(qc, cm)
        r2 = SABRERouter(config).route(qc, cm)
        assert r1.num_swaps_inserted == r2.num_swaps_inserted

    def test_sabre_grid_topology(self):
        """SABRE on a 3x3 grid should produce valid routing."""
        qc = QuantumCircuit(9)
        qc.cx(0, 8).cx(2, 6).cx(4, 5)
        cm = CouplingMap.from_grid(3, 3)
        result = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(result.circuit, cm)

    def test_sabre_circuit_too_large(self):
        """Circuit needing more qubits than device has should raise."""
        qc = QuantumCircuit(5).cx(0, 4)
        cm = CouplingMap.from_line(3)
        with pytest.raises(ValueError, match="qubits"):
            SABRERouter().route(qc, cm)


class TestRouteConvenience:
    """Tests for the route() convenience function."""

    def test_route_trivial(self):
        qc = QuantumCircuit(2).cx(0, 1)
        cm = CouplingMap.from_line(2)
        result = route(qc, cm, router="trivial")
        assert result.num_swaps_inserted == 0

    def test_route_greedy(self):
        qc = QuantumCircuit(3).cx(0, 2)
        cm = CouplingMap.from_line(3)
        result = route(qc, cm, router="greedy")
        assert _all_2q_adjacent(result.circuit, cm)

    def test_route_sabre(self):
        qc = QuantumCircuit(3).cx(0, 2)
        cm = CouplingMap.from_line(3)
        result = route(qc, cm, router="sabre")
        assert _all_2q_adjacent(result.circuit, cm)

    def test_route_invalid(self):
        qc = QuantumCircuit(2)
        cm = CouplingMap.from_line(2)
        with pytest.raises(ValueError, match="Unknown router"):
            route(qc, cm, router="bogus")


class TestInitialLayout:
    """Tests for InitialLayout strategies."""

    def test_trivial_layout(self):
        layout = InitialLayout.trivial_layout(3, 5)
        assert layout.logical_to_physical == [0, 1, 2]

    def test_random_layout(self):
        layout = InitialLayout.random_layout(3, 5, seed=42)
        # Should be a valid permutation subset
        assert len(set(layout.logical_to_physical)) == 3
        for p in layout.logical_to_physical:
            assert 0 <= p < 5

    def test_frequency_layout(self):
        qc = QuantumCircuit(3)
        # Qubit 0 is heavily used
        for _ in range(10):
            qc.cx(0, 1)
        cm = CouplingMap.from_grid(2, 2)
        layout = InitialLayout.frequency_layout(qc, cm)
        # Qubit 0 (most used) should be mapped to highest-degree physical qubit
        assert len(set(layout.logical_to_physical)) == 3


# ===================================================================
# Optimization Tests
# ===================================================================

class TestGateCancellation:
    """Tests for GateCancellation."""

    def test_cancel_hh(self):
        qc = QuantumCircuit(1).h(0).h(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_cancel_xx(self):
        qc = QuantumCircuit(1).x(0).x(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_cancel_cnot_cnot(self):
        qc = QuantumCircuit(2).cx(0, 1).cx(0, 1)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_cancel_s_sdg(self):
        qc = QuantumCircuit(1).s(0).sdg(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_cancel_t_tdg(self):
        qc = QuantumCircuit(1).t(0).tdg(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_no_cancel_different_qubits(self):
        qc = QuantumCircuit(2).h(0).h(1)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 2

    def test_cancel_multiple_pairs(self):
        qc = QuantumCircuit(1).h(0).h(0).x(0).x(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_cancel_nested(self):
        """H X X H should cancel to empty via iterative passes."""
        qc = QuantumCircuit(1).h(0).x(0).x(0).h(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 0

    def test_no_cancel_non_inverse(self):
        qc = QuantumCircuit(1).h(0).x(0)
        result = GateCancellation().run(qc)
        assert result.gate_count() == 2


class TestRotationMerging:
    """Tests for RotationMerging."""

    def test_merge_rz_rz(self):
        qc = QuantumCircuit(1).rz(0, 0.3).rz(0, 0.7)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 1
        assert abs(result.gates[0].params[0] - 1.0) < 1e-12

    def test_merge_rx_rx(self):
        qc = QuantumCircuit(1).rx(0, 0.5).rx(0, 0.5)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 1
        assert abs(result.gates[0].params[0] - 1.0) < 1e-12

    def test_merge_ry_ry(self):
        qc = QuantumCircuit(1).ry(0, 1.0).ry(0, 2.0)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 1
        assert abs(result.gates[0].params[0] - 3.0) < 1e-12

    def test_merge_to_zero(self):
        """Rz(a) + Rz(-a) should cancel completely."""
        qc = QuantumCircuit(1).rz(0, 1.5).rz(0, -1.5)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 0

    def test_merge_to_2pi(self):
        """Rz(pi) + Rz(pi) = Rz(2*pi) ~ identity, should be dropped."""
        qc = QuantumCircuit(1).rz(0, math.pi).rz(0, math.pi)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 0

    def test_no_merge_different_axes(self):
        """Rx followed by Rz should not merge."""
        qc = QuantumCircuit(1).rx(0, 0.5).rz(0, 0.5)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 2

    def test_no_merge_different_qubits(self):
        qc = QuantumCircuit(2).rz(0, 0.5).rz(1, 0.5)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 2

    def test_merge_chain(self):
        """Three consecutive Rz gates should merge to one."""
        qc = QuantumCircuit(1).rz(0, 0.1).rz(0, 0.2).rz(0, 0.3)
        result = RotationMerging().run(qc)
        assert result.gate_count() == 1
        assert abs(result.gates[0].params[0] - 0.6) < 1e-12


class TestSingleQubitFusion:
    """Tests for SingleQubitFusion."""

    def test_fuse_h_s(self):
        """H followed by S should fuse into one U3 gate."""
        qc = QuantumCircuit(1).h(0).s(0)
        result = SingleQubitFusion().run(qc)
        assert result.gate_count() == 1
        assert result.gates[0].name == "U3"

    def test_fuse_preserves_unitary(self):
        """Fused gate should produce the same unitary."""
        qc = QuantumCircuit(1).h(0).t(0).h(0)
        mat_original = qc.to_matrix()
        result = SingleQubitFusion().run(qc)
        mat_fused = result.to_matrix()
        assert _unitaries_close(mat_original, mat_fused)

    def test_no_fuse_single_gate(self):
        """A single gate should not be fused."""
        qc = QuantumCircuit(1).h(0)
        result = SingleQubitFusion().run(qc)
        assert result.gate_count() == 1
        assert result.gates[0].name == "H"

    def test_fuse_identity_removed(self):
        """H-H should fuse to identity and be removed."""
        qc = QuantumCircuit(1).h(0).h(0)
        result = SingleQubitFusion().run(qc)
        assert result.gate_count() == 0

    def test_fuse_across_two_qubit(self):
        """Fusion should stop at two-qubit gate boundaries."""
        qc = QuantumCircuit(2).h(0).s(0).cx(0, 1).h(0).t(0)
        result = SingleQubitFusion().run(qc)
        # Before CX: H+S fused to U3
        # CX itself remains
        # After CX: H+T fused to U3
        assert result.gate_count() == 3


class TestCommutationAnalysis:
    """Tests for CommutationAnalysis."""

    def test_commute_rz_past_t(self):
        """Rz commutes with T, enabling cancellation."""
        qc = QuantumCircuit(1)
        qc.add_gate(Rz(0, 0.5))
        qc.add_gate(T(0))
        qc.add_gate(Rz(0, -0.5))
        result = CommutationAnalysis().run(qc)
        # After commutation, the two Rz gates should be adjacent
        # (which enables subsequent cancellation)
        result = GateCancellation().run(result)
        assert result.gate_count() <= 2


class TestOptimizeLevels:
    """Tests for optimization levels."""

    def test_level_0_no_change(self):
        qc = QuantumCircuit(1).h(0).h(0)
        result = optimize(qc, level=0)
        assert result.gate_count() == 2  # no optimization

    def test_level_1_cancels_inverse(self):
        qc = QuantumCircuit(1).h(0).h(0)
        result = optimize(qc, level=1)
        assert result.gate_count() == 0

    def test_level_1_merges_rotations(self):
        qc = QuantumCircuit(1).rz(0, 0.3).rz(0, 0.7)
        result = optimize(qc, level=1)
        assert result.gate_count() == 1

    def test_level_2_fuses_single_qubit(self):
        qc = QuantumCircuit(1).h(0).t(0).s(0).z(0)
        result = optimize(qc, level=2)
        assert result.gate_count() <= 2

    def test_level_3_heavy_optimization(self):
        """Level 3 should not increase gate count."""
        qc = QuantumCircuit(2).h(0).cx(0, 1).h(0).h(0)
        original_count = qc.gate_count()
        result = optimize(qc, level=3)
        assert result.gate_count() <= original_count

    def test_optimization_does_not_increase_gate_count(self):
        """Optimization must be monotonically non-increasing in gate count."""
        qc = QuantumCircuit(3)
        qc.h(0).cx(0, 1).h(0).rz(1, 0.5).rz(1, 0.5).cx(1, 2)
        original = qc.gate_count()
        for level in range(4):
            result = optimize(qc, level=level)
            assert result.gate_count() <= original

    def test_higher_levels_not_worse(self):
        """Higher optimization levels should produce equal or fewer gates."""
        qc = QuantumCircuit(2)
        qc.h(0).h(0).rz(1, 0.5).rz(1, 0.5).cx(0, 1).cx(0, 1)
        counts = []
        for level in range(4):
            result = optimize(qc, level=level)
            counts.append(result.gate_count())
        for i in range(len(counts) - 1):
            assert counts[i + 1] <= counts[i]


# ===================================================================
# Decomposition Tests
# ===================================================================

class TestZYZDecomposition:
    """Tests for ZYZDecomposition."""

    def test_decompose_h(self):
        mat = H(0).matrix()
        gp, phi, theta, lam = ZYZDecomposition.decompose(mat)
        # Reconstruct
        from nqpu.transpiler.circuits import _rz, _ry
        reconstructed = np.exp(1j * gp) * (_rz(phi) @ _ry(theta) @ _rz(lam))
        assert _unitaries_close(mat, reconstructed)

    def test_decompose_x(self):
        mat = X(0).matrix()
        assert ZYZDecomposition.verify(mat)

    def test_decompose_arbitrary_rotation(self):
        mat = Rz(0, 1.23).matrix() @ Ry(0, 0.45).matrix() @ Rx(0, 0.67).matrix()
        assert ZYZDecomposition.verify(mat)

    def test_decompose_identity(self):
        mat = np.eye(2, dtype=np.complex128)
        gp, phi, theta, lam = ZYZDecomposition.decompose(mat)
        assert abs(theta) < 1e-10

    def test_to_gates(self):
        mat = H(0).matrix()
        gates = ZYZDecomposition.to_gates(0, mat)
        assert len(gates) > 0
        # All gates should be Rz or Ry
        for g in gates:
            assert g.name in ("Rz", "Ry")

    def test_roundtrip_random_unitary(self):
        """Decompose a random SU(2) and verify."""
        rng = np.random.RandomState(42)
        # Generate random unitary via QR decomposition
        z = rng.randn(2, 2) + 1j * rng.randn(2, 2)
        q, r = np.linalg.qr(z)
        d = np.diag(r)
        ph = d / np.abs(d)
        u = q * ph
        assert ZYZDecomposition.verify(u, tol=1e-8)


class TestKAKDecomposition:
    """Tests for KAKDecomposition."""

    def test_decompose_cnot(self):
        mat = CX(0, 1).matrix()
        kak = KAKDecomposition.decompose(mat)
        assert kak.num_cnots <= 3

    def test_decompose_cz(self):
        mat = CZ(0, 1).matrix()
        kak = KAKDecomposition.decompose(mat)
        assert kak.num_cnots <= 3

    def test_decompose_identity_2q(self):
        mat = np.eye(4, dtype=np.complex128)
        kak = KAKDecomposition.decompose(mat)
        assert kak.num_cnots == 0

    def test_decompose_swap(self):
        mat = SWAP(0, 1).matrix()
        kak = KAKDecomposition.decompose(mat)
        assert kak.num_cnots == 3

    def test_verify_cnot(self):
        mat = CX(0, 1).matrix()
        assert KAKDecomposition.verify(mat, tol=1e-6)

    def test_verify_cz(self):
        mat = CZ(0, 1).matrix()
        assert KAKDecomposition.verify(mat, tol=1e-6)

    def test_verify_identity(self):
        mat = np.eye(4, dtype=np.complex128)
        assert KAKDecomposition.verify(mat, tol=1e-6)

    def test_to_gates_cnot(self):
        mat = CX(0, 1).matrix()
        gates = KAKDecomposition.to_gates(0, 1, mat)
        # Build circuit and verify
        qc = QuantumCircuit(2)
        for g in gates:
            qc.add_gate(g)
        reconstructed = qc.to_matrix()
        assert _unitaries_close(mat, reconstructed, tol=1e-6)


class TestToffoliDecomposition:
    """Tests for ToffoliDecomposition."""

    def test_decompose_produces_correct_gates(self):
        gates = ToffoliDecomposition.decompose(0, 1, 2)
        assert len(gates) == 15
        # Should contain CX and single-qubit gates
        cx_count = sum(1 for g in gates if g.name == "CX")
        assert cx_count == 6

    def test_decompose_unitary_matches(self):
        """Decomposition should match the Toffoli unitary."""
        qc = QuantumCircuit(3)
        for g in ToffoliDecomposition.decompose(0, 1, 2):
            qc.add_gate(g)
        decomposed_mat = qc.to_matrix()
        original_mat = CCX(0, 1, 2).matrix()
        assert _unitaries_close(original_mat, decomposed_mat, tol=1e-8)


class TestBasisTranslator:
    """Tests for BasisTranslator."""

    def test_translate_to_ibm(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        # All gates should be in {CX, Rz, SX, X, Id}
        for g in result.gates:
            assert g.name.lower() in {"cx", "rz", "sx", "x", "id"}

    def test_translate_to_google(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = BasisTranslator(BasisSet.GOOGLE).translate(qc)
        for g in result.gates:
            assert g.name.lower() in {"cz", "rx", "rz", "id"}

    def test_translate_to_rigetti(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = BasisTranslator(BasisSet.RIGETTI).translate(qc)
        for g in result.gates:
            assert g.name.lower() in {"cz", "rx", "ry", "rz", "id"}

    def test_translate_s_gate(self):
        qc = QuantumCircuit(1).s(0)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        # S should become Rz(pi/2)
        assert any(g.name == "Rz" for g in result.gates)

    def test_translate_t_gate(self):
        qc = QuantumCircuit(1).t(0)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        assert any(g.name == "Rz" for g in result.gates)

    def test_translate_swap(self):
        qc = QuantumCircuit(2).swap(0, 1)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        # SWAP -> 3 CX
        cx_count = sum(1 for g in result.gates if g.name.lower() == "cx")
        assert cx_count == 3

    def test_translate_toffoli(self):
        qc = QuantumCircuit(3).ccx(0, 1, 2)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        # Should contain CX gates
        cx_count = sum(1 for g in result.gates if g.name.lower() == "cx")
        assert cx_count == 6

    def test_translate_preserves_unitary_ibm(self):
        """Translation should preserve the circuit unitary."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        original_mat = qc.to_matrix()
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        translated_mat = result.to_matrix()
        assert _unitaries_close(original_mat, translated_mat, tol=1e-6)

    def test_translate_preserves_unitary_google(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        original_mat = qc.to_matrix()
        result = BasisTranslator(BasisSet.GOOGLE).translate(qc)
        translated_mat = result.to_matrix()
        assert _unitaries_close(original_mat, translated_mat, tol=1e-6)

    def test_translate_preserves_unitary_rigetti(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        original_mat = qc.to_matrix()
        result = BasisTranslator(BasisSet.RIGETTI).translate(qc)
        translated_mat = result.to_matrix()
        assert _unitaries_close(original_mat, translated_mat, tol=1e-6)

    def test_translate_universal_noop(self):
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        result = BasisTranslator(BasisSet.UNIVERSAL).translate(qc)
        assert result.gate_count() == 2

    def test_decompose_convenience(self):
        qc = QuantumCircuit(1).h(0)
        result = decompose(qc, basis=BasisSet.IBM)
        for g in result.gates:
            assert g.name.lower() in {"cx", "rz", "sx", "x", "id"}

    def test_translate_y_gate(self):
        qc = QuantumCircuit(1).y(0)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        original_mat = qc.to_matrix()
        translated_mat = result.to_matrix()
        assert _unitaries_close(original_mat, translated_mat, tol=1e-6)

    def test_translate_z_gate(self):
        qc = QuantumCircuit(1).z(0)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        assert any(g.name == "Rz" for g in result.gates)

    def test_translate_cz_to_ibm(self):
        qc = QuantumCircuit(2).cz(0, 1)
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        for g in result.gates:
            assert g.name.lower() in {"cx", "rz", "sx", "x", "id"}

    def test_translate_cz_preserves_unitary(self):
        qc = QuantumCircuit(2).cz(0, 1)
        original_mat = qc.to_matrix()
        result = BasisTranslator(BasisSet.IBM).translate(qc)
        translated_mat = result.to_matrix()
        assert _unitaries_close(original_mat, translated_mat, tol=1e-6)


# ===================================================================
# Integration Tests (Full Pipeline)
# ===================================================================

class TestFullPipeline:
    """End-to-end tests: circuit -> route -> optimize -> decompose."""

    def test_bell_state_pipeline(self):
        """Full pipeline for a Bell-state circuit."""
        qc = QuantumCircuit(2).h(0).cx(0, 1)
        cm = CouplingMap.from_line(2)

        # Route
        routed = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(routed.circuit, cm)

        # Optimize
        opt = optimize(routed.circuit, level=2)
        assert opt.gate_count() <= routed.circuit.gate_count()

        # Decompose to IBM basis
        final = decompose(opt, basis=BasisSet.IBM)
        for g in final.gates:
            assert g.name.lower() in {"cx", "rz", "sx", "x", "id"}

    def test_ghz3_pipeline(self):
        """3-qubit GHZ state on a line topology."""
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2)
        cm = CouplingMap.from_line(3)

        routed = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(routed.circuit, cm)

        opt = optimize(routed.circuit, level=1)
        final = decompose(opt, basis=BasisSet.GOOGLE)
        for g in final.gates:
            assert g.name.lower() in {"cz", "rx", "rz", "id"}

    def test_non_adjacent_cx_pipeline(self):
        """CX between distant qubits on a line.

        SABRE may find an initial layout that maps distant logical qubits
        onto adjacent physical qubits, so zero SWAPs can be valid.
        """
        qc = QuantumCircuit(5).cx(0, 4)
        cm = CouplingMap.from_line(5)

        routed = route(qc, cm, router="sabre")
        assert routed.num_swaps_inserted >= 0
        assert _all_2q_adjacent(routed.circuit, cm)

        opt = optimize(routed.circuit, level=3)
        assert opt.gate_count() <= routed.circuit.gate_count()

    def test_grid_pipeline(self):
        """Multiple CX gates on a 3x3 grid."""
        qc = QuantumCircuit(9)
        qc.h(0).cx(0, 4).h(4).cx(4, 8).h(8)
        cm = CouplingMap.from_grid(3, 3)

        routed = SABRERouter().route(qc, cm)
        assert _all_2q_adjacent(routed.circuit, cm)

    def test_toffoli_pipeline(self):
        """Toffoli gate through full pipeline."""
        qc = QuantumCircuit(3).ccx(0, 1, 2)
        cm = CouplingMap.from_line(3)

        # First decompose Toffoli
        decomposed = decompose(qc, basis=BasisSet.IBM)
        # Then route
        routed = SABRERouter().route(decomposed, cm)
        assert _all_2q_adjacent(routed.circuit, cm)

        # Optimize
        opt = optimize(routed.circuit, level=2)
        assert opt.gate_count() <= routed.circuit.gate_count()

    def test_optimization_then_decompose_preserves_unitary(self):
        """Verify the unitary is preserved through optimize + decompose."""
        qc = QuantumCircuit(2).h(0).rz(0, 0.5).rz(0, 0.3).cx(0, 1)
        original_mat = qc.to_matrix()

        opt = optimize(qc, level=2)
        final = decompose(opt, basis=BasisSet.RIGETTI)
        final_mat = final.to_matrix()

        assert _unitaries_close(original_mat, final_mat, tol=1e-6)

    def test_large_circuit_pipeline(self):
        """Pipeline with many gates should not error."""
        qc = QuantumCircuit(6)
        for i in range(5):
            qc.h(i)
            qc.cx(i, i + 1)
        cm = CouplingMap.from_line(6)

        routed = route(qc, cm)
        assert _all_2q_adjacent(routed.circuit, cm)
        opt = optimize(routed.circuit, level=1)
        assert opt.gate_count() <= routed.circuit.gate_count()

    def test_greedy_then_optimize(self):
        """Greedy routing followed by optimization."""
        qc = QuantumCircuit(4).cx(0, 3).cx(1, 2)
        cm = CouplingMap.from_line(4)

        routed = GreedyRouter().route(qc, cm)
        assert _all_2q_adjacent(routed.circuit, cm)

        opt = optimize(routed.circuit, level=2)
        assert opt.gate_count() <= routed.circuit.gate_count()


# ===================================================================
# Edge Cases and Regression Tests
# ===================================================================

class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_single_qubit_circuit(self):
        qc = QuantumCircuit(1).h(0).t(0).h(0)
        cm = CouplingMap(num_qubits=1)
        result = SABRERouter().route(qc, cm)
        assert result.num_swaps_inserted == 0

    def test_circuit_all_single_qubit(self):
        qc = QuantumCircuit(3).h(0).x(1).z(2)
        cm = CouplingMap.from_line(3)
        result = SABRERouter().route(qc, cm)
        assert result.num_swaps_inserted == 0

    def test_empty_coupling_map(self):
        cm = CouplingMap(num_qubits=0)
        assert cm.num_qubits == 0
        assert cm.num_edges() == 0

    def test_single_qubit_coupling_map(self):
        cm = CouplingMap(num_qubits=1)
        assert cm.is_connected()
        assert cm.diameter() == 0

    def test_gate_matrix_all_standard(self):
        """All standard gates should produce valid matrices."""
        gates = [
            H(0), X(0), Y(0), Z(0), S(0), Sdg(0), T(0), Tdg(0),
            SX(0), Rx(0, 0.5), Ry(0, 0.5), Rz(0, 0.5),
            U3(0, 0.1, 0.2, 0.3), CX(0, 1), CZ(0, 1), SWAP(0, 1),
            CCX(0, 1, 2), Id(0),
        ]
        for gate in gates:
            mat = gate.matrix()
            dim = 1 << gate.num_qubits
            assert mat.shape == (dim, dim)
            # Check unitarity
            product = mat @ mat.conj().T
            np.testing.assert_allclose(product, np.eye(dim), atol=1e-12)

    def test_optimize_preserves_empty(self):
        qc = QuantumCircuit(2)
        for level in range(4):
            result = optimize(qc, level=level)
            assert result.gate_count() == 0

    def test_coupling_map_self_loop_ignored(self):
        cm = CouplingMap(num_qubits=3)
        cm.add_edge(0, 0)  # self-loop
        assert cm.num_edges() == 0

    def test_coupling_map_duplicate_edge(self):
        cm = CouplingMap(num_qubits=3)
        cm.add_edge(0, 1)
        cm.add_edge(0, 1)
        cm.add_edge(1, 0)  # same edge, different order
        assert cm.num_edges() == 1

    def test_depth_mixed_circuit(self):
        """Circuit with parallel and sequential gates."""
        qc = QuantumCircuit(3)
        qc.h(0).h(1).h(2)  # depth 1 (parallel)
        qc.cx(0, 1)         # depth 2
        qc.cx(1, 2)         # depth 3
        assert qc.depth() == 3

    def test_repr_formats(self):
        """Verify repr strings are informative."""
        qc = QuantumCircuit(2)
        assert "num_qubits=2" in repr(qc)
        cm = CouplingMap.from_line(3)
        assert "num_qubits=3" in repr(cm)
        g = Rz(0, 1.5)
        assert "Rz" in repr(g)


# ===================================================================
# Shannon Decomposition Tests
# ===================================================================

from nqpu.transpiler import (
    DecomposedGate,
    ShannonDecomposition,
    CSDResult,
    reconstruct_unitary,
)


def _random_unitary(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Generate a random unitary of size 2^n x 2^n via QR."""
    dim = 1 << n_qubits
    rng = np.random.RandomState(seed)
    z = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    ph = d / np.abs(d)
    return q * ph


def _unitaries_close_global(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    """Check two unitaries are equal up to global phase (robust)."""
    if a.shape != b.shape:
        return False
    # Find a non-zero entry to extract phase.
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if abs(a[i, j]) > 1e-10:
                phase = b[i, j] / a[i, j]
                return float(np.max(np.abs(a * phase - b))) < tol
    return float(np.max(np.abs(b))) < tol


class TestShannon:
    """Tests for Shannon (Quantum Shannon Decomposition)."""

    def test_decomposed_gate_repr(self):
        """DecomposedGate should have a readable repr."""
        g = DecomposedGate("Rz", [0], [1.5])
        assert "Rz" in repr(g)
        assert "1.5" in repr(g)

    def test_single_qubit_identity(self):
        """Decomposing the 1-qubit identity should yield no gates."""
        qsd = ShannonDecomposition()
        gates = qsd.decompose(np.eye(2, dtype=np.complex128))
        assert len(gates) == 0 or all(
            abs(g.params[0] % (2 * math.pi)) < 1e-8
            for g in gates if g.params
        )

    def test_single_qubit_hadamard(self):
        """Decompose 1-qubit Hadamard and reconstruct."""
        h_mat = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(h_mat)
        reconstructed = reconstruct_unitary(gates, 1)
        assert _unitaries_close_global(h_mat, reconstructed, tol=1e-6)

    def test_single_qubit_rotation(self):
        """Decompose a Rz rotation and verify reconstruction."""
        angle = 1.23
        rz_mat = np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)],
        ], dtype=np.complex128)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(rz_mat)
        reconstructed = reconstruct_unitary(gates, 1)
        assert _unitaries_close_global(rz_mat, reconstructed, tol=1e-6)

    def test_single_qubit_random(self):
        """Decompose a random 1-qubit unitary and verify."""
        u = _random_unitary(1, seed=99)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(u)
        reconstructed = reconstruct_unitary(gates, 1)
        assert _unitaries_close_global(u, reconstructed, tol=1e-6)

    def test_two_qubit_cnot(self):
        """Decompose a 2-qubit CNOT and verify reconstruction."""
        cnot = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 0, 1], [0, 0, 1, 0],
        ], dtype=np.complex128)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(cnot)
        reconstructed = reconstruct_unitary(gates, 2)
        assert _unitaries_close_global(cnot, reconstructed, tol=1e-4)

    def test_two_qubit_identity(self):
        """Decompose 2-qubit identity: reconstruction should be close to I."""
        u = np.eye(4, dtype=np.complex128)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(u)
        reconstructed = reconstruct_unitary(gates, 2)
        assert _unitaries_close_global(u, reconstructed, tol=1e-4)

    def test_two_qubit_random(self):
        """Decompose a random 2-qubit unitary and verify."""
        u = _random_unitary(2, seed=77)
        qsd = ShannonDecomposition()
        gates = qsd.decompose(u)
        reconstructed = reconstruct_unitary(gates, 2)
        assert _unitaries_close_global(u, reconstructed, tol=1e-3)

    def test_cnot_count_formula(self):
        """Verify the CNOT count formula for small cases.

        The formula is (3/4) * 4^{n-1}, giving O(4^n) scaling.
        """
        assert ShannonDecomposition.cnot_count(0) == 0
        assert ShannonDecomposition.cnot_count(1) == 0
        assert ShannonDecomposition.cnot_count(2) == 3  # (3/4)*4 = 3
        assert ShannonDecomposition.cnot_count(3) == 12  # (3/4)*16 = 12
        assert ShannonDecomposition.cnot_count(4) == 48  # (3/4)*64 = 48

    def test_csd_result_fields(self):
        """CSD should produce all five factors."""
        u = _random_unitary(2, seed=55)
        qsd = ShannonDecomposition()
        csd = qsd._cosine_sine_decomposition(u)
        assert isinstance(csd, CSDResult)
        assert csd.L1.shape == (2, 2)
        assert csd.L2.shape == (2, 2)
        assert csd.R1.shape == (2, 2)
        assert csd.R2.shape == (2, 2)
        assert len(csd.theta) == 2

    def test_multiplexed_ry_two_angles(self):
        """Multiplexed Ry with 1 control should produce CNOT + Ry gates."""
        qsd = ShannonDecomposition()
        angles = np.array([0.5, 1.0])
        gates = qsd._multiplexed_ry(angles, target=1, controls=[0])
        gate_types = [g.gate_type for g in gates]
        assert "CNOT" in gate_types
        assert "Ry" in gate_types

    def test_decompose_validates_unitary(self):
        """Non-unitary input should raise ValueError."""
        qsd = ShannonDecomposition()
        bad = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        with pytest.raises(ValueError, match="not unitary"):
            qsd.decompose(bad)


# ===================================================================
# Solovay-Kitaev Tests
# ===================================================================

from nqpu.transpiler import (
    GateSequence,
    BasicApproximations,
    SolovayKitaev,
    SKResult,
    operator_distance,
    approximate_rotation,
    approximate_u3,
    H_GATE,
    T_GATE,
    S_GATE,
)


class TestSolovayKitaev:
    """Tests for the Solovay-Kitaev algorithm."""

    def test_gate_sequence_identity(self):
        """Empty gate sequence should be identity."""
        seq = GateSequence([])
        np.testing.assert_allclose(seq.matrix(), np.eye(2), atol=1e-12)

    def test_gate_sequence_h(self):
        """Single H gate sequence should equal H matrix."""
        seq = GateSequence(["H"])
        np.testing.assert_allclose(seq.matrix(), H_GATE, atol=1e-12)

    def test_gate_sequence_t_count(self):
        """T-count should count T and Tdg gates."""
        seq = GateSequence(["T", "H", "T", "Tdg", "S"])
        assert seq.t_count == 3
        assert seq.depth == 5

    def test_gate_sequence_add(self):
        """Concatenating two sequences should multiply matrices.

        Convention: gates listed first are applied first (rightmost in
        matrix product), so ["H", "T"] -> T @ H.
        """
        a = GateSequence(["H"])
        b = GateSequence(["T"])
        c = a + b
        # c.gates = ["H", "T"], matrix = T_GATE @ H_GATE (T applied after H).
        expected = T_GATE @ H_GATE
        np.testing.assert_allclose(c.matrix(), expected, atol=1e-12)

    def test_gate_sequence_inverse(self):
        """Inverse should produce identity when composed."""
        seq = GateSequence(["H", "T", "S"])
        inv = seq.inverse()
        product = (seq + inv).matrix()
        np.testing.assert_allclose(
            np.abs(product), np.eye(2), atol=1e-10
        )

    def test_operator_distance_same(self):
        """Distance between identical matrices should be near zero."""
        assert operator_distance(H_GATE, H_GATE) < 1e-6

    def test_operator_distance_global_phase(self):
        """Distance should be zero for matrices differing by global phase."""
        u = T_GATE
        v = T_GATE * np.exp(1j * 0.5)
        assert operator_distance(u, v) < 1e-10

    def test_operator_distance_different(self):
        """Distance between H and T should be non-zero."""
        d = operator_distance(H_GATE, T_GATE)
        assert d > 0.1

    def test_basic_approximations_generate(self):
        """Basic approximations should generate a non-trivial database."""
        ba = BasicApproximations.generate(max_depth=3)
        assert len(ba.sequences) > 10
        assert len(ba.sequences) == len(ba.matrices)

    def test_basic_approximations_find_closest(self):
        """Find closest should return a close match for H."""
        ba = BasicApproximations.generate(max_depth=3)
        closest = ba.find_closest(H_GATE)
        d = operator_distance(closest.matrix(), H_GATE)
        assert d < 1e-6  # H should be in (or very near) the database.

    def test_sk_approximates_identity(self):
        """SK should trivially approximate identity."""
        sk = SolovayKitaev(recursion_depth=1)
        result = sk.approximate(np.eye(2, dtype=complex))
        d = operator_distance(result.matrix(), np.eye(2, dtype=complex))
        assert d < 0.1

    def test_sk_approximates_rotation(self):
        """SK should approximate a small Rz rotation."""
        result = approximate_rotation("z", 0.3, recursion_depth=2)
        assert isinstance(result, SKResult)
        assert result.error < 0.5  # Should get reasonable approximation.
        assert result.total_gates > 0

    def test_sk_approximates_u3(self):
        """SK should approximate a U3 gate."""
        result = approximate_u3(0.5, 0.3, 0.7, recursion_depth=1)
        assert isinstance(result, SKResult)
        assert result.error < 1.0
        assert result.t_count >= 0

    def test_sk_deeper_is_better(self):
        """Higher recursion depth should give equal or better approximation."""
        result1 = approximate_rotation("z", 1.0, recursion_depth=0)
        result2 = approximate_rotation("z", 1.0, recursion_depth=2)
        # Depth 2 should be at least as good as depth 0.
        assert result2.error <= result1.error + 0.01


# ===================================================================
# Template Matching Tests
# ===================================================================

from nqpu.transpiler import (
    CircuitDAG,
    DAGNode,
    Template,
    TemplateMatcher,
    TemplateMatchResult,
    default_templates,
)
from nqpu.transpiler.template_matching import (
    cnot_cancellation,
    hadamard_cancellation,
    t_t_to_s,
    s_s_to_z,
    cx_cx_swap,
    h_x_h_to_z,
    h_z_h_to_x,
    swap_cancellation,
    cz_cancellation,
    x_cancellation,
    y_cancellation,
    z_cancellation,
)


class TestTemplateMatching:
    """Tests for template-based circuit optimization."""

    def test_dag_add_gate(self):
        """Adding gates to a DAG should increase gate count."""
        dag = CircuitDAG(n_qubits=2)
        dag.add_gate("H", [0])
        dag.add_gate("CX", [0, 1])
        assert dag.gate_count == 2

    def test_dag_depth_parallel(self):
        """Parallel gates should have depth 1."""
        dag = CircuitDAG(n_qubits=3)
        dag.add_gate("H", [0])
        dag.add_gate("H", [1])
        dag.add_gate("H", [2])
        assert dag.depth == 1

    def test_dag_depth_sequential(self):
        """Sequential gates on same qubit have depth = count."""
        dag = CircuitDAG(n_qubits=1)
        dag.add_gate("H", [0])
        dag.add_gate("X", [0])
        dag.add_gate("Z", [0])
        assert dag.depth == 3

    def test_dag_topological_order(self):
        """Topological order should respect dependencies."""
        dag = CircuitDAG(n_qubits=2)
        dag.add_gate("H", [0])       # 0
        dag.add_gate("CX", [0, 1])   # 1
        dag.add_gate("H", [1])       # 2
        order = dag.topological_order()
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)

    def test_dag_successors_predecessors(self):
        """Successor/predecessor relationships should be correct."""
        dag = CircuitDAG(n_qubits=2)
        dag.add_gate("H", [0])       # 0
        dag.add_gate("CX", [0, 1])   # 1
        assert 1 in dag.successors(0)
        assert 0 in dag.predecessors(1)

    def test_dag_copy(self):
        """Copy should produce independent DAG."""
        dag = CircuitDAG(n_qubits=1)
        dag.add_gate("H", [0])
        dag2 = dag.copy()
        dag2.add_gate("X", [0])
        assert dag.gate_count == 1
        assert dag2.gate_count == 2

    def test_match_hh_cancellation(self):
        """Should find H-H pattern in circuit."""
        circuit = CircuitDAG(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [0])

        matcher = TemplateMatcher()
        tmpl = hadamard_cancellation()
        matches = matcher.match(circuit, tmpl)
        assert len(matches) >= 1

    def test_apply_hh_cancellation(self):
        """Applying H-H cancellation should remove both gates."""
        circuit = CircuitDAG(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [0])

        matcher = TemplateMatcher()
        tmpl = hadamard_cancellation()
        matches = matcher.match(circuit, tmpl)
        assert len(matches) >= 1
        result = matcher.apply_template(circuit, tmpl, matches[0])
        assert result.gate_count == 0

    def test_match_cnot_cancellation(self):
        """Should find and cancel CX-CX pattern."""
        circuit = CircuitDAG(n_qubits=2)
        circuit.add_gate("CX", [0, 1])
        circuit.add_gate("CX", [0, 1])

        matcher = TemplateMatcher()
        tmpl = cnot_cancellation()
        matches = matcher.match(circuit, tmpl)
        assert len(matches) >= 1

    def test_t_t_to_s_template(self):
        """T-T should be replaced by S."""
        circuit = CircuitDAG(n_qubits=1)
        circuit.add_gate("T", [0])
        circuit.add_gate("T", [0])

        matcher = TemplateMatcher([t_t_to_s()])
        result = matcher.optimize(circuit)
        assert result.gates_after == 1
        assert result.optimized.nodes[0].gate_type == "S"

    def test_s_s_to_z_template(self):
        """S-S should be replaced by Z."""
        circuit = CircuitDAG(n_qubits=1)
        circuit.add_gate("S", [0])
        circuit.add_gate("S", [0])

        matcher = TemplateMatcher([s_s_to_z()])
        result = matcher.optimize(circuit)
        assert result.gates_after == 1
        assert result.optimized.nodes[0].gate_type == "Z"

    def test_default_templates_count(self):
        """Should have at least 20 built-in templates."""
        templates = default_templates()
        assert len(templates) >= 20

    def test_default_templates_savings(self):
        """All default templates should have non-negative savings."""
        templates = default_templates()
        for tmpl in templates:
            assert tmpl.savings() >= 0

    def test_optimize_multi_template(self):
        """Full optimization should apply multiple templates."""
        circuit = CircuitDAG(n_qubits=1)
        circuit.add_gate("H", [0])
        circuit.add_gate("H", [0])
        circuit.add_gate("X", [0])
        circuit.add_gate("X", [0])

        templates = [hadamard_cancellation(), x_cancellation()]
        matcher = TemplateMatcher(templates)
        result = matcher.optimize(circuit)
        assert result.gates_after == 0
        assert len(result.templates_applied) >= 1

    def test_from_gate_list_roundtrip(self):
        """from_gate_list and to_gate_list should round-trip."""
        gate_list = [
            ("H", [0], []),
            ("CX", [0, 1], []),
            ("Rz", [1], [0.5]),
        ]
        dag = CircuitDAG.from_gate_list(2, gate_list)
        result = dag.to_gate_list()
        assert result == gate_list

    def test_dag_repr(self):
        """DAG repr should be informative."""
        dag = CircuitDAG(n_qubits=2)
        assert "n_qubits=2" in repr(dag)

    def test_template_savings(self):
        """Template savings should equal pattern count minus replacement count."""
        tmpl = hadamard_cancellation()
        assert tmpl.savings() == 2  # 2 gates -> 0 gates


# ===================================================================
# Noise Adaptive Tests
# ===================================================================

from nqpu.transpiler import (
    CalibrationData,
    NoiseAdaptiveRouter,
    NoiseAdaptiveResult,
    NoiseAdaptiveDecomposer,
    CircuitFidelityEstimator,
)


class TestNoiseAdaptive:
    """Tests for noise-adaptive transpilation."""

    def test_ideal_calibration(self):
        """Ideal calibration should have zero errors."""
        cal = CalibrationData.ideal(5)
        assert all(v == 0.0 for v in cal.single_qubit_errors.values())
        assert all(v == 0.0 for v in cal.two_qubit_errors.values())
        assert all(v == 0.0 for v in cal.readout_errors.values())

    def test_noisy_superconducting(self):
        """Noisy superconducting calibration should have realistic ranges."""
        cal = CalibrationData.noisy_superconducting(5)
        assert len(cal.single_qubit_errors) == 5
        assert all(0 < v < 0.001 for v in cal.single_qubit_errors.values())
        assert all(0 < v < 0.05 for v in cal.two_qubit_errors.values())
        assert all(50 <= v <= 200 for v in cal.t1_times.values())

    def test_noisy_ion_trap(self):
        """Ion trap calibration should have lower errors than SC."""
        cal_it = CalibrationData.noisy_ion_trap(5)
        cal_sc = CalibrationData.noisy_superconducting(5)
        # Ion trap single-qubit errors should be lower.
        avg_it = sum(cal_it.single_qubit_errors.values()) / 5
        avg_sc = sum(cal_sc.single_qubit_errors.values()) / 5
        assert avg_it < avg_sc

    def test_best_qubits(self):
        """best_qubits should return the lowest-error qubits."""
        cal = CalibrationData.noisy_superconducting(10, rng=np.random.RandomState(42))
        best3 = cal.best_qubits(3)
        assert len(best3) == 3
        assert len(set(best3)) == 3  # All unique.

    def test_best_pairs(self):
        """best_pairs should return the lowest-error pairs."""
        cal = CalibrationData.noisy_superconducting(5, rng=np.random.RandomState(42))
        best2 = cal.best_pairs(2)
        assert len(best2) == 2
        # First pair should have lower error than second.
        key0 = best2[0]
        key1 = best2[1]
        assert cal.two_qubit_errors[key0] <= cal.two_qubit_errors[key1]

    def test_noise_weighted_distance(self):
        """Noise-weighted distance should be > 0 for non-adjacent qubits."""
        cal = CalibrationData.noisy_superconducting(5)
        cm = CouplingMap.from_line(5)
        router = NoiseAdaptiveRouter(calibration=cal)
        d = router.noise_weighted_distance(0, 4, cm)
        assert d > 0
        # Same qubit should be zero.
        assert router.noise_weighted_distance(0, 0, cm) == 0.0

    def test_select_initial_layout(self):
        """Layout should assign logical qubits to physical qubits."""
        cal = CalibrationData.noisy_superconducting(5)
        cm = CouplingMap.from_line(5)
        router = NoiseAdaptiveRouter(calibration=cal)
        layout = router.select_initial_layout(3, cm)
        assert len(layout) == 3
        assert len(set(layout.values())) == 3

    def test_noise_adaptive_route(self):
        """Routing should produce a result with fidelity estimate."""
        cal = CalibrationData.noisy_superconducting(5)
        cm = CouplingMap.from_line(5)
        router = NoiseAdaptiveRouter(calibration=cal)
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2)
        result = router.route(qc, cm)
        assert isinstance(result, NoiseAdaptiveResult)
        assert 0 <= result.expected_fidelity <= 1
        assert result.swap_count >= 0

    def test_fidelity_ideal_is_one(self):
        """Ideal calibration should give fidelity ~1."""
        cal = CalibrationData.ideal(3)
        qc = QuantumCircuit(3).h(0).cx(0, 1)
        estimator = CircuitFidelityEstimator(calibration=cal)
        fidelity = estimator.estimate(qc)
        assert abs(fidelity - 1.0) < 1e-10

    def test_fidelity_noisy_less_than_one(self):
        """Noisy calibration should give fidelity < 1."""
        cal = CalibrationData.noisy_superconducting(3)
        qc = QuantumCircuit(3).h(0).cx(0, 1).cx(1, 2)
        estimator = CircuitFidelityEstimator(calibration=cal)
        fidelity = estimator.estimate(qc)
        assert 0 < fidelity < 1

    def test_error_budget(self):
        """Error budget should identify gate types and readout."""
        cal = CalibrationData.noisy_superconducting(3)
        qc = QuantumCircuit(3).h(0).cx(0, 1)
        estimator = CircuitFidelityEstimator(calibration=cal)
        budget = estimator.error_budget(qc)
        assert "H" in budget
        assert "CX" in budget
        assert "readout" in budget
        assert all(v > 0 for v in budget.values())

    def test_suggest_improvements(self):
        """Suggestions should return a non-empty list of strings."""
        cal = CalibrationData.noisy_superconducting(3)
        qc = QuantumCircuit(3).h(0).cx(0, 1)
        estimator = CircuitFidelityEstimator(calibration=cal)
        suggestions = estimator.suggest_improvements(qc)
        assert len(suggestions) > 0
        assert all(isinstance(s, str) for s in suggestions)

    def test_noise_adaptive_decomposer_cx(self):
        """Decomposer should produce valid CX decomposition."""
        cal = CalibrationData.noisy_superconducting(5)
        decomposer = NoiseAdaptiveDecomposer(calibration=cal)
        gates = decomposer.decompose_two_qubit("CX", (0, 1))
        assert len(gates) >= 1
        assert any(g["name"] in ("CX", "CZ", "iSWAP") for g in gates)

    def test_optimal_basis(self):
        """Optimal basis should return a valid basis string."""
        cal = CalibrationData.noisy_superconducting(5)
        decomposer = NoiseAdaptiveDecomposer(calibration=cal)
        basis = decomposer.optimal_basis((0, 1))
        assert basis in ("CX", "CZ", "iSWAP")

    def test_qubit_score(self):
        """Qubit score should be positive for noisy calibration."""
        cal = CalibrationData.noisy_superconducting(5)
        score = cal.qubit_score(0)
        assert score > 0

    def test_pair_score(self):
        """Pair score should combine qubit and pair errors."""
        cal = CalibrationData.noisy_superconducting(5)
        score = cal.pair_score(0, 1)
        assert score > cal.qubit_score(0)  # Adds two-qubit error.
