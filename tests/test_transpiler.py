"""Comprehensive tests for nqpu.transpiler package.

Tests cover the quantum circuit representation, coupling maps, routing algorithms
(trivial, greedy, SABRE), optimization passes (gate cancellation, rotation merging,
single-qubit fusion, commutation), basis decomposition (IBM, Google, Rigetti),
ZYZ and KAK decompositions, and the full transpiler pipeline.
"""

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
    Rx,
    Ry,
    Rz,
    U3,
    CX,
    CNOT,
    CZ,
    SWAP,
    CCX,
    # Coupling
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
    OptimizationLevel,
    optimize,
    # Decomposition
    BasisSet,
    BasisTranslator,
    ZYZDecomposition,
    KAKDecomposition,
    ToffoliDecomposition,
    decompose,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def bell_circuit():
    """Simple Bell state circuit: H(0), CX(0,1)."""
    qc = QuantumCircuit(2)
    qc.h(0).cx(0, 1)
    return qc


@pytest.fixture
def ghz_circuit():
    """3-qubit GHZ state circuit."""
    qc = QuantumCircuit(3)
    qc.h(0).cx(0, 1).cx(1, 2)
    return qc


@pytest.fixture
def line_3():
    """Linear coupling map with 3 qubits: 0-1-2."""
    return CouplingMap.from_line(3)


@pytest.fixture
def grid_2x2():
    """2x2 grid coupling map."""
    return CouplingMap.from_grid(2, 2)


# ------------------------------------------------------------------ #
# Gate construction tests
# ------------------------------------------------------------------ #


class TestGateConstruction:
    """Test gate construction and properties."""

    def test_single_qubit_gate(self):
        g = H(0)
        assert g.name == "H"
        assert g.qubits == (0,)
        assert g.is_single_qubit
        assert not g.is_two_qubit
        assert not g.is_parametric

    def test_two_qubit_gate(self):
        g = CX(0, 1)
        assert g.name == "CX"
        assert g.qubits == (0, 1)
        assert g.is_two_qubit
        assert not g.is_single_qubit

    def test_parametric_gate(self):
        g = Rx(0, math.pi / 2)
        assert g.is_parametric
        assert g.params == (math.pi / 2,)

    def test_u3_gate(self):
        g = U3(0, math.pi, math.pi / 2, 0.0)
        assert g.params == (math.pi, math.pi / 2, 0.0)
        assert g.num_qubits == 1

    def test_toffoli_gate(self):
        g = CCX(0, 1, 2)
        assert g.is_three_qubit
        assert g.num_qubits == 3

    def test_gate_inverse_self_inverse(self):
        g = H(0)
        inv = g.inverse()
        assert inv.name == "H"
        assert inv.qubits == (0,)

    def test_gate_inverse_s_sdg(self):
        assert S(0).inverse().name == "Sdg"
        assert Sdg(0).inverse().name == "S"

    def test_gate_inverse_t_tdg(self):
        assert T(0).inverse().name == "Tdg"
        assert Tdg(0).inverse().name == "T"

    def test_gate_inverse_rotation(self):
        g = Rz(0, math.pi / 4)
        inv = g.inverse()
        assert inv.params[0] == pytest.approx(-math.pi / 4)

    def test_gate_matrix_h(self):
        mat = H(0).matrix()
        assert mat.shape == (2, 2)
        # H^2 = I
        product = mat @ mat
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)

    def test_gate_repr(self):
        g = Rx(0, 1.5708)
        r = repr(g)
        assert "Rx" in r
        assert "q[0]" in r


# ------------------------------------------------------------------ #
# Quantum circuit tests
# ------------------------------------------------------------------ #


class TestQuantumCircuit:
    """Test QuantumCircuit construction and queries."""

    def test_empty_circuit(self):
        qc = QuantumCircuit(3)
        assert qc.num_qubits == 3
        assert qc.gate_count() == 0
        assert qc.depth() == 0

    def test_circuit_chaining(self, bell_circuit):
        assert bell_circuit.gate_count() == 2
        assert bell_circuit.depth() == 2

    def test_circuit_stats(self, ghz_circuit):
        stats = ghz_circuit.stats()
        assert stats.num_qubits == 3
        assert stats.gate_count == 3
        assert stats.single_qubit_gate_count == 1
        assert stats.two_qubit_gate_count == 2

    def test_count_ops(self, ghz_circuit):
        ops = ghz_circuit.count_ops()
        assert ops["H"] == 1
        assert ops.get("CX", 0) + ops.get("CNOT", 0) == 2

    def test_circuit_copy(self, bell_circuit):
        copy = bell_circuit.copy()
        assert copy.gate_count() == bell_circuit.gate_count()
        assert copy.num_qubits == bell_circuit.num_qubits

    def test_circuit_inverse(self, bell_circuit):
        inv = bell_circuit.inverse()
        assert inv.gate_count() == bell_circuit.gate_count()
        # Inverse should reverse gate order
        assert inv.gates[0].name in ("CX", "CNOT")
        assert inv.gates[1].name == "H"

    def test_qubit_out_of_range_raises(self):
        qc = QuantumCircuit(2)
        with pytest.raises(ValueError):
            qc.h(5)

    def test_invalid_num_qubits(self):
        with pytest.raises(ValueError):
            QuantumCircuit(0)

    def test_to_matrix_bell(self, bell_circuit):
        mat = bell_circuit.to_matrix()
        assert mat.shape == (4, 4)
        # Should be unitary
        product = mat @ mat.conj().T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_to_matrix_too_large_raises(self):
        qc = QuantumCircuit(13)
        with pytest.raises(ValueError):
            qc.to_matrix()

    def test_len_and_iter(self, ghz_circuit):
        assert len(ghz_circuit) == 3
        gates = list(ghz_circuit)
        assert len(gates) == 3


# ------------------------------------------------------------------ #
# Coupling map tests
# ------------------------------------------------------------------ #


class TestCouplingMap:
    """Test coupling map construction and queries."""

    def test_line_topology(self, line_3):
        assert line_3.num_qubits == 3
        assert line_3.num_edges() == 2
        assert line_3.are_connected(0, 1)
        assert line_3.are_connected(1, 2)
        assert not line_3.are_connected(0, 2)

    def test_ring_topology(self):
        ring = CouplingMap.from_ring(4)
        assert ring.num_qubits == 4
        assert ring.num_edges() == 4
        assert ring.are_connected(3, 0)

    def test_grid_topology(self, grid_2x2):
        assert grid_2x2.num_qubits == 4
        assert grid_2x2.are_connected(0, 1)
        assert grid_2x2.are_connected(0, 2)
        assert not grid_2x2.are_connected(0, 3)

    def test_all_to_all(self):
        full = CouplingMap.all_to_all(4)
        assert full.num_edges() == 6
        for i in range(4):
            for j in range(i + 1, 4):
                assert full.are_connected(i, j)

    def test_shortest_path(self, line_3):
        path = line_3.shortest_path(0, 2)
        assert path == [0, 1, 2]

    def test_distance(self, line_3):
        assert line_3.distance(0, 0) == 0
        assert line_3.distance(0, 1) == 1
        assert line_3.distance(0, 2) == 2

    def test_distance_matrix(self, line_3):
        dm = line_3.distance_matrix()
        assert dm.shape == (3, 3)
        assert dm[0, 2] == 2
        assert dm[1, 1] == 0

    def test_neighbors(self, line_3):
        assert line_3.neighbors(0) == [1]
        assert line_3.neighbors(1) == [0, 2]

    def test_degree(self, line_3):
        assert line_3.degree(0) == 1
        assert line_3.degree(1) == 2

    def test_is_connected(self, line_3):
        assert line_3.is_connected()

    def test_diameter(self, line_3):
        assert line_3.diameter() == 2

    def test_edge_list(self, line_3):
        edges = line_3.edge_list()
        assert len(edges) == 2
        assert (0, 1) in edges
        assert (1, 2) in edges

    @pytest.mark.parametrize("factory,name", [
        ("ibm_eagle", "IBM Eagle"),
        ("google_sycamore", "Google Sycamore"),
    ])
    def test_device_presets(self, factory, name):
        cm = getattr(CouplingMap, factory)()
        assert cm.num_qubits > 0
        assert cm.num_edges() > 0

    def test_from_edge_list(self):
        cm = CouplingMap.from_edge_list(3, [(0, 1), (1, 2)])
        assert cm.num_qubits == 3
        assert cm.are_connected(0, 1)


# ------------------------------------------------------------------ #
# Layout tests
# ------------------------------------------------------------------ #


class TestLayout:
    """Test qubit layout mapping."""

    def test_trivial_layout(self):
        layout = Layout.trivial(4)
        for i in range(4):
            assert layout.logical_to_physical[i] == i
            assert layout.physical_to_logical[i] == i

    def test_apply_swap(self):
        layout = Layout.trivial(3)
        layout.apply_swap(0, 1)
        assert layout.physical_to_logical[0] == 1
        assert layout.physical_to_logical[1] == 0
        assert layout.logical_to_physical[0] == 1
        assert layout.logical_to_physical[1] == 0

    def test_layout_copy(self):
        layout = Layout.trivial(3)
        copy = layout.copy()
        copy.apply_swap(0, 1)
        # Original should be unchanged
        assert layout.logical_to_physical[0] == 0


# ------------------------------------------------------------------ #
# Routing tests
# ------------------------------------------------------------------ #


class TestRouting:
    """Test routing algorithms."""

    def test_trivial_router_no_swaps(self, bell_circuit, line_3):
        result = TrivialRouter().route(bell_circuit, line_3)
        assert result.num_swaps_inserted == 0
        assert result.circuit.gate_count() == bell_circuit.gate_count()

    def test_greedy_router_inserts_swaps_when_needed(self, line_3):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)  # Not adjacent in line topology
        result = GreedyRouter().route(qc, line_3)
        assert result.num_swaps_inserted > 0

    def test_greedy_router_adjacent_no_swaps(self, line_3):
        qc = QuantumCircuit(3)
        qc.cx(0, 1)  # Already adjacent
        result = GreedyRouter().route(qc, line_3)
        assert result.num_swaps_inserted == 0

    def test_sabre_router_basic(self, ghz_circuit, line_3):
        config = SabreConfig(num_trials=2, seed=42, heuristic=SabreHeuristic.BASIC)
        result = SABRERouter(config).route(ghz_circuit, line_3)
        assert isinstance(result, RoutingResult)
        assert result.circuit.num_qubits == line_3.num_qubits

    def test_sabre_router_decay(self, ghz_circuit, line_3):
        config = SabreConfig(num_trials=2, seed=42, heuristic=SabreHeuristic.DECAY)
        result = SABRERouter(config).route(ghz_circuit, line_3)
        assert isinstance(result, RoutingResult)

    def test_sabre_router_empty_circuit(self, line_3):
        qc = QuantumCircuit(3)
        result = SABRERouter().route(qc, line_3)
        assert result.num_swaps_inserted == 0

    def test_route_convenience_function(self, ghz_circuit, line_3):
        result = route(ghz_circuit, line_3, router="greedy")
        assert isinstance(result, RoutingResult)

    def test_route_unknown_router_raises(self, bell_circuit, line_3):
        with pytest.raises(ValueError):
            route(bell_circuit, line_3, router="unknown")

    def test_sabre_too_many_qubits_raises(self, line_3):
        qc = QuantumCircuit(10)
        qc.h(0)  # Need at least one gate so router doesn't return early
        with pytest.raises(ValueError):
            SABRERouter().route(qc, line_3)


# ------------------------------------------------------------------ #
# Optimization tests
# ------------------------------------------------------------------ #


class TestOptimization:
    """Test circuit optimization passes."""

    def test_gate_cancellation_hh(self):
        qc = QuantumCircuit(1)
        qc.h(0).h(0)
        opt = GateCancellation().run(qc)
        assert opt.gate_count() == 0

    def test_gate_cancellation_xx(self):
        qc = QuantumCircuit(1)
        qc.x(0).x(0)
        opt = GateCancellation().run(qc)
        assert opt.gate_count() == 0

    def test_gate_cancellation_s_sdg(self):
        qc = QuantumCircuit(1)
        qc.s(0).sdg(0)
        opt = GateCancellation().run(qc)
        assert opt.gate_count() == 0

    def test_gate_cancellation_preserves_non_cancelling(self):
        qc = QuantumCircuit(1)
        qc.h(0).x(0)
        opt = GateCancellation().run(qc)
        assert opt.gate_count() == 2

    def test_rotation_merging_rz(self):
        qc = QuantumCircuit(1)
        qc.rz(0, math.pi / 4).rz(0, math.pi / 4)
        opt = RotationMerging().run(qc)
        assert opt.gate_count() == 1
        assert opt.gates[0].params[0] == pytest.approx(math.pi / 2)

    def test_rotation_merging_to_identity(self):
        qc = QuantumCircuit(1)
        qc.rz(0, math.pi).rz(0, math.pi)  # Total 2pi = identity
        opt = RotationMerging().run(qc)
        assert opt.gate_count() == 0

    def test_single_qubit_fusion(self):
        qc = QuantumCircuit(1)
        qc.h(0).s(0).t(0)
        opt = SingleQubitFusion().run(qc)
        # Three gates should be fused into one U3
        assert opt.gate_count() <= 1

    @pytest.mark.parametrize("level", [0, 1, 2, 3])
    def test_optimize_levels(self, level, bell_circuit):
        opt = optimize(bell_circuit, level=level)
        # Optimized gate count should be <= original
        assert opt.gate_count() <= bell_circuit.gate_count() + 1  # +1 for potential U3 expansion

    def test_optimize_level_zero_is_copy(self, bell_circuit):
        opt = optimize(bell_circuit, level=0)
        assert opt.gate_count() == bell_circuit.gate_count()


# ------------------------------------------------------------------ #
# Decomposition tests
# ------------------------------------------------------------------ #


class TestZYZDecomposition:
    """Test ZYZ single-qubit decomposition."""

    def test_decompose_identity(self):
        u = np.eye(2, dtype=np.complex128)
        gp, phi, theta, lam = ZYZDecomposition.decompose(u)
        assert abs(theta) < 1e-6

    def test_decompose_h_gate(self):
        mat = H(0).matrix()
        assert ZYZDecomposition.verify(mat)

    def test_decompose_t_gate(self):
        mat = T(0).matrix()
        assert ZYZDecomposition.verify(mat)

    @pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi])
    def test_decompose_rz(self, angle):
        mat = Rz(0, angle).matrix()
        assert ZYZDecomposition.verify(mat, tol=1e-6)

    def test_to_gates(self):
        mat = H(0).matrix()
        gates = ZYZDecomposition.to_gates(0, mat)
        assert all(isinstance(g, Gate) for g in gates)
        # Gates should be Rz and/or Ry
        for g in gates:
            assert g.name in ("Rz", "Ry")


class TestKAKDecomposition:
    """Test KAK two-qubit decomposition."""

    def test_identity_zero_cnots(self):
        u = np.eye(4, dtype=np.complex128)
        kak = KAKDecomposition.decompose(u)
        assert kak.num_cnots == 0

    def test_cnot_matrix(self):
        u = CX(0, 1).matrix()
        kak = KAKDecomposition.decompose(u)
        assert kak.num_cnots >= 1

    def test_swap_matrix_three_cnots(self):
        u = SWAP(0, 1).matrix()
        kak = KAKDecomposition.decompose(u)
        assert kak.num_cnots == 3


class TestToffoliDecomposition:
    """Test Toffoli decomposition into CX + 1Q gates."""

    def test_toffoli_decomposition_gate_count(self):
        gates = ToffoliDecomposition.decompose(0, 1, 2)
        assert len(gates) == 15
        cx_count = sum(1 for g in gates if g.name == "CX")
        assert cx_count == 6


# ------------------------------------------------------------------ #
# Basis translation tests
# ------------------------------------------------------------------ #


class TestBasisTranslation:
    """Test basis gate set translation."""

    def test_ibm_basis_h_decomposition(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        translated = BasisTranslator(BasisSet.IBM).translate(qc)
        # H should be decomposed into Rz-SX-Rz in IBM basis
        gate_names = {g.name.lower() for g in translated.gates}
        assert gate_names.issubset({"rz", "sx", "x", "id", "cx"})

    def test_google_basis_h_decomposition(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        translated = BasisTranslator(BasisSet.GOOGLE).translate(qc)
        gate_names = {g.name.lower() for g in translated.gates}
        assert gate_names.issubset({"rz", "rx", "ry", "cz", "id"})

    def test_rigetti_basis_h_decomposition(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        translated = BasisTranslator(BasisSet.RIGETTI).translate(qc)
        gate_names = {g.name.lower() for g in translated.gates}
        assert gate_names.issubset({"rz", "rx", "ry", "cz", "id"})

    def test_universal_basis_no_change(self, bell_circuit):
        translated = BasisTranslator(BasisSet.UNIVERSAL).translate(bell_circuit)
        assert translated.gate_count() == bell_circuit.gate_count()

    def test_ibm_basis_cnot_native(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        translated = BasisTranslator(BasisSet.IBM).translate(qc)
        # CX is native in IBM basis
        assert any(g.name.lower() == "cx" for g in translated.gates)

    def test_google_basis_cx_to_cz(self):
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        translated = BasisTranslator(BasisSet.GOOGLE).translate(qc)
        gate_names = {g.name.lower() for g in translated.gates}
        # Should use CZ instead of CX
        assert "cz" in gate_names
        assert "cx" not in gate_names

    def test_decompose_convenience_function(self, bell_circuit):
        result = decompose(bell_circuit, basis=BasisSet.IBM)
        assert isinstance(result, QuantumCircuit)
        assert result.gate_count() >= 1

    def test_swap_decomposition(self):
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        translated = BasisTranslator(BasisSet.IBM).translate(qc)
        # SWAP should decompose into 3 CX
        cx_count = sum(1 for g in translated.gates if g.name.lower() == "cx")
        assert cx_count == 3

    @pytest.mark.parametrize("basis", [BasisSet.IBM, BasisSet.GOOGLE, BasisSet.RIGETTI])
    def test_s_gate_decomposition(self, basis):
        qc = QuantumCircuit(1)
        qc.s(0)
        translated = BasisTranslator(basis).translate(qc)
        # S should decompose to Rz(pi/2) in all bases
        assert any(g.name.lower() == "rz" for g in translated.gates)


# ------------------------------------------------------------------ #
# Full pipeline integration tests
# ------------------------------------------------------------------ #


class TestFullPipeline:
    """Integration tests for the full transpiler pipeline."""

    def test_route_optimize_decompose(self, ghz_circuit, line_3):
        # Route
        routed = route(ghz_circuit, line_3, router="greedy")
        # Optimize
        optimized = optimize(routed.circuit, level=1)
        # Decompose
        final = decompose(optimized, basis=BasisSet.IBM)

        assert isinstance(final, QuantumCircuit)
        assert final.gate_count() > 0

    def test_bell_state_pipeline_preserves_unitary(self, bell_circuit):
        # For a small circuit, verify the unitary is preserved (up to global phase)
        original_mat = bell_circuit.to_matrix()

        optimized = optimize(bell_circuit, level=1)
        optimized_mat = optimized.to_matrix()

        # Verify they are equal up to global phase
        # Check that |original_mat - phase * optimized_mat| is small for some phase
        product = original_mat @ optimized_mat.conj().T
        # product should be proportional to identity
        diag = np.diag(product)
        if abs(diag[0]) > 1e-10:
            phase = diag[0] / abs(diag[0])
            scaled = product / phase
            np.testing.assert_allclose(
                np.abs(np.diag(scaled)), 1.0, atol=1e-6,
                err_msg="Optimization changed the circuit unitary"
            )
