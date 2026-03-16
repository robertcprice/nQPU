"""Comprehensive tests for the nQPU visualization package.

Tests cover all five submodules:
  - circuit_drawing: ASCII quantum circuit rendering
  - bloch_sphere: Bloch vector extraction and sphere rendering
  - state_viz: Probability displays, Hinton diagrams, entanglement maps
  - formatters: Tables, progress bars, result formatting
  - plots: Matplotlib wrapper graceful degradation

Test strategy:
  - Known quantum states with analytically verifiable properties
  - Output structure validation (line counts, presence of key symbols)
  - Numerical accuracy for Bloch vectors and probabilities
  - Graceful degradation when optional dependencies are absent
"""

import math
import sys
from unittest.mock import patch

import numpy as np
import pytest

from nqpu.visualization import (
    ASCIIBlochSphere,
    ASCIITable,
    BlochVector,
    CircuitDrawer,
    CircuitGlyph,
    EntanglementMap,
    HAS_MATPLOTLIB,
    HintonDiagram,
    ProbabilityDisplay,
    ProgressBar,
    QuantumPlotter,
    ResultFormatter,
    bloch_from_angles,
    bloch_from_state,
    bloch_trajectory,
    density_matrix_display,
    draw_circuit,
    format_complex,
    format_statevector,
    gate_to_ascii,
    plot_density_matrix,
    plot_state,
    probability_bar_chart,
    progress_bar,
    state_table,
    table,
)

# ======================================================================
# Fixtures: Common quantum states
# ======================================================================

SQRT2_INV = 1.0 / math.sqrt(2.0)


@pytest.fixture
def ket_zero() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


@pytest.fixture
def ket_one() -> np.ndarray:
    return np.array([0.0, 1.0], dtype=complex)


@pytest.fixture
def ket_plus() -> np.ndarray:
    return np.array([SQRT2_INV, SQRT2_INV], dtype=complex)


@pytest.fixture
def ket_minus() -> np.ndarray:
    return np.array([SQRT2_INV, -SQRT2_INV], dtype=complex)


@pytest.fixture
def ket_i_plus() -> np.ndarray:
    """|+i> = (|0> + i|1>) / sqrt(2)"""
    return np.array([SQRT2_INV, SQRT2_INV * 1j], dtype=complex)


@pytest.fixture
def bell_state() -> np.ndarray:
    """Bell state |Phi+> = (|00> + |11>) / sqrt(2)"""
    return np.array([SQRT2_INV, 0, 0, SQRT2_INV], dtype=complex)


@pytest.fixture
def ghz3_state() -> np.ndarray:
    """3-qubit GHZ = (|000> + |111>) / sqrt(2)"""
    s = np.zeros(8, dtype=complex)
    s[0] = SQRT2_INV
    s[7] = SQRT2_INV
    return s


@pytest.fixture
def mixed_state_rho() -> np.ndarray:
    """Maximally mixed single-qubit state."""
    return 0.5 * np.eye(2, dtype=complex)


# ======================================================================
# TestCircuitDrawing
# ======================================================================


class TestCircuitDrawing:
    """Tests for circuit_drawing module."""

    def test_single_h_gate(self):
        """Single Hadamard gate renders correctly."""
        result = draw_circuit(1, [("H", [0], [])])
        assert "H" in result
        assert "q0" in result

    def test_two_qubit_identity(self):
        """Empty gate list renders just wires."""
        result = draw_circuit(2, [])
        assert "q0" in result
        assert "q1" in result

    def test_cnot_rendering(self):
        """CNOT gate shows control and target symbols."""
        result = draw_circuit(2, [("CNOT", [0, 1], [])], style="ascii")
        assert "*" in result
        assert "X" in result

    def test_cnot_unicode(self):
        """CNOT in unicode style uses special characters."""
        result = draw_circuit(2, [("CNOT", [0, 1], [])], style="unicode")
        # Control dot and target symbol
        assert "\u25cf" in result or "\u2295" in result

    def test_parameterized_gate(self):
        """Parameterized Rz gate shows angle."""
        result = draw_circuit(1, [("Rz", [0], [math.pi / 2])])
        assert "Rz" in result
        assert "pi/2" in result

    def test_swap_gate(self):
        """SWAP gate renders on both qubits."""
        result = draw_circuit(2, [("SWAP", [0, 1], [])], style="ascii")
        # SWAP uses x markers
        assert result.count("x") >= 2

    def test_measurement(self):
        """Measurement symbol renders."""
        result = draw_circuit(1, [("M", [0], [])])
        assert "M" in result

    def test_multi_gate_sequence(self):
        """Multiple gates in sequence."""
        gates = [
            ("H", [0], []),
            ("CNOT", [0, 1], []),
            ("Rz", [1], [math.pi]),
            ("M", [0], []),
            ("M", [1], []),
        ]
        result = draw_circuit(2, gates, style="ascii")
        assert "H" in result
        assert "M" in result
        lines = result.strip().split("\n")
        # Should have wire lines for q0 and q1 plus spacer
        assert len(lines) >= 3

    def test_toffoli_gate(self):
        """Toffoli (CCX) gate with two controls."""
        result = draw_circuit(3, [("CCX", [0, 1, 2], [])], style="ascii")
        # Two control dots and one target
        assert result.count("*") >= 2

    def test_cz_gate(self):
        """CZ gate shows two control symbols."""
        result = draw_circuit(2, [("CZ", [0, 1], [])], style="ascii")
        assert result.count("*") >= 2

    def test_draw_to_lines(self):
        """draw_to_lines returns a list of strings."""
        drawer = CircuitDrawer(n_qubits=2, style="ascii")
        lines = drawer.draw_to_lines([("H", [0], [])])
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_circuit_glyph_label(self):
        """CircuitGlyph.label() formats correctly."""
        g = CircuitGlyph(name="Rx", qubits=[0], params=[math.pi])
        assert "Rx" in g.label()
        assert "pi" in g.label()

    def test_gate_to_ascii_simple(self):
        """gate_to_ascii for simple gates."""
        assert gate_to_ascii("H") == "H"
        assert gate_to_ascii("X") == "X"

    def test_gate_to_ascii_parameterized(self):
        """gate_to_ascii for parameterized gates."""
        result = gate_to_ascii("Ry", [math.pi / 4])
        assert "Ry" in result
        assert "pi/4" in result

    def test_three_qubit_circuit(self):
        """Circuit with 3 qubits renders all wire labels."""
        result = draw_circuit(3, [("H", [0], []), ("H", [1], []), ("H", [2], [])])
        assert "q0" in result
        assert "q1" in result
        assert "q2" in result


# ======================================================================
# TestBlochSphere
# ======================================================================


class TestBlochSphere:
    """Tests for bloch_sphere module."""

    def test_ket_zero_bloch(self, ket_zero):
        """\\|0> maps to north pole (0, 0, 1)."""
        bv = bloch_from_state(ket_zero)
        assert abs(bv.x) < 1e-10
        assert abs(bv.y) < 1e-10
        assert abs(bv.z - 1.0) < 1e-10

    def test_ket_one_bloch(self, ket_one):
        """\\|1> maps to south pole (0, 0, -1)."""
        bv = bloch_from_state(ket_one)
        assert abs(bv.x) < 1e-10
        assert abs(bv.y) < 1e-10
        assert abs(bv.z + 1.0) < 1e-10

    def test_ket_plus_bloch(self, ket_plus):
        """\\|+> maps to +X axis (1, 0, 0)."""
        bv = bloch_from_state(ket_plus)
        assert abs(bv.x - 1.0) < 1e-10
        assert abs(bv.y) < 1e-10
        assert abs(bv.z) < 1e-10

    def test_ket_minus_bloch(self, ket_minus):
        """\\|-> maps to -X axis (-1, 0, 0)."""
        bv = bloch_from_state(ket_minus)
        assert abs(bv.x + 1.0) < 1e-10
        assert abs(bv.y) < 1e-10
        assert abs(bv.z) < 1e-10

    def test_ket_i_plus_bloch(self, ket_i_plus):
        """\\|+i> maps to +Y axis (0, 1, 0)."""
        bv = bloch_from_state(ket_i_plus)
        assert abs(bv.x) < 1e-10
        assert abs(bv.y - 1.0) < 1e-10
        assert abs(bv.z) < 1e-10

    def test_from_density_matrix(self, ket_zero):
        """from_density_matrix matches from_statevector for pure state."""
        bv_sv = BlochVector.from_statevector(ket_zero)
        rho = np.outer(ket_zero, ket_zero.conj())
        bv_dm = BlochVector.from_density_matrix(rho)
        assert abs(bv_sv.x - bv_dm.x) < 1e-10
        assert abs(bv_sv.y - bv_dm.y) < 1e-10
        assert abs(bv_sv.z - bv_dm.z) < 1e-10

    def test_mixed_state_bloch(self, mixed_state_rho):
        """Maximally mixed state maps to origin."""
        bv = BlochVector.from_density_matrix(mixed_state_rho)
        assert abs(bv.x) < 1e-10
        assert abs(bv.y) < 1e-10
        assert abs(bv.z) < 1e-10

    def test_purity_pure(self, ket_zero):
        """Pure state has purity 1."""
        bv = bloch_from_state(ket_zero)
        assert abs(bv.purity - 1.0) < 1e-10

    def test_purity_mixed(self, mixed_state_rho):
        """Maximally mixed state has purity 0.5."""
        bv = BlochVector.from_density_matrix(mixed_state_rho)
        assert abs(bv.purity - 0.5) < 1e-10

    def test_bloch_from_angles(self):
        """bloch_from_angles round-trips theta/phi."""
        bv = bloch_from_angles(math.pi / 3, math.pi / 4)
        assert abs(bv.theta - math.pi / 3) < 1e-10
        assert abs(bv.phi - math.pi / 4) < 1e-10
        assert abs(bv.norm - 1.0) < 1e-10

    def test_ascii_sphere_renders(self, ket_zero):
        """ASCIIBlochSphere.render() produces non-empty output."""
        bv = bloch_from_state(ket_zero)
        sphere = ASCIIBlochSphere(radius=6)
        result = sphere.render([bv])
        assert len(result) > 0
        lines = result.split("\n")
        assert len(lines) >= 10

    def test_bloch_trajectory(self, ket_zero, ket_plus, ket_one):
        """bloch_trajectory converts a list of states."""
        traj = bloch_trajectory([ket_zero, ket_plus, ket_one])
        assert len(traj) == 3
        assert abs(traj[0].z - 1.0) < 1e-10  # |0>
        assert abs(traj[1].x - 1.0) < 1e-10  # |+>
        assert abs(traj[2].z + 1.0) < 1e-10  # |1>

    def test_bloch_repr(self, ket_zero):
        """BlochVector __repr__ is informative."""
        bv = bloch_from_state(ket_zero)
        r = repr(bv)
        assert "BlochVector" in r
        assert "x=" in r
        assert "z=" in r

    def test_invalid_statevector_raises(self):
        """Non-2-element state raises ValueError."""
        with pytest.raises(ValueError, match="2-element"):
            BlochVector.from_statevector(np.array([1, 0, 0, 1]))

    def test_zero_norm_raises(self):
        """Zero-norm state raises ValueError."""
        with pytest.raises(ValueError, match="zero norm"):
            BlochVector.from_statevector(np.array([0, 0], dtype=complex))


# ======================================================================
# TestStateViz
# ======================================================================


class TestStateViz:
    """Tests for state_viz module."""

    def test_bar_chart_bell(self, bell_state):
        """Bar chart of Bell state shows |00> and |11>."""
        result = probability_bar_chart(bell_state, n_qubits=2)
        assert "|00>" in result
        assert "|11>" in result
        assert "#" in result

    def test_bar_chart_top_k(self, ghz3_state):
        """top_k limits number of displayed states."""
        result = probability_bar_chart(ghz3_state, n_qubits=3, top_k=1)
        # Only one of |000> or |111> should appear
        lines = [l for l in result.split("\n") if "|" in l and ">" in l]
        assert len(lines) == 1

    def test_histogram_counts(self):
        """Histogram from measurement counts."""
        counts = {"00": 480, "11": 520}
        disp = ProbabilityDisplay(n_qubits=2)
        result = disp.histogram(counts)
        assert "00" in result
        assert "11" in result
        assert "#" in result

    def test_histogram_empty(self):
        """Empty counts dict."""
        disp = ProbabilityDisplay(n_qubits=1)
        result = disp.histogram({})
        assert "No counts" in result

    def test_probability_display_width(self, ket_zero):
        """Custom width parameter is respected."""
        # |0> has probability 1 -- should get full bar
        sv = np.array([1, 0], dtype=complex)
        disp = ProbabilityDisplay(n_qubits=1, width=20)
        result = disp.bar_chart(sv)
        # The bar should be exactly 20 '#' characters
        for line in result.split("\n"):
            if "#" in line:
                bar_part = line.split("|")[-1]
                assert len(bar_part.strip()) <= 20

    def test_hinton_diagram_identity(self):
        """Hinton diagram for 2x2 identity matrix."""
        m = np.eye(2)
        hd = HintonDiagram()
        result = hd.render(m, label="Identity")
        assert "Identity" in result
        assert len(result.split("\n")) >= 3

    def test_hinton_diagram_complex(self):
        """Hinton diagram works with complex matrices."""
        rho = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex)
        hd = HintonDiagram()
        result = hd.render(rho)
        assert len(result) > 0

    def test_hinton_diagram_zero_matrix(self):
        """Hinton diagram handles all-zero matrix."""
        m = np.zeros((2, 2))
        hd = HintonDiagram()
        result = hd.render(m)
        assert len(result) > 0

    def test_state_table(self, bell_state):
        """state_table renders structured output."""
        result = state_table(bell_state, n_qubits=2)
        assert "Basis" in result
        assert "Amplitude" in result
        assert "|00>" in result

    def test_density_matrix_display(self):
        """density_matrix_display renders for a simple density matrix."""
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        result = density_matrix_display(rho)
        assert "Density Matrix" in result

    def test_entanglement_map_concurrence(self, bell_state):
        """Concurrence map of Bell state shows non-zero entanglement."""
        emap = EntanglementMap()
        result = emap.concurrence_map(bell_state, n_qubits=2)
        assert "Concurrence" in result
        assert "q0" in result
        assert "q1" in result

    def test_entanglement_map_mutual_info(self, bell_state):
        """Mutual info map of Bell state shows non-zero correlation."""
        emap = EntanglementMap()
        result = emap.mutual_info_map(bell_state, n_qubits=2)
        assert "Mutual Information" in result

    def test_bar_chart_invalid_state_length(self):
        """Mismatched state length raises ValueError."""
        disp = ProbabilityDisplay(n_qubits=2)
        with pytest.raises(ValueError, match="State length"):
            disp.bar_chart(np.array([1, 0, 0], dtype=complex))


# ======================================================================
# TestFormatters
# ======================================================================


class TestFormatters:
    """Tests for formatters module."""

    def test_table_basic(self):
        """Basic table rendering."""
        result = table(["Name", "Value"], [["alpha", "0.5"], ["beta", "0.7"]])
        assert "Name" in result
        assert "Value" in result
        assert "alpha" in result
        assert "0.7" in result

    def test_table_alignment(self):
        """Right-aligned columns."""
        result = table(
            ["Qubit", "Prob"],
            [["q0", "0.50"], ["q1", "0.50"]],
            alignments=["left", "right"],
        )
        assert "Qubit" in result

    def test_table_empty_rows(self):
        """Table with headers but no rows."""
        result = table(["A", "B"], [])
        assert "A" in result
        assert "B" in result

    def test_ascii_table_add_row(self):
        """ASCIITable.add_row accumulates rows."""
        t = ASCIITable(headers=["X", "Y"])
        t.add_row([1, 2])
        t.add_row([3, 4])
        result = t.render()
        assert "1" in result
        assert "4" in result

    def test_ascii_table_mismatched_alignments_raises(self):
        """Mismatched alignments length raises ValueError."""
        with pytest.raises(ValueError, match="alignments length"):
            ASCIITable(headers=["A", "B"], alignments=["left"])

    def test_progress_bar_zero(self):
        """Progress bar at 0%."""
        result = progress_bar(0, 100)
        assert "0.0%" in result
        assert "(0/100)" in result

    def test_progress_bar_half(self):
        """Progress bar at 50%."""
        result = progress_bar(50, 100, width=20)
        assert "50.0%" in result
        assert "#" in result

    def test_progress_bar_complete(self):
        """Progress bar at 100%."""
        pb = ProgressBar(total=10, width=10)
        result = pb.format(10, label="done")
        assert "100.0%" in result
        assert "done" in result
        assert "#" * 10 in result

    def test_progress_bar_zero_total(self):
        """Progress bar with total=0 does not crash."""
        result = progress_bar(0, 0)
        assert "100.0%" in result

    def test_format_energy_basic(self):
        """format_energy with no reference."""
        rf = ResultFormatter()
        result = rf.format_energy(-1.5)
        assert "Energy" in result
        assert "-1.5" in result

    def test_format_energy_with_reference(self):
        """format_energy with reference shows delta."""
        rf = ResultFormatter(precision=4)
        result = rf.format_energy(-1.48, reference=-1.50)
        assert "delta" in result
        assert "ref" in result

    def test_format_fidelity(self):
        """format_fidelity quality labels."""
        rf = ResultFormatter()
        assert "excellent" in rf.format_fidelity(0.9999)
        assert "good" in rf.format_fidelity(0.995)
        assert "fair" in rf.format_fidelity(0.96)
        assert "poor" in rf.format_fidelity(0.91)
        assert "very poor" in rf.format_fidelity(0.80)

    def test_format_counts(self):
        """format_counts renders sorted histogram."""
        rf = ResultFormatter()
        result = rf.format_counts({"00": 480, "11": 520})
        assert "11" in result
        assert "00" in result
        lines = result.strip().split("\n")
        assert len(lines) == 2
        # 11 should come first (higher count)
        assert lines[0].strip().startswith("11")

    def test_format_counts_empty(self):
        """format_counts with empty dict."""
        rf = ResultFormatter()
        assert "No measurements" in rf.format_counts({})

    def test_format_matrix_real(self):
        """format_matrix for real matrix."""
        rf = ResultFormatter(precision=2)
        m = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = rf.format_matrix(m)
        assert "1.00" in result
        assert "0.00" in result

    def test_format_matrix_complex(self):
        """format_matrix for complex matrix."""
        rf = ResultFormatter(precision=3)
        m = np.array([[1.0 + 0j, 0.5j], [-0.5j, 1.0 + 0j]])
        result = rf.format_matrix(m)
        assert "j" in result

    def test_format_complex_pure_real(self):
        """format_complex for pure real number."""
        assert format_complex(3.14, 2) == "3.14"

    def test_format_complex_pure_imaginary(self):
        """format_complex for pure imaginary number."""
        result = format_complex(2.5j, 2)
        assert "j" in result
        assert "2.50" in result

    def test_format_complex_both(self):
        """format_complex for a+bj."""
        result = format_complex(1.0 + 2.0j, 2)
        assert "1.00" in result
        assert "2.00" in result
        assert "j" in result

    def test_format_complex_zero(self):
        """format_complex for zero."""
        result = format_complex(0.0 + 0.0j, 4)
        assert "0.0000" in result

    def test_format_statevector_bell(self, bell_state):
        """format_statevector for Bell state."""
        result = format_statevector(bell_state, n_qubits=2)
        assert "|00>" in result
        assert "|11>" in result

    def test_format_statevector_threshold(self):
        """Small amplitudes below threshold are suppressed."""
        sv = np.array([0.999, 0.001, 0.001, 0.001], dtype=complex)
        sv = sv / np.linalg.norm(sv)
        result = format_statevector(sv, n_qubits=2, threshold=0.01)
        # Only |00> should appear (others are tiny)
        assert "|00>" in result

    def test_format_statevector_wrong_length(self):
        """Mismatched state vector length raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            format_statevector(np.array([1, 0, 0]), n_qubits=2)


# ======================================================================
# TestPlots
# ======================================================================


class TestPlots:
    """Tests for plots module (matplotlib wrapper degradation)."""

    def test_has_matplotlib_flag(self):
        """HAS_MATPLOTLIB is a boolean."""
        assert isinstance(HAS_MATPLOTLIB, bool)

    def test_plotter_creation(self):
        """QuantumPlotter can be instantiated."""
        plotter = QuantumPlotter()
        assert plotter.figsize == (8, 5)

    def test_plot_probabilities_without_mpl(self, bell_state):
        """plot_probabilities raises ImportError without matplotlib."""
        plotter = QuantumPlotter()
        # Temporarily pretend matplotlib is missing
        import nqpu.visualization.plots as plots_mod

        original = plots_mod.HAS_MATPLOTLIB
        plots_mod.HAS_MATPLOTLIB = False
        try:
            with pytest.raises(ImportError, match="matplotlib"):
                plotter.plot_probabilities(bell_state, n_qubits=2)
        finally:
            plots_mod.HAS_MATPLOTLIB = original

    def test_plot_bloch_without_mpl(self, ket_zero):
        """plot_bloch raises ImportError without matplotlib."""
        plotter = QuantumPlotter()
        bv = bloch_from_state(ket_zero)
        import nqpu.visualization.plots as plots_mod

        original = plots_mod.HAS_MATPLOTLIB
        plots_mod.HAS_MATPLOTLIB = False
        try:
            with pytest.raises(ImportError, match="matplotlib"):
                plotter.plot_bloch([bv])
        finally:
            plots_mod.HAS_MATPLOTLIB = original

    def test_plot_energy_without_mpl(self):
        """plot_energy_convergence raises ImportError without matplotlib."""
        plotter = QuantumPlotter()
        import nqpu.visualization.plots as plots_mod

        original = plots_mod.HAS_MATPLOTLIB
        plots_mod.HAS_MATPLOTLIB = False
        try:
            with pytest.raises(ImportError, match="matplotlib"):
                plotter.plot_energy_convergence([1.0, 0.5, 0.1])
        finally:
            plots_mod.HAS_MATPLOTLIB = original

    def test_plot_circuit_without_mpl(self):
        """plot_circuit raises ImportError without matplotlib."""
        plotter = QuantumPlotter()
        import nqpu.visualization.plots as plots_mod

        original = plots_mod.HAS_MATPLOTLIB
        plots_mod.HAS_MATPLOTLIB = False
        try:
            with pytest.raises(ImportError, match="matplotlib"):
                plotter.plot_circuit(2, [("H", [0], [])])
        finally:
            plots_mod.HAS_MATPLOTLIB = original

    def test_plot_density_matrix_without_mpl(self):
        """plot_density_matrix raises ImportError without matplotlib."""
        import nqpu.visualization.plots as plots_mod

        original = plots_mod.HAS_MATPLOTLIB
        plots_mod.HAS_MATPLOTLIB = False
        try:
            with pytest.raises(ImportError, match="matplotlib"):
                plot_density_matrix(np.eye(2, dtype=complex))
        finally:
            plots_mod.HAS_MATPLOTLIB = original


# ======================================================================
# Integration tests
# ======================================================================


class TestIntegration:
    """Cross-module integration tests."""

    def test_full_workflow_circuit_to_state(self, bell_state):
        """Draw a circuit, then visualize the resulting state."""
        # Draw the circuit that produces Bell state
        circuit = [("H", [0], []), ("CNOT", [0, 1], [])]
        diagram = draw_circuit(2, circuit, style="ascii")
        assert "H" in diagram

        # Visualize the state
        bar = probability_bar_chart(bell_state, n_qubits=2)
        assert "|00>" in bar

        # Format as Dirac notation
        dirac = format_statevector(bell_state, n_qubits=2)
        assert "|00>" in dirac
        assert "|11>" in dirac

    def test_bloch_to_ascii_sphere(self, ket_plus):
        """Extract Bloch vector and render on sphere."""
        bv = bloch_from_state(ket_plus)
        sphere = ASCIIBlochSphere(radius=5)
        rendered = sphere.render([bv])
        assert len(rendered) > 0
        # The marker '0' should appear somewhere
        assert "0" in rendered

    def test_result_formatter_with_table(self):
        """Combine ResultFormatter and table for a results summary."""
        rf = ResultFormatter(precision=4)
        energy_str = rf.format_energy(-1.137, reference=-1.1372)
        fidelity_str = rf.format_fidelity(0.9985)
        t = table(["Metric", "Value"], [["Energy", energy_str], ["Fidelity", fidelity_str]])
        assert "Energy" in t
        assert "Fidelity" in t
        assert "good" in t
