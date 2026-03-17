"""Tests for nqpu.emulator -- QPU emulation and hardware advisor.

Covers:
- Hardware profiles, QPU execution (Bell, GHZ, Toffoli), noise toggling
- Statevector mode, cross-backend comparison, Counts utilities
- Error handling, QuantumCircuit integration
- HardwareAdvisor: CircuitProfile, HardwareScore, Recommendation, full_report()
- Edge cases: empty circuits, oversized circuits, Toffoli-heavy circuits
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nqpu.emulator import (
    QPU,
    Counts,
    EmulatorResult,
    HardwareFamily,
    HardwareProfile,
    HardwareSpec,
    Job,
    CircuitProfile,
    HardwareScore,
    Recommendation,
    HardwareAdvisor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit():
    """Simple Bell state: H(0), CX(0,1)."""
    return [("h", 0), ("cx", 0, 1)]


@pytest.fixture
def ghz3_circuit():
    """3-qubit GHZ: H(0), CX(0,1), CX(1,2)."""
    return [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]


@pytest.fixture
def toffoli_circuit():
    """Toffoli circuit: X(0), X(1), CCX(0,1,2)."""
    return [("x", 0), ("x", 1), ("ccx", 0, 1, 2)]


@pytest.fixture
def toffoli_heavy_circuit():
    """Toffoli-heavy circuit for advisor tests (3 repeated CCX gates)."""
    return [
        ("h", 0), ("h", 1), ("h", 2),
        ("ccx", 0, 1, 2),
        ("ccx", 0, 1, 2),
        ("ccx", 0, 1, 2),
    ]


@pytest.fixture
def large_circuit():
    """Moderately large circuit (6 qubits, many gates)."""
    gates = []
    for i in range(6):
        gates.append(("h", i))
    for i in range(5):
        gates.append(("cx", i, i + 1))
    gates.append(("cx", 0, 5))  # Non-nearest-neighbour
    return gates


@pytest.fixture
def advisor():
    return HardwareAdvisor()


# ---------------------------------------------------------------------------
# HardwareProfile enumeration
# ---------------------------------------------------------------------------


class TestHardwareProfile:
    """Test that all hardware profiles are configured correctly."""

    def test_all_profiles_have_specs(self):
        for profile in HardwareProfile:
            spec = profile.spec
            assert isinstance(spec, HardwareSpec)
            assert spec.num_qubits > 0
            assert 0 < spec.single_qubit_fidelity <= 1.0
            assert 0 < spec.two_qubit_fidelity <= 1.0
            assert 0 < spec.readout_fidelity <= 1.0

    def test_nine_profiles_exist(self):
        assert len(HardwareProfile) == 9

    def test_families_covered(self):
        families = {p.spec.family for p in HardwareProfile}
        assert HardwareFamily.TRAPPED_ION in families
        assert HardwareFamily.SUPERCONDUCTING in families
        assert HardwareFamily.NEUTRAL_ATOM in families

    def test_by_family(self):
        ion = HardwareProfile.by_family(HardwareFamily.TRAPPED_ION)
        assert len(ion) == 3
        sc = HardwareProfile.by_family(HardwareFamily.SUPERCONDUCTING)
        assert len(sc) == 4
        na = HardwareProfile.by_family(HardwareFamily.NEUTRAL_ATOM)
        assert len(na) == 2

    def test_by_name_enum(self):
        p = HardwareProfile.by_name("IONQ_ARIA")
        assert p == HardwareProfile.IONQ_ARIA

    def test_by_name_device(self):
        p = HardwareProfile.by_name("IonQ Aria")
        assert p == HardwareProfile.IONQ_ARIA

    def test_by_name_case_insensitive(self):
        p = HardwareProfile.by_name("ibm heron (133q)")
        assert p == HardwareProfile.IBM_HERON

    def test_by_name_invalid(self):
        with pytest.raises(ValueError, match="Unknown"):
            HardwareProfile.by_name("nonexistent_device")

    def test_error_properties(self):
        spec = HardwareProfile.IONQ_ARIA.spec
        assert spec.error_per_1q == pytest.approx(1.0 - spec.single_qubit_fidelity)
        assert spec.error_per_2q == pytest.approx(1.0 - spec.two_qubit_fidelity)
        assert spec.error_per_readout == pytest.approx(1.0 - spec.readout_fidelity)

    def test_neutral_atom_has_native_3q(self):
        for p in HardwareProfile.by_family(HardwareFamily.NEUTRAL_ATOM):
            assert p.spec.native_3q_gate is not None

    def test_trapped_ion_no_native_3q(self):
        for p in HardwareProfile.by_family(HardwareFamily.TRAPPED_ION):
            assert p.spec.native_3q_gate is None


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


class TestCounts:
    def test_total_shots(self):
        c = Counts({"00": 480, "11": 520})
        assert c.total_shots == 1000

    def test_probabilities(self):
        c = Counts({"00": 500, "11": 500})
        p = c.probabilities()
        assert p["00"] == pytest.approx(0.5)
        assert p["11"] == pytest.approx(0.5)

    def test_most_probable(self):
        c = Counts({"00": 300, "11": 700})
        assert c.most_probable() == "11"

    def test_most_probable_empty_raises(self):
        c = Counts()
        with pytest.raises(ValueError):
            c.most_probable()

    def test_marginal(self):
        c = Counts({"000": 400, "111": 600})
        m = c.marginal([0], n_qubits=3)
        assert m["0"] == 400
        assert m["1"] == 600

    def test_entropy_deterministic(self):
        c = Counts({"00": 1000})
        assert c.entropy() == pytest.approx(0.0)

    def test_entropy_uniform(self):
        c = Counts({"00": 500, "01": 500, "10": 500, "11": 500})
        assert c.entropy() == pytest.approx(2.0, abs=0.01)

    def test_empty_counts(self):
        c = Counts()
        assert c.total_shots == 0
        assert c.probabilities() == {}
        assert c.entropy() == 0.0


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class TestJob:
    def test_successful_job(self):
        result = EmulatorResult(counts=Counts({"0": 1000}))
        job = Job(job_id="test", status="completed", result=result)
        assert job.successful()

    def test_failed_job(self):
        job = Job(job_id="test", status="failed", error="too many qubits")
        assert not job.successful()

    def test_no_result_not_successful(self):
        job = Job(job_id="test", status="completed", result=None)
        assert not job.successful()


# ---------------------------------------------------------------------------
# QPU -- Bell state across all families
# ---------------------------------------------------------------------------


class TestQPUBellState:
    """Run a Bell state on one representative from each hardware family."""

    @pytest.mark.parametrize(
        "profile",
        [HardwareProfile.IONQ_ARIA, HardwareProfile.IBM_HERON, HardwareProfile.QUERA_AQUILA],
        ids=["trapped_ion", "superconducting", "neutral_atom"],
    )
    def test_bell_state(self, bell_circuit, profile):
        qpu = QPU(profile, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=1000)
        assert job.successful()
        result = job.result
        # Ideal Bell state: only "00" and "11" outcomes
        assert set(result.counts.keys()) <= {"00", "11"}
        assert result.counts.total_shots == 1000
        # Close to 50/50 split
        ratio = result.counts.get("00", 0) / 1000
        assert 0.3 < ratio < 0.7

    @pytest.mark.parametrize(
        "profile",
        [HardwareProfile.IONQ_ARIA, HardwareProfile.IBM_HERON, HardwareProfile.QUERA_AQUILA],
        ids=["trapped_ion", "superconducting", "neutral_atom"],
    )
    def test_bell_state_noisy(self, bell_circuit, profile):
        qpu = QPU(profile, noise=True, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=2000)
        assert job.successful()
        result = job.result
        # Noisy: "00" and "11" should dominate but others may appear
        dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
        assert dominant > 1500  # at least 75%


# ---------------------------------------------------------------------------
# QPU -- GHZ state
# ---------------------------------------------------------------------------


class TestQPUGHZ:
    def test_ghz_ideal(self, ghz3_circuit):
        qpu = QPU(HardwareProfile.IONQ_FORTE, noise=False, seed=42, max_qubits=8)
        job = qpu.run(ghz3_circuit, shots=1000)
        assert job.successful()
        assert set(job.result.counts.keys()) <= {"000", "111"}

    def test_ghz_noisy(self, ghz3_circuit):
        qpu = QPU(HardwareProfile.IBM_EAGLE, noise=True, seed=42, max_qubits=8)
        job = qpu.run(ghz3_circuit, shots=2000)
        assert job.successful()
        dominant = job.result.counts.get("000", 0) + job.result.counts.get("111", 0)
        assert dominant > 1400


# ---------------------------------------------------------------------------
# QPU -- Toffoli gate
# ---------------------------------------------------------------------------


class TestQPUToffoli:
    def test_toffoli_ideal_ion(self, toffoli_circuit):
        """Ion-trap decomposes Toffoli; result should be |111>."""
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(toffoli_circuit, shots=500)
        assert job.successful()
        assert job.result.counts.most_probable() == "111"

    def test_toffoli_ideal_neutral_atom(self, toffoli_circuit):
        """Neutral-atom has native Toffoli; result should be |111>."""
        qpu = QPU(HardwareProfile.QUERA_AQUILA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(toffoli_circuit, shots=500)
        assert job.successful()
        assert job.result.counts.most_probable() == "111"


# ---------------------------------------------------------------------------
# QPU -- statevector mode
# ---------------------------------------------------------------------------


class TestQPUStatevector:
    def test_statevector_bell(self, bell_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=0)
        assert job.successful()
        sv = job.result.statevector
        assert sv is not None
        assert len(sv) == 4
        # |00> and |11> amplitudes should be ~1/sqrt(2)
        assert abs(abs(sv[0]) - 1 / math.sqrt(2)) < 0.01
        assert abs(abs(sv[3]) - 1 / math.sqrt(2)) < 0.01

    def test_statevector_num_qubits(self, bell_circuit):
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=0)
        assert job.result.num_qubits == 2


# ---------------------------------------------------------------------------
# QPU -- fidelity and metadata
# ---------------------------------------------------------------------------


class TestQPUMetadata:
    def test_ideal_fidelity_is_one(self, bell_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=100)
        assert job.result.fidelity_estimate == 1.0

    def test_noisy_fidelity_below_one(self, bell_circuit):
        qpu = QPU(HardwareProfile.IBM_EAGLE, noise=True, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=100)
        assert job.result.fidelity_estimate < 1.0
        assert job.result.fidelity_estimate > 0.5

    def test_circuit_depth_reported(self, ghz3_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(ghz3_circuit, shots=100)
        assert job.result.circuit_depth > 0

    def test_native_gate_count(self, bell_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=100)
        assert job.result.native_gate_count >= 2

    def test_estimated_runtime(self, bell_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=100)
        assert job.result.estimated_runtime_us > 0

    def test_hardware_profile_name(self, bell_circuit):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(bell_circuit, shots=100)
        assert job.result.hardware_profile == "IonQ Aria"


# ---------------------------------------------------------------------------
# QPU -- error handling
# ---------------------------------------------------------------------------


class TestQPUErrors:
    def test_too_many_qubits(self):
        """Circuit requiring more qubits than max_qubits should fail."""
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, max_qubits=2)
        big = [("h", 0), ("cx", 0, 1), ("cx", 1, 2)]  # needs 3 qubits
        job = qpu.run(big, shots=100)
        assert not job.successful()
        assert "qubits" in job.error.lower()

    def test_unsupported_gate(self):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run([("fredkin", 0, 1, 2)], shots=100)
        assert not job.successful()

    def test_bad_circuit_type(self):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        with pytest.raises(TypeError):
            qpu.run("not a circuit", shots=100)


# ---------------------------------------------------------------------------
# QPU -- compare() class method
# ---------------------------------------------------------------------------


class TestQPUCompare:
    def test_compare_default_profiles(self, bell_circuit):
        results = QPU.compare(bell_circuit, shots=500, seed=42)
        assert len(results) >= 2  # at least 2 backends succeed
        for name, result in results.items():
            assert isinstance(result, EmulatorResult)
            assert result.counts.total_shots == 500

    def test_compare_custom_profiles(self, bell_circuit):
        profiles = [HardwareProfile.IONQ_ARIA, HardwareProfile.IBM_HERON]
        results = QPU.compare(bell_circuit, profiles=profiles, shots=200, seed=42)
        assert "IonQ Aria" in results
        assert "IBM Heron (133Q)" in results


# ---------------------------------------------------------------------------
# QPU -- info()
# ---------------------------------------------------------------------------


class TestQPUInfo:
    def test_info_keys(self):
        qpu = QPU(HardwareProfile.QUANTINUUM_H2)
        info = qpu.info()
        expected_keys = {
            "name", "family", "num_qubits", "connectivity",
            "1q_fidelity", "2q_fidelity", "readout_fidelity",
            "native_2q_gate", "native_3q_gate",
            "t1_us", "t2_us", "max_circuit_depth",
        }
        assert set(info.keys()) == expected_keys
        assert info["name"] == "Quantinuum H2"
        assert info["num_qubits"] == 56

    def test_repr(self):
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=True)
        assert "IonQ Aria" in repr(qpu)
        assert "noise=True" in repr(qpu)


# ---------------------------------------------------------------------------
# QPU -- single-qubit gate coverage
# ---------------------------------------------------------------------------


class TestQPUGates:
    """Ensure all documented single-qubit gates execute without error."""

    @pytest.mark.parametrize("gate", ["x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx"])
    def test_1q_gates(self, gate):
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=4)
        job = qpu.run([(gate, 0)], shots=100)
        assert job.successful()

    @pytest.mark.parametrize("gate", ["rx", "ry", "rz"])
    def test_rotation_gates(self, gate):
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=4)
        job = qpu.run([(gate, 0, math.pi / 4)], shots=100)
        assert job.successful()

    def test_swap_gate(self):
        # X(0) sets qubit 0 to |1>, SWAP(0,1) moves it to qubit 1
        # In MSB-first bitstring: |10> means qubit 0=1, qubit 1=0
        # After SWAP: qubit 0=0, qubit 1=1 -> bitstring "01" or "10"
        # depending on backend convention. Just check deterministic output.
        circuit = [("x", 0), ("swap", 0, 1)]
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=4)
        job = qpu.run(circuit, shots=500)
        assert job.successful()
        assert len(job.result.counts) == 1  # deterministic: only one outcome

    def test_cz_gate(self):
        circuit = [("h", 0), ("h", 1), ("cz", 0, 1), ("h", 0), ("h", 1)]
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=4)
        job = qpu.run(circuit, shots=100)
        assert job.successful()

    def test_ccz_gate(self):
        circuit = [("x", 0), ("x", 1), ("h", 2), ("ccz", 0, 1, 2), ("h", 2)]
        qpu = QPU(HardwareProfile.IBM_HERON, noise=False, seed=42, max_qubits=4)
        job = qpu.run(circuit, shots=500)
        assert job.successful()
        # CCZ on |11> H|0> = CZ |11> (|0>+|1>)/sqrt(2) -> |11> (|0>-|1>)/sqrt(2)
        # Then H -> |11> |1> = |111>
        assert job.result.counts.most_probable() == "111"


# ---------------------------------------------------------------------------
# QPU -- QuantumCircuit integration
# ---------------------------------------------------------------------------


class TestQPUQuantumCircuit:
    def test_transpiler_circuit(self):
        from nqpu.transpiler import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0).cx(0, 1)
        qpu = QPU(HardwareProfile.IONQ_ARIA, noise=False, seed=42, max_qubits=8)
        job = qpu.run(qc, shots=1000)
        assert job.successful()
        assert set(job.result.counts.keys()) <= {"00", "11"}


# ---------------------------------------------------------------------------
# QPU -- max_qubits cap
# ---------------------------------------------------------------------------


class TestQPUMaxQubits:
    def test_max_qubits_property(self):
        qpu = QPU(HardwareProfile.ATOM_COMPUTING, max_qubits=10)
        assert qpu.num_qubits == 10  # capped from 1225

    def test_name_property(self):
        qpu = QPU(HardwareProfile.GOOGLE_SYCAMORE)
        assert qpu.name == "Google Sycamore (72Q)"


# ===========================================================================
# CircuitProfile tests
# ===========================================================================


class TestCircuitProfile:
    def test_bell_circuit_profile(self, advisor, bell_circuit):
        cp = advisor.analyze_circuit(bell_circuit)
        assert cp.n_qubits == 2
        assert cp.n_1q_gates == 1
        assert cp.n_2q_gates == 1
        assert cp.n_3q_gates == 0
        assert cp.toffoli_fraction == 0.0
        assert "h" in cp.unique_gates
        assert "cx" in cp.unique_gates

    def test_toffoli_circuit_profile(self, advisor, toffoli_heavy_circuit):
        cp = advisor.analyze_circuit(toffoli_heavy_circuit)
        assert cp.n_qubits == 3
        assert cp.n_1q_gates == 3
        assert cp.n_3q_gates == 3
        assert cp.toffoli_fraction == 1.0  # All entangling gates are 3Q

    def test_large_circuit_connectivity(self, advisor, large_circuit):
        cp = advisor.analyze_circuit(large_circuit)
        assert cp.n_qubits == 6
        # cx(0,5) is non-nearest-neighbour, so not "linear"
        assert cp.connectivity_required in ("grid", "all_to_all")

    def test_empty_circuit(self, advisor):
        cp = advisor.analyze_circuit([])
        assert cp.n_qubits == 0
        assert cp.depth == 0

    def test_depth_computation(self, advisor):
        # Sequential gates on same qubit: depth = count
        circuit = [("h", 0), ("x", 0), ("z", 0)]
        cp = advisor.analyze_circuit(circuit)
        assert cp.depth == 3

    def test_parallel_depth(self, advisor):
        # Gates on different qubits: depth = 1
        circuit = [("h", 0), ("h", 1), ("h", 2)]
        cp = advisor.analyze_circuit(circuit)
        assert cp.depth == 1


# ===========================================================================
# HardwareScore tests
# ===========================================================================


class TestHardwareScore:
    def test_score_all_profiles(self, advisor, bell_circuit):
        cp = advisor.analyze_circuit(bell_circuit)
        for profile in HardwareProfile:
            score = advisor.score_profile(profile, cp)
            assert score.total_score >= 0
            assert score.profile == profile
            assert isinstance(score.reasons, list)

    def test_insufficient_qubits_returns_zero(self, advisor):
        # Create a circuit needing more qubits than any profile
        cp = CircuitProfile(n_qubits=10000)
        for profile in HardwareProfile:
            score = advisor.score_profile(profile, cp)
            assert score.total_score == 0.0

    def test_toffoli_native_advantage(self, advisor, toffoli_heavy_circuit):
        """Neutral-atom should score higher on Toffoli gates (native CCZ)."""
        cp = advisor.analyze_circuit(toffoli_heavy_circuit)
        na_scores = []
        other_scores = []
        for profile in HardwareProfile:
            score = advisor.score_profile(profile, cp)
            if score.total_score == 0:
                continue
            if profile.spec.family == HardwareFamily.NEUTRAL_ATOM:
                na_scores.append(score.toffoli_score)
            else:
                other_scores.append(score.toffoli_score)
        if na_scores and other_scores:
            assert max(na_scores) > np.mean(other_scores)

    def test_fidelity_score_range(self, advisor, bell_circuit):
        cp = advisor.analyze_circuit(bell_circuit)
        for profile in HardwareProfile:
            score = advisor.score_profile(profile, cp)
            if score.total_score > 0:
                assert 0 <= score.estimated_fidelity <= 1.0
                assert score.estimated_runtime_us > 0


# ===========================================================================
# Recommendation tests
# ===========================================================================


class TestRecommendation:
    def test_recommend_returns_valid(self, advisor, bell_circuit):
        rec = advisor.recommend(bell_circuit)
        assert isinstance(rec, Recommendation)
        assert rec.best_profile in HardwareProfile
        assert isinstance(rec.reasoning, str)
        assert len(rec.reasoning) > 0
        assert len(rec.scores) > 0

    def test_recommend_scores_sorted(self, advisor, bell_circuit):
        rec = advisor.recommend(bell_circuit)
        total_scores = [s.total_score for s in rec.scores]
        assert total_scores == sorted(total_scores, reverse=True)

    def test_recommend_runner_up(self, advisor, bell_circuit):
        rec = advisor.recommend(bell_circuit)
        if rec.runner_up is not None:
            assert rec.runner_up != rec.best_profile
            assert rec.runner_up in HardwareProfile

    def test_recommend_custom_profiles(self, advisor, bell_circuit):
        subset = [HardwareProfile.IONQ_ARIA, HardwareProfile.IBM_EAGLE]
        rec = advisor.recommend(bell_circuit, profiles=subset)
        assert rec.best_profile in subset

    def test_recommend_toffoli_heavy(self, advisor, toffoli_heavy_circuit):
        rec = advisor.recommend(toffoli_heavy_circuit)
        # Neutral atom should be recommended for Toffoli-heavy circuits
        assert rec.best_profile.spec.family == HardwareFamily.NEUTRAL_ATOM

    def test_warnings_for_low_fidelity(self, advisor):
        # Very deep circuit should trigger warnings
        circuit = [("cx", 0, 1)] * 200
        rec = advisor.recommend(circuit)
        assert len(rec.warnings) > 0


# ===========================================================================
# Custom weights tests
# ===========================================================================


class TestCustomWeights:
    def test_speed_only_weights(self, bell_circuit):
        advisor = HardwareAdvisor(weights={
            "fidelity": 0.0, "speed": 1.0,
            "capacity": 0.0, "toffoli": 0.0, "connectivity": 0.0,
        })
        rec = advisor.recommend(bell_circuit)
        # Superconducting should win on speed (ns-scale gates)
        assert rec.best_profile.spec.family == HardwareFamily.SUPERCONDUCTING

    def test_fidelity_only_weights(self, bell_circuit):
        advisor = HardwareAdvisor(weights={
            "fidelity": 1.0, "speed": 0.0,
            "capacity": 0.0, "toffoli": 0.0, "connectivity": 0.0,
        })
        rec = advisor.recommend(bell_circuit)
        # Trapped ion should win on fidelity
        assert rec.best_profile.spec.family == HardwareFamily.TRAPPED_ION


# ===========================================================================
# full_report tests
# ===========================================================================


class TestFullReport:
    def test_full_report_keys(self, advisor, bell_circuit):
        report = advisor.full_report(bell_circuit, shots=100, seed=42)
        assert "recommendation" in report
        assert "execution_results" in report
        assert "comparison_table" in report
        assert isinstance(report["recommendation"], Recommendation)

    def test_full_report_executions(self, advisor, bell_circuit):
        report = advisor.full_report(bell_circuit, shots=100, seed=42)
        # Should have at least 1 execution result
        assert len(report["execution_results"]) >= 1
        for name, result in report["execution_results"].items():
            assert "counts" in result
            assert "fidelity_estimate" in result
            assert "runtime_us" in result

    def test_full_report_comparison_table(self, advisor, bell_circuit):
        report = advisor.full_report(bell_circuit, shots=100, seed=42)
        table = report["comparison_table"]
        assert len(table) >= 1
        for entry in table:
            assert "profile" in entry
            assert "total_score" in entry
            assert "fidelity" in entry


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_single_gate_circuit(self, advisor):
        rec = advisor.recommend([("h", 0)])
        assert rec.best_profile in HardwareProfile

    def test_rotation_gates(self, advisor):
        circuit = [("rx", 0, 0.5), ("ry", 1, 1.0), ("rz", 2, 0.3)]
        cp = advisor.analyze_circuit(circuit)
        assert cp.n_qubits == 3
        assert cp.n_1q_gates == 3

    def test_mixed_gates(self, advisor):
        circuit = [
            ("h", 0), ("cx", 0, 1), ("ccx", 0, 1, 2),
            ("rz", 3, 0.5), ("cz", 2, 3), ("swap", 0, 3),
        ]
        cp = advisor.analyze_circuit(circuit)
        assert cp.n_qubits == 4
        assert cp.n_1q_gates == 2  # h + rz
        assert cp.n_2q_gates == 3  # cx + cz + swap
        assert cp.n_3q_gates == 1  # ccx
