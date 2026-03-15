import json
import re
from dataclasses import replace
import numpy as np
import pytest

from nqpu import (
    AdaptiveDQPTDiagnosticsResult,
    DQPTCandidate,
    DQPTDiagnosticsResult,
    DQPTScanResult,
    DynamicStructureFactorResult,
    EntanglementSpectrumResult,
    FrequencyStructureFactorResult,
    GroundStateResult,
    LoschmidtEchoResult,
    ModelQPU,
    NQPUBackend,
    QuantumCircuit,
    ResponseSpectrumResult,
    TransverseFieldIsing1D,
    TwoTimeCorrelatorResult,
    fourier_transform_structure_factor,
    load_dynamic_structure_factor_result,
    load_dqpt_diagnostics_result,
    load_dqpt_scan_result,
    load_entanglement_spectrum_result,
    load_frequency_structure_factor_result,
    load_ground_state_result,
    load_loschmidt_echo_result,
    load_response_spectrum_result,
    load_sweep_result,
    load_tensor_network_state,
    load_two_time_correlator_result,
    load_time_evolution_result,
    analyze_dqpt_from_loschmidt,
    response_spectrum_from_correlator,
    save_dynamic_structure_factor_result,
    save_dqpt_diagnostics_result,
    save_dqpt_scan_result,
    save_entanglement_spectrum_result,
    save_frequency_structure_factor_result,
    save_ground_state_result,
    save_loschmidt_echo_result,
    save_response_spectrum_result,
    save_sweep_result,
    save_tensor_network_state,
    save_two_time_correlator_result,
    save_time_evolution_result,
)
from nqpu.physics import AutoSolver, HeisenbergXXZ1D, HeisenbergXYZ1D, RustTensorNetworkSolver
from nqpu.physics import SweepPoint, SweepResult
from nqpu.physics.solvers import make_initial_state
from nqpu.physics import state_io


class FakeTensorNetworkStateHandle:
    def __init__(
        self,
        num_sites: int,
        solver: str,
        *,
        model_name: str = "transverse_field_ising_1d",
        source_site: int | None = None,
        source_pauli: str | None = None,
    ):
        self.num_sites = num_sites
        self.solver = solver
        self.model_name = model_name
        self.source_site = source_site
        self.source_pauli = source_pauli

    def to_json(self) -> str:
        return json.dumps(
            {
                "num_sites": self.num_sites,
                "solver": self.solver,
                "model_name": self.model_name,
                "source_site": self.source_site,
                "source_pauli": self.source_pauli,
            }
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())

    @staticmethod
    def from_json(payload: str) -> "FakeTensorNetworkStateHandle":
        data = json.loads(payload)
        return FakeTensorNetworkStateHandle(
            int(data["num_sites"]),
            str(data["solver"]),
            model_name=str(data.get("model_name", "transverse_field_ising_1d")),
            source_site=None if data.get("source_site") is None else int(data["source_site"]),
            source_pauli=None if data.get("source_pauli") is None else str(data["source_pauli"]),
        )

    @staticmethod
    def load_json(path: str) -> "FakeTensorNetworkStateHandle":
        with open(path, "r", encoding="utf-8") as handle:
            return FakeTensorNetworkStateHandle.from_json(handle.read())


class FakeRustBindings:
    TensorNetworkState1D = FakeTensorNetworkStateHandle
    PAULI_TOKEN = re.compile(r"([XYZ])(\d+)")

    def __init__(self):
        self.calls = []

    def _ground_observable(self, label: str) -> float:
        if label == "magnetization_z":
            return 0.125
        if label == "staggered_magnetization_z":
            return 0.0
        if label == "Z0Z1":
            return 0.875

        tokens = self.PAULI_TOKEN.findall(label)
        if not tokens or "".join(f"{pauli}{site}" for pauli, site in tokens) != label:
            return 0.0
        if len(tokens) == 1:
            site = int(tokens[0][1])
            return 0.25 if site % 2 == 0 else 0.0
        if len(tokens) == 2:
            left = int(tokens[0][1])
            right = int(tokens[1][1])
            return max(0.0, 1.0 - 0.125 * abs(left - right))
        return 0.0

    def _time_observable(self, label: str, num_times: int) -> list[float]:
        if label == "magnetization_z":
            return [1.0 - 0.25 * index for index in range(num_times)]
        if label == "Z0Z1":
            return [1.0 - 0.5 * index for index in range(num_times)]

        tokens = self.PAULI_TOKEN.findall(label)
        if not tokens or "".join(f"{pauli}{site}" for pauli, site in tokens) != label:
            return [0.0 for _ in range(num_times)]
        if len(tokens) == 1:
            site = int(tokens[0][1])
            return [
                max(-1.0, 0.8 - 0.2 * index - 0.1 * (site % 2))
                for index in range(num_times)
            ]
        if len(tokens) == 2:
            left = int(tokens[0][1])
            right = int(tokens[1][1])
            distance = abs(left - right)
            return [
                max(-1.0, 0.9 - 0.3 * index - 0.05 * distance)
                for index in range(num_times)
            ]
        return [0.0 for _ in range(num_times)]

    def _transition_observable(
        self,
        label: str,
        num_times: int,
        *,
        source_site: int | None,
    ) -> list[tuple[float, float]]:
        tokens = self.PAULI_TOKEN.findall(label)
        if not tokens or len(tokens) != 1:
            return [(0.0, 0.0) for _ in range(num_times)]

        site = int(tokens[0][1])
        distance = abs(site - (0 if source_site is None else source_site))
        return [
            (
                max(-1.0, 0.9 - 0.2 * index - 0.1 * distance),
                0.05 * distance * index,
            )
            for index in range(num_times)
        ]

    def dmrg_ground_state_1d(self, **kwargs):
        state_handle = FakeTensorNetworkStateHandle(
            kwargs["num_sites"],
            "dmrg",
            model_name=kwargs["model"],
        )
        self.calls.append(kwargs)
        observables = {
            label: self._ground_observable(label)
            for label in kwargs.get("observables", [])
        }
        return {
            "model_name": kwargs["model"],
            "solver": "dmrg",
            "ground_state_energy": -5.25,
            "observables": observables,
            "entanglement_entropy": 0.5,
            "spectral_gap": None,
            "state_handle": state_handle,
        }

    def entanglement_spectrum_1d(self, **kwargs):
        self.calls.append(kwargs)
        bond = int(kwargs["bond"])
        if bond <= 0:
            spectrum = [0.95, 0.05]
        else:
            spectrum = [0.8, 0.2]
        entropy = -sum(value * np.log2(value) for value in spectrum if value > 0.0)
        return {
            "spectrum": spectrum,
            "entropy": entropy,
        }

    def tdvp_time_evolution_1d(self, **kwargs):
        state_handle = FakeTensorNetworkStateHandle(
            kwargs["num_sites"],
            "tdvp",
            model_name=kwargs["model"],
        )
        self.calls.append(kwargs)
        num_times = len(kwargs["times"])
        result = {
            "model_name": kwargs["model"],
            "solver": "tdvp",
            "times": list(kwargs["times"]),
            "observables": {
                label: self._time_observable(label, num_times)
                for label in kwargs.get("observables", [])
            },
            "state_handle": state_handle,
        }
        if kwargs.get("entropy_bond") is not None:
            result["entanglement_entropy"] = [
                0.2 + 0.1 * kwargs["entropy_bond"] + 0.05 * index for index in range(num_times)
            ]
        return result

    def statevector_to_mps_1d(self, **kwargs):
        self.calls.append(kwargs)
        return FakeTensorNetworkStateHandle(
            kwargs["num_sites"],
            "compressed_state",
            model_name=str(kwargs.get("model_name", "compressed_state")),
        )

    def tdvp_loschmidt_echo_1d(self, **kwargs):
        self.calls.append(kwargs)
        num_times = len(kwargs["times"])
        base_real = 1.0
        if kwargs.get("reference_state_handle") is not None:
            base_real = 0.7
        elif kwargs.get("reference_initial_state") is not None:
            base_real = 0.8

        return {
            "model_name": kwargs["model"],
            "solver": "tdvp_overlap",
            "times": list(kwargs["times"]),
            "amplitudes": [
                (base_real - 0.1 * index, -0.05 * index)
                for index in range(num_times)
            ],
            "state_handle": FakeTensorNetworkStateHandle(
                kwargs["num_sites"],
                "tdvp",
                model_name=kwargs["model"],
            ),
        }

    def apply_local_pauli_1d(self, **kwargs):
        state_handle = kwargs["state_handle"]
        return FakeTensorNetworkStateHandle(
            state_handle.num_sites,
            state_handle.solver,
            model_name=state_handle.model_name,
            source_site=int(kwargs["site"]),
            source_pauli=str(kwargs["pauli"]),
        )

    def tdvp_transition_observables_1d(self, **kwargs):
        self.calls.append(kwargs)
        num_times = len(kwargs["times"])
        source_state = kwargs["source_state_handle"]
        result = {
            "model_name": kwargs["model"],
            "solver": "tdvp_transition",
            "times": list(kwargs["times"]),
            "observables": {
                label: self._transition_observable(
                    label,
                    num_times,
                    source_site=getattr(source_state, "source_site", None),
                )
                for label in kwargs.get("observables", [])
            },
            "state_handle": source_state,
        }
        return result


class InterruptingSweepSolver:
    def __init__(self, *, fail_on_call: int | None = None):
        self.fail_on_call = fail_on_call
        self.calls: list[float] = []

    def solve_ground_state(self, model, observables=(), subsystem=None):
        sweep_value = float(getattr(model, "transverse_field", getattr(model, "anisotropy", 0.0)))
        self.calls.append(sweep_value)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("interrupted sweep")

        return GroundStateResult(
            model_name=model.model_name,
            solver="fake_solver",
            dimension=1 << model.num_sites,
            ground_state_energy=-sweep_value,
            spectral_gap=1.0 + sweep_value,
            eigenvalues=np.asarray([-sweep_value], dtype=np.float64),
            ground_state=None,
            observables={label: sweep_value for label in observables},
            entanglement_entropy=0.25 if subsystem is not None else None,
            model_metadata={
                "model_name": model.model_name,
                "num_sites": model.num_sites,
                "transverse_field": sweep_value,
            },
        )


class SyntheticAdaptiveSolver:
    def __init__(self):
        self.calls: list[float] = []

    def solve_ground_state(self, model, observables=(), initial_state=None, subsystem=None):
        del initial_state, subsystem
        sweep_value = float(getattr(model, "transverse_field", getattr(model, "anisotropy", 0.0)))
        self.calls.append(sweep_value)

        resolved_observables = {}
        for label in observables:
            if label == "curvature_probe":
                resolved_observables[label] = float(np.exp(-40.0 * (sweep_value - 1.0) ** 2))
            elif label == "order_parameter":
                resolved_observables[label] = sweep_value
            else:
                resolved_observables[label] = sweep_value

        return GroundStateResult(
            model_name=model.model_name,
            solver="synthetic_solver",
            dimension=1 << model.num_sites,
            ground_state_energy=-(sweep_value**2),
            spectral_gap=0.25 + (sweep_value - 1.0) ** 2,
            eigenvalues=np.asarray([-(sweep_value**2)], dtype=np.float64),
            ground_state=None,
            observables=resolved_observables,
            entanglement_entropy=0.5 * float(np.exp(-8.0 * (sweep_value - 1.0) ** 2)),
            model_metadata={
                "model_name": model.model_name,
                "num_sites": model.num_sites,
                "transverse_field": sweep_value,
            },
        )


class SyntheticAdaptiveDQPTQPU(ModelQPU):
    def __init__(self, *, fail_on_call: int | None = None):
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls: list[float] = []

    def dqpt_diagnostics(self, model, times, **kwargs):
        del kwargs
        sweep_value = float(getattr(model, "transverse_field", 0.0))
        self.calls.append(sweep_value)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("interrupted adaptive dqpt scan")

        return_rate = np.asarray(
            [
                0.1 + 0.2 * sweep_value,
                0.35 + np.exp(-40.0 * (sweep_value - 1.0) ** 2),
                0.15 + 0.1 * sweep_value,
            ],
            dtype=np.float64,
        )
        strongest_time = 0.4 + 0.5 * sweep_value
        candidate = DQPTCandidate(
            index=1,
            time=strongest_time,
            return_rate=float(return_rate[1]),
            prominence=0.2 + 0.4 * sweep_value,
            cusp_strength=float(np.exp(-40.0 * (sweep_value - 1.0) ** 2)),
            left_slope=1.0,
            right_slope=-1.0,
        )
        return DQPTDiagnosticsResult(
            model_name=model.model_name,
            solver="synthetic_dqpt",
            num_sites=model.num_sites,
            times=np.asarray(list(times), dtype=np.float64),
            return_rate=return_rate,
            candidates=(candidate,),
            amplitudes=np.exp(-0.5 * model.num_sites * return_rate).astype(np.complex128),
            model_metadata={
                "model_name": model.model_name,
                "num_sites": model.num_sites,
                "transverse_field": sweep_value,
            },
        )


class SyntheticTimeAdaptiveDQPTQPU(ModelQPU):
    def __init__(self):
        super().__init__()
        self.calls: list[np.ndarray] = []

    def dqpt_diagnostics(self, model, times, **kwargs):
        del model, kwargs
        times_array = np.asarray(list(times), dtype=np.float64)
        self.calls.append(times_array.copy())
        return_rate = 0.1 + np.exp(-40.0 * (times_array - 0.8) ** 2)
        peak_index = int(np.argmax(return_rate))
        candidate = DQPTCandidate(
            index=peak_index,
            time=float(times_array[peak_index]),
            return_rate=float(return_rate[peak_index]),
            prominence=max(float(return_rate[peak_index] - np.min(return_rate)), 0.0),
            cusp_strength=float(np.exp(-40.0 * (times_array[peak_index] - 0.8) ** 2)),
            left_slope=1.0,
            right_slope=-1.0,
        )
        amplitudes = np.exp(-2.0 * return_rate).astype(np.complex128)
        return DQPTDiagnosticsResult(
            model_name="synthetic_time_dqpt",
            solver="synthetic_dqpt",
            num_sites=4,
            times=times_array,
            return_rate=return_rate,
            candidates=(candidate,),
            amplitudes=amplitudes,
            model_metadata={"model_name": "synthetic_time_dqpt", "num_sites": 4},
        )


class SyntheticLinearTimeDQPTQPU(ModelQPU):
    def dqpt_diagnostics(self, model, times, **kwargs):
        del model, kwargs
        times_array = np.asarray(list(times), dtype=np.float64)
        return_rate = times_array.copy()
        peak_index = int(np.argmax(return_rate))
        candidate = DQPTCandidate(
            index=peak_index,
            time=float(times_array[peak_index]),
            return_rate=float(return_rate[peak_index]),
            prominence=0.0,
            cusp_strength=0.0,
            left_slope=1.0,
            right_slope=1.0,
        )
        amplitudes = np.exp(-2.0 * return_rate).astype(np.complex128)
        return DQPTDiagnosticsResult(
            model_name="synthetic_linear_time_dqpt",
            solver="synthetic_dqpt",
            num_sites=4,
            times=times_array,
            return_rate=return_rate,
            candidates=(candidate,),
            amplitudes=amplitudes,
            model_metadata={"model_name": "synthetic_linear_time_dqpt", "num_sites": 4},
        )


def test_quantum_circuit_backend_runs_bell_state():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure_all()

    backend = NQPUBackend(seed=7)
    result = backend.run(circuit, shots=512)

    dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
    assert dominant > 450
    assert result.shots == 512


def test_model_qpu_ground_state_and_observables():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.ground_state(
        model,
        observables=["magnetization_z", "Z0Z1"],
        subsystem=[0, 1],
    )

    assert result.ground_state_energy < 0.0
    assert "magnetization_z" in result.observables
    assert "Z0Z1" in result.observables
    assert result.entanglement_entropy is not None


def test_model_qpu_correlation_matrix_supports_connected_correlators():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    raw = qpu.correlation_matrix(model, pauli="Z", connected=False)
    connected = qpu.correlation_matrix(model, pauli="Z", connected=True)

    expected_connected = raw.matrix - np.outer(
        raw.single_site_expectations,
        raw.single_site_expectations,
    )
    assert raw.solver == "exact_diagonalization"
    assert raw.matrix.shape == (4, 4)
    assert np.allclose(raw.matrix, raw.matrix.T)
    assert np.allclose(np.diag(raw.matrix), np.ones(4))
    assert np.allclose(connected.matrix, expected_connected)


def test_model_qpu_structure_factor_matches_manual_correlator_sum():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    momenta = np.asarray([0.0, np.pi / 2.0, np.pi], dtype=np.float64)

    correlations = qpu.correlation_matrix(model, pauli="Z", connected=True)
    structure = qpu.structure_factor(
        model,
        momenta,
        pauli="Z",
        connected=True,
    )

    positions = np.arange(model.num_sites, dtype=np.float64)
    manual = []
    for momentum in momenta:
        phase = np.exp(1.0j * momentum * (positions[:, None] - positions[None, :]))
        manual.append(np.real(np.sum(phase * correlations.matrix) / model.num_sites))

    assert structure.values.shape == (3,)
    assert np.allclose(structure.values, manual)
    assert np.allclose(structure.correlation_matrix, correlations.matrix)


def test_model_qpu_dynamic_structure_factor_uses_quench_observables():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.dynamic_structure_factor(
        model,
        [0.0, 0.2],
        [0.0, np.pi],
        pauli="Z",
        initial_state="all_up",
    )

    assert result.solver == "exact_diagonalization"
    assert result.values.shape == (2, 2)
    assert np.isclose(result.values[0, 0], 4.0)
    assert abs(result.values[0, 1]) < 1e-8


def test_fourier_transform_structure_factor_matches_constant_signal_integral():
    dynamic = DynamicStructureFactorResult(
        model_name="synthetic",
        solver="synthetic",
        pauli="Z",
        connected=False,
        times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        momenta=np.asarray([0.0], dtype=np.float64),
        values=np.ones((3, 1), dtype=np.float64),
    )

    spectral = fourier_transform_structure_factor(
        dynamic,
        frequencies=[0.0],
        window="none",
    )

    assert isinstance(spectral, FrequencyStructureFactorResult)
    assert spectral.values.shape == (1, 1)
    assert np.isclose(spectral.values[0, 0], 2.0 + 0.0j)
    assert np.isclose(spectral.intensity[0, 0], 2.0)


def test_model_qpu_frequency_structure_factor_builds_frequency_response():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    spectral = qpu.frequency_structure_factor(
        model,
        times=[0.0, 0.2, 0.4, 0.6],
        momenta=[0.0, np.pi],
        pauli="Z",
        initial_state="all_up",
        frequencies=[0.0],
        window="none",
    )

    assert isinstance(spectral, FrequencyStructureFactorResult)
    assert spectral.solver == "exact_diagonalization"
    assert spectral.values.shape == (1, 2)
    assert np.all(spectral.intensity >= 0.0)


def test_model_qpu_two_time_correlator_matches_equal_time_connected_correlations():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    equal_time = qpu.correlation_matrix(model, pauli="Z", connected=True)
    two_time = qpu.two_time_correlator(
        model,
        [0.0],
        pauli="Z",
        connected=True,
    )

    assert isinstance(two_time, TwoTimeCorrelatorResult)
    assert two_time.solver == "exact_diagonalization"
    assert two_time.values.shape == (1, 4, 4)
    assert np.allclose(two_time.values[0], equal_time.matrix)


def test_model_qpu_two_time_correlator_can_select_source_and_measure_sites():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    full = qpu.two_time_correlator(
        model,
        [0.0, 0.2],
        pauli="Z",
        connected=False,
    )
    subset = qpu.two_time_correlator(
        model,
        [0.0, 0.2],
        pauli="Z",
        connected=False,
        measure_sites=[0, 2],
        source_sites=[1, 3],
    )

    assert subset.values.shape == (2, 2, 2)
    assert subset.measure_sites == (0, 2)
    assert subset.source_sites == (1, 3)
    assert subset.num_sites == 4
    assert np.allclose(subset.values, full.values[:, [0, 2]][:, :, [1, 3]])
    assert np.allclose(
        subset.dynamic_single_site_expectations,
        full.dynamic_single_site_expectations[:, [0, 2]],
    )
    assert np.allclose(
        subset.initial_single_site_expectations,
        full.initial_single_site_expectations[[1, 3]],
    )


def test_response_spectrum_from_correlator_matches_simple_commutator_integral():
    correlator = TwoTimeCorrelatorResult(
        model_name="synthetic",
        solver="exact_diagonalization",
        pauli="Z",
        connected=False,
        num_sites=1,
        times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        values=np.asarray([[[0.0j]], [[1.0j]], [[2.0j]]], dtype=np.complex128),
        dynamic_single_site_expectations=np.zeros((3, 1), dtype=np.float64),
        initial_single_site_expectations=np.zeros(1, dtype=np.float64),
        measure_sites=(0,),
        source_sites=(0,),
    )

    response = response_spectrum_from_correlator(
        correlator,
        momenta=[0.0],
        frequencies=[0.0],
        window="none",
    )

    assert isinstance(response, ResponseSpectrumResult)
    assert response.time_response.shape == (3, 1)
    assert np.allclose(response.time_response[:, 0], [0.0, 2.0, 4.0])
    assert np.isclose(response.values[0, 0], 4.0 + 0.0j)


def test_response_spectrum_from_correlator_requires_matching_site_sets():
    correlator = TwoTimeCorrelatorResult(
        model_name="synthetic",
        solver="exact_diagonalization",
        pauli="Z",
        connected=False,
        num_sites=2,
        times=np.asarray([0.0, 1.0], dtype=np.float64),
        values=np.asarray([[[0.0j]], [[1.0j]]], dtype=np.complex128),
        dynamic_single_site_expectations=np.zeros((2, 1), dtype=np.float64),
        initial_single_site_expectations=np.zeros(1, dtype=np.float64),
        measure_sites=(0,),
        source_sites=(1,),
    )

    with pytest.raises(ValueError, match="matching measure_sites and source_sites"):
        response_spectrum_from_correlator(correlator, momenta=[0.0], frequencies=[0.0], window="none")


def test_model_qpu_linear_response_spectrum_builds_exact_response():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    response = qpu.linear_response_spectrum(
        model,
        times=[0.0, 0.2, 0.4, 0.6],
        momenta=[0.0, np.pi],
        pauli="Z",
        frequencies=[0.0],
        window="none",
    )

    assert isinstance(response, ResponseSpectrumResult)
    assert response.solver == "exact_diagonalization"
    assert response.values.shape == (1, 2)
    assert response.time_response.shape == (4, 2)


def test_model_qpu_linear_response_spectrum_can_use_matching_site_subset():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    response = qpu.linear_response_spectrum(
        model,
        times=[0.0, 0.2, 0.4, 0.6],
        momenta=[0.0, np.pi],
        pauli="Z",
        source_sites=[0, 2],
        frequencies=[0.0],
        window="none",
    )

    assert response.solver == "exact_diagonalization"
    assert response.values.shape == (1, 2)
    assert response.time_response.shape == (4, 2)
    assert response.measure_sites == (0, 2)
    assert response.source_sites == (0, 2)
    assert response.num_sites == 4


def test_model_qpu_entanglement_spectrum_matches_ground_state_entropy():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    ground = qpu.ground_state(model, subsystem=[0, 1])
    spectrum = qpu.entanglement_spectrum(model, subsystem=[0, 1])

    assert isinstance(spectrum, EntanglementSpectrumResult)
    assert spectrum.solver == "exact_diagonalization"
    assert spectrum.subsystem == (0, 1)
    assert spectrum.num_sites == 4
    assert np.isclose(np.sum(spectrum.eigenvalues), 1.0)
    assert np.isclose(spectrum.entropy, ground.entanglement_entropy)
    assert np.all(spectrum.schmidt_values >= 0.0)
    assert np.all(np.isfinite(spectrum.entanglement_energies))


def test_model_qpu_entanglement_spectrum_can_use_final_dense_time_evolution_state():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    evolution = qpu.quench(
        model,
        [0.0, 0.2, 0.4],
        initial_state="neel",
        observables=["magnetization_z"],
    )

    spectrum = qpu.entanglement_spectrum(
        model,
        subsystem=[0, 1],
        initial_state=evolution,
        num_levels=2,
    )

    assert spectrum.solver == "exact_diagonalization"
    assert spectrum.eigenvalues.shape == (2,)
    assert np.isclose(np.sum(spectrum.eigenvalues), 1.0)


def test_model_qpu_loschmidt_echo_tracks_eigenstate_phase():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    ground = qpu.ground_state(model)
    times = np.asarray([0.0, 0.2, 0.4], dtype=np.float64)

    result = qpu.loschmidt_echo(model, times, initial_state=ground)

    assert isinstance(result, LoschmidtEchoResult)
    assert result.solver == "exact_diagonalization"
    assert result.num_sites == 4
    assert np.allclose(result.times, times)
    assert np.allclose(
        result.amplitudes,
        np.exp(-1.0j * ground.ground_state_energy * times),
    )
    assert np.allclose(result.echo, np.ones_like(times))
    assert np.allclose(result.return_rate, np.zeros_like(times))
    assert result.backend_state is None


def test_model_qpu_loschmidt_echo_accepts_separate_reference_state():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    ground = qpu.ground_state(model)

    result = qpu.loschmidt_echo(
        model,
        [0.0, 0.2],
        initial_state="all_up",
        reference_state=ground,
    )

    expected_overlap = np.conjugate(ground.ground_state[0])

    assert result.solver == "exact_diagonalization"
    assert np.isclose(result.amplitudes[0], expected_overlap)
    assert result.amplitudes.shape == (2,)


def test_make_initial_state_supports_extended_product_state_labels():
    plus_x = make_initial_state(1, "plus_x")
    plus_y = make_initial_state(1, "R")
    domain_wall = make_initial_state(4, "domain_wall")

    assert np.allclose(plus_x, [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    assert np.allclose(plus_y, [1.0 / np.sqrt(2.0), 1.0j / np.sqrt(2.0)])
    assert np.allclose(domain_wall, make_initial_state(4, "0011"))


def test_heisenberg_xyz_model_supports_exact_ground_state_workflows():
    qpu = ModelQPU()
    model = HeisenbergXYZ1D(
        num_sites=2,
        coupling_x=1.2,
        coupling_y=0.7,
        coupling_z=-0.5,
        field_z=0.3,
    )

    hamiltonian = model.hamiltonian()
    result = qpu.ground_state(model, observables=["energy", "Z0Z1"])

    assert np.allclose(hamiltonian, hamiltonian.conj().T)
    assert result.solver == "exact_diagonalization"
    assert result.model_name == "heisenberg_xyz_1d"
    assert np.isclose(result.observables["energy"], result.ground_state_energy)
    assert result.model_metadata is not None
    assert result.model_metadata["coupling_x"] == 1.2
    assert result.model_metadata["coupling_y"] == 0.7
    assert result.model_metadata["coupling_z"] == -0.5


def test_model_qpu_loschmidt_echo_respects_dense_dimension_limit():
    solver = AutoSolver()
    solver.exact_solver.max_dimension = 8
    solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    with pytest.raises(ValueError, match="Loschmidt echo currently requires exact diagonalization"):
        qpu.loschmidt_echo(model, [0.0, 0.2], initial_state="all_up")


def test_model_qpu_loschmidt_echo_can_use_rust_tdvp_overlap_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.loschmidt_echo(model, [0.0, 0.2, 0.4], initial_state="neel")

    overlap_calls = [
        call
        for call in solver.tensor_network_solver.bindings.calls
        if "initial_state" in call or "state_handle" in call
        if "times" in call and "amplitudes" not in call
    ]

    assert isinstance(result, LoschmidtEchoResult)
    assert result.solver == "tdvp_overlap"
    assert result.amplitudes.shape == (3,)
    assert np.allclose(result.amplitudes, [1.0 + 0.0j, 0.9 - 0.05j, 0.8 - 0.1j])
    assert result.backend_state is not None
    assert len(overlap_calls) == 1
    assert overlap_calls[0]["initial_state"] == "1010"


def test_model_qpu_quench_can_use_dense_statevector_via_rust_compression():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    dense_state = make_initial_state(4, "plus_x")
    result = qpu.quench(
        model,
        [0.0, 0.2],
        initial_state=dense_state,
        observables=["magnetization_z"],
    )

    compression_call = bindings.calls[0]
    tdvp_call = bindings.calls[1]

    assert result.solver == "tdvp"
    assert compression_call["num_sites"] == 4
    assert compression_call["model_name"] == model.model_name
    assert "statevector" in compression_call
    assert tdvp_call["state_handle"].solver == "compressed_state"
    assert "initial_state" not in tdvp_call


def test_model_qpu_loschmidt_echo_can_use_extended_rust_product_state_label():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.loschmidt_echo(model, [0.0, 0.2], initial_state="+-RL")

    overlap_call = bindings.calls[-1]

    assert result.solver == "tdvp_overlap"
    assert np.allclose(result.amplitudes, [1.0 + 0.0j, 0.9 - 0.05j])
    assert overlap_call["initial_state"] == "+-RL"


def test_model_qpu_loschmidt_echo_can_use_dense_ground_state_result_via_rust_compression():
    exact_qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    ground = exact_qpu.ground_state(model)

    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)

    result = qpu.loschmidt_echo(model, [0.0, 0.2], initial_state=ground)

    compression_call = bindings.calls[0]
    overlap_call = bindings.calls[1]

    assert result.solver == "tdvp_overlap"
    assert compression_call["num_sites"] == 4
    assert compression_call["model_name"] == model.model_name
    assert "statevector" in compression_call
    assert overlap_call["state_handle"].solver == "compressed_state"
    assert np.allclose(result.amplitudes, [1.0 + 0.0j, 0.9 - 0.05j])


def test_model_qpu_loschmidt_echo_can_use_rust_backend_reference_state():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    reference = qpu.ground_state(model)
    result = qpu.loschmidt_echo(
        model,
        [0.0, 0.2],
        initial_state="neel",
        reference_state=reference,
    )

    overlap_call = bindings.calls[-1]

    assert result.solver == "tdvp_overlap"
    assert np.allclose(result.amplitudes, [0.7 + 0.0j, 0.6 - 0.05j])
    assert overlap_call["reference_state_handle"] is reference.backend_state


def test_rust_loschmidt_bridge_receives_xyz_model_parameters():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.2,
        coupling_y=0.8,
        coupling_z=1.3,
        field_z=0.25,
    )

    result = qpu.loschmidt_echo(
        model,
        [0.0, 0.2],
        initial_state="plus_y",
        reference_state="minus_x",
    )

    call = bindings.calls[-1]

    assert result.solver == "tdvp_overlap"
    assert call["model"] == "heisenberg_xyz_1d"
    assert call["coupling_x"] == 1.2
    assert call["coupling_y"] == 0.8
    assert call["coupling_z"] == 1.3
    assert call["field_z"] == 0.25
    assert call["initial_state"] == "RRRR"
    assert call["reference_initial_state"] == "----"


def test_analyze_dqpt_from_loschmidt_detects_peak_candidate():
    return_rate = np.asarray([0.05, 0.2, 0.8, 0.25, 0.1], dtype=np.float64)
    amplitudes = np.exp(-2.0 * return_rate).astype(np.complex128)
    echo = LoschmidtEchoResult(
        model_name="synthetic",
        solver="synthetic",
        num_sites=4,
        times=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        amplitudes=amplitudes,
    )

    diagnostics = analyze_dqpt_from_loschmidt(
        echo,
        min_prominence=0.2,
        min_cusp=5.0,
    )

    assert isinstance(diagnostics, DQPTDiagnosticsResult)
    assert diagnostics.solver == "synthetic"
    assert np.allclose(diagnostics.return_rate, return_rate)
    assert np.allclose(diagnostics.candidate_times, [0.2])
    assert len(diagnostics.candidates) == 1
    candidate = diagnostics.candidates[0]
    assert candidate.index == 2
    assert np.isclose(candidate.return_rate, 0.8)
    assert candidate.prominence > 0.5
    assert candidate.cusp_strength > 5.0


def test_model_qpu_dqpt_diagnostics_uses_exact_loschmidt_path():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    diagnostics = qpu.dqpt_diagnostics(model, [0.0, 0.1, 0.2], initial_state="all_up")

    assert isinstance(diagnostics, DQPTDiagnosticsResult)
    assert diagnostics.solver == "exact_diagonalization"
    assert diagnostics.return_rate.shape == (3,)
    assert diagnostics.amplitudes is not None


def test_model_qpu_dqpt_diagnostics_can_use_rust_loschmidt_path():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    diagnostics = qpu.dqpt_diagnostics(
        model,
        [0.0, 0.2, 0.4],
        initial_state="plus_x",
    )

    call = bindings.calls[-1]

    assert diagnostics.solver == "tdvp_overlap"
    assert diagnostics.return_rate.shape == (3,)
    assert diagnostics.amplitudes is not None
    assert diagnostics.backend_state is not None
    assert call["model"] == "heisenberg_xyz_1d"


def test_model_qpu_adaptive_dqpt_diagnostics_refines_peak_region():
    qpu = SyntheticTimeAdaptiveDQPTQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.adaptive_dqpt_diagnostics(
        model,
        [0.0, 0.5, 1.0, 1.5],
        metric="return_rate",
        strategy="curvature",
        max_refinement_rounds=1,
    )

    assert isinstance(result, AdaptiveDQPTDiagnosticsResult)
    assert result.refinement_metric == "return_rate"
    assert result.refinement_strategy == "curvature"
    assert result.insertion_policy == "equal_spacing"
    assert result.completed_rounds == 1
    assert np.allclose(result.seed_times, [0.0, 0.5, 1.0, 1.5])
    assert np.isclose(result.refinement_history[0]["inserted_values"][0], 0.75)
    assert np.any(np.isclose(result.times, 0.75))
    assert len(qpu.calls) == 2


def test_adaptive_dqpt_diagnostics_target_crossing_can_insert_requested_time():
    qpu = SyntheticLinearTimeDQPTQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.adaptive_dqpt_diagnostics(
        model,
        [0.0, 0.5, 1.0, 1.5],
        metric="return_rate",
        strategy="target_crossing",
        target_value=0.8,
        max_refinement_rounds=1,
    )

    assert result.refinement_strategy == "target_crossing"
    assert np.isclose(result.refinement_target_value, 0.8)
    assert result.insertion_policy == "target_linear"
    assert np.isclose(result.refinement_history[0]["inserted_values"][0], 0.8)
    assert np.any(np.isclose(result.times, 0.8))


def test_model_qpu_adaptive_dqpt_diagnostics_works_on_exact_path():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.adaptive_dqpt_diagnostics(
        model,
        [0.0, 0.1, 0.2],
        metric="return_rate",
        strategy="gradient_magnitude",
        max_refinement_rounds=1,
    )

    assert isinstance(result, AdaptiveDQPTDiagnosticsResult)
    assert result.solver == "exact_diagonalization"
    assert result.times.ndim == 1
    assert result.return_rate.shape == result.times.shape
    assert result.completed_rounds in (0, 1)


def test_model_qpu_dqpt_parameter_scan_builds_exact_summary_traces():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    scan = qpu.scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.5, 0.7, 0.9],
        [0.0, 0.1, 0.2],
        initial_state="all_up",
    )

    assert isinstance(scan, DQPTScanResult)
    assert scan.solver == "exact_diagonalization"
    assert scan.values.shape == (3,)
    assert scan.times.shape == (3,)
    assert scan.return_rates.shape == (3, 3)
    assert scan.candidate_counts.shape == (3,)
    assert scan.max_return_rates.shape == (3,)
    assert scan.completed_points == 3
    assert scan.is_complete
    assert len(scan.points) == 3
    assert scan.points[0].return_rate.shape == (3,)


def test_model_qpu_dqpt_parameter_scan_can_use_rust_loschmidt_path():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    scan = qpu.scan_dqpt_parameter(
        model,
        "field_z",
        [0.05, 0.15],
        [0.0, 0.2, 0.4],
        initial_state="plus_x",
    )

    assert scan.solver == "tdvp_overlap"
    assert scan.return_rates.shape == (2, 3)
    assert len(scan.points) == 2
    assert scan.points[0].backend_state is not None
    assert bindings.calls[-1]["model"] == "heisenberg_xyz_1d"


def test_model_qpu_adaptive_dqpt_scan_refines_strong_cusp_region():
    qpu = SyntheticAdaptiveDQPTQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    scan = qpu.adaptive_scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [0.0, 0.1, 0.2],
        metric="strongest_cusp_strength",
        strategy="curvature",
        max_refinement_rounds=1,
    )

    assert np.allclose(scan.seed_values, [0.0, 0.5, 1.0, 1.5, 2.0])
    assert scan.refinement_metric == "strongest_cusp_strength"
    assert scan.refinement_strategy == "curvature"
    assert scan.insertion_policy == "equal_spacing"
    assert scan.completed_rounds == 1
    assert scan.is_complete
    assert np.isclose(scan.refinement_history[0]["inserted_values"][0], 0.75)
    assert np.any(np.isclose(scan.values, 0.75))


def test_adaptive_dqpt_scan_target_crossing_prefers_interval_near_target():
    qpu = SyntheticAdaptiveDQPTQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    scan = qpu.adaptive_scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 0.1, 0.2],
        metric="strongest_candidate_time",
        strategy="target_crossing",
        target_value=0.8,
        max_refinement_rounds=1,
    )

    assert scan.refinement_metric == "strongest_candidate_time"
    assert scan.refinement_strategy == "target_crossing"
    assert np.isclose(scan.refinement_target_value, 0.8)
    assert scan.insertion_policy == "target_linear"
    assert scan.points_per_interval == 1
    assert np.isclose(scan.refinement_history[0]["inserted_values"][0], 0.8)
    assert np.any(np.isclose(scan.values, 0.8))


def test_adaptive_dqpt_scan_allows_missing_strongest_candidate_metrics():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    scan = qpu.adaptive_scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.5, 1.0, 1.5],
        [0.0, 0.1, 0.2],
        metric="strongest_cusp_strength",
        strategy="gradient_magnitude",
        max_refinement_rounds=1,
    )

    assert scan.completed_points == 3
    assert scan.is_complete
    assert scan.completed_rounds == 0
    assert len(scan.refinement_history) == 0


def test_model_qpu_parameter_sweep():
    qpu = ModelQPU()
    model = HeisenbergXXZ1D(num_sites=4, anisotropy=0.5)

    sweep = qpu.sweep_parameter(
        model,
        "anisotropy",
        np.linspace(0.25, 1.25, 5),
        observables=["magnetization_z"],
        subsystem=[0, 1],
    )

    assert sweep.parameter == "anisotropy"
    assert sweep.model_name == model.model_name
    assert sweep.solver == "exact_diagonalization"
    assert sweep.values.shape == (5,)
    assert sweep.energies.shape == (5,)
    assert sweep.spectral_gaps.shape == (5,)
    assert sweep.entanglement_entropy.shape == (5,)
    assert sweep.observables["magnetization_z"].shape == (5,)
    assert sweep.observables_requested == ("magnetization_z",)
    assert sweep.subsystem == (0, 1)
    assert sweep.base_model_metadata["anisotropy"] == 0.5
    assert sweep.points[0].solver == "exact_diagonalization"
    assert sweep.points[0].model_metadata["anisotropy"] == 0.25
    assert sweep.completed_points == 5
    assert sweep.is_complete


def test_model_qpu_adaptive_sweep_refines_gap_region():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    coarse = qpu.sweep_parameter(
        model,
        "transverse_field",
        [0.25, 0.75, 1.25, 1.75],
        observables=["magnetization_z"],
        subsystem=[0, 1],
    )

    sweep = qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.25, 0.75, 1.25, 1.75],
        observables=["magnetization_z"],
        metric="spectral_gap",
        max_refinement_rounds=1,
        refinements_per_round=1,
        subsystem=[0, 1],
    )

    assert np.allclose(sweep.seed_values, [0.25, 0.75, 1.25, 1.75])
    assert sweep.refinement_metric == "spectral_gap"
    assert sweep.completed_rounds == 1
    assert sweep.is_complete
    assert sweep.values.shape == (5,)
    assert len(sweep.refinement_history) == 1
    inserted_values = sweep.refinement_history[0]["inserted_values"]
    interval_scores = 0.5 * (coarse.spectral_gaps[:-1] + coarse.spectral_gaps[1:])
    best_interval = int(np.argmin(interval_scores))
    expected_midpoint = 0.5 * (coarse.values[best_interval] + coarse.values[best_interval + 1])
    assert len(inserted_values) == 1
    assert np.isclose(inserted_values[0], expected_midpoint)
    assert np.any(np.isclose(sweep.values, expected_midpoint))


def test_adaptive_sweep_curvature_strategy_prefers_high_curvature_interval():
    qpu = ModelQPU(solver=SyntheticAdaptiveSolver())
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    sweep = qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5, 2.0],
        metric="curvature_probe",
        strategy="curvature",
        max_refinement_rounds=1,
    )

    assert sweep.refinement_strategy == "curvature"
    assert sweep.refinement_target_value is None
    assert sweep.insertion_policy == "equal_spacing"
    assert sweep.points_per_interval == 1
    assert sweep.observables_requested == ("curvature_probe",)
    assert sweep.refinement_history[0]["strategy"] == "curvature"
    assert sweep.refinement_history[0]["insertion_policy"] == "equal_spacing"
    assert np.isclose(sweep.refinement_history[0]["inserted_values"][0], 0.75)
    assert np.any(np.isclose(sweep.values, 0.75))


def test_adaptive_sweep_target_crossing_prefers_interval_near_target():
    qpu = ModelQPU(solver=SyntheticAdaptiveSolver())
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    sweep = qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5],
        metric="order_parameter",
        strategy="target_crossing",
        target_value=0.8,
        max_refinement_rounds=1,
    )

    assert sweep.refinement_strategy == "target_crossing"
    assert np.isclose(sweep.refinement_target_value, 0.8)
    assert sweep.insertion_policy == "target_linear"
    assert sweep.points_per_interval == 1
    assert sweep.observables_requested == ("order_parameter",)
    assert sweep.refinement_history[0]["strategy"] == "target_crossing"
    assert np.isclose(sweep.refinement_history[0]["target_value"], 0.8)
    assert sweep.refinement_history[0]["insertion_policy"] == "target_linear"
    assert np.isclose(sweep.refinement_history[0]["inserted_values"][0], 0.8)
    assert np.any(np.isclose(sweep.values, 0.8))


def test_adaptive_sweep_equal_spacing_can_insert_multiple_points_per_interval():
    qpu = ModelQPU(solver=SyntheticAdaptiveSolver())
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    sweep = qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5],
        metric="order_parameter",
        strategy="target_crossing",
        target_value=0.8,
        insertion_policy="equal_spacing",
        points_per_interval=2,
        max_refinement_rounds=1,
    )

    assert sweep.insertion_policy == "equal_spacing"
    assert sweep.points_per_interval == 2
    assert np.allclose(
        sweep.refinement_history[0]["inserted_values"],
        [2.0 / 3.0, 5.0 / 6.0],
    )
    assert np.allclose(
        sweep.refinement_history[0]["inserted_by_interval"][0],
        [2.0 / 3.0, 5.0 / 6.0],
    )
    assert np.allclose(sweep.values, [0.0, 0.5, 2.0 / 3.0, 5.0 / 6.0, 1.0, 1.5])


def test_exact_quench_accepts_ground_state_result_as_initial_condition():
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    ground = qpu.ground_state(model, subsystem=[0, 1])
    evolution = qpu.quench(
        model,
        [0.0],
        initial_state=ground,
        observables=["energy"],
        subsystem=[0, 1],
    )

    assert evolution.solver == "exact_diagonalization"
    assert evolution.final_state is not None
    assert np.isclose(evolution.observables["energy"][0], ground.ground_state_energy)
    assert evolution.entanglement_subsystem == (0, 1)
    assert evolution.entanglement_entropy is not None
    assert np.isclose(evolution.entanglement_entropy[0], ground.entanglement_entropy)


def test_auto_solver_uses_rust_dmrg_when_exact_diagonalization_would_overflow():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8

    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    result = solver.solve_ground_state(
        model,
        observables=["magnetization_z", "Z0Z1"],
        subsystem=[0, 1],
    )

    assert result.solver == "dmrg"
    assert result.ground_state is None
    assert result.observables["magnetization_z"] == 0.125
    assert result.entanglement_entropy == 0.5
    assert result.backend_state is not None


def test_model_qpu_correlation_matrix_can_use_rust_dmrg_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    correlations = qpu.correlation_matrix(model, pauli="Z", connected=True)

    assert correlations.solver == "dmrg"
    assert correlations.matrix.shape == (4, 4)
    assert correlations.single_site_expectations.shape == (4,)
    assert np.allclose(correlations.matrix, correlations.matrix.T)


def test_rust_dmrg_bridge_receives_model_parameters_and_entropy_bond():
    bindings = FakeRustBindings()
    solver = RustTensorNetworkSolver(bindings=bindings)
    model = HeisenbergXXZ1D(num_sites=6, coupling_xy=1.5, anisotropy=0.75, field_z=0.2)

    solver.solve_ground_state(model, observables=["magnetization_z"], subsystem=[0, 1, 2])

    assert bindings.calls
    call = bindings.calls[0]
    assert call["model"] == "heisenberg_xxz_1d"
    assert call["coupling"] == 1.5
    assert call["anisotropy"] == 0.75
    assert call["field_z"] == 0.2
    assert call["entropy_bond"] == 2


def test_rust_dmrg_bridge_receives_xyz_model_parameters_and_entropy_bond():
    bindings = FakeRustBindings()
    solver = RustTensorNetworkSolver(bindings=bindings)
    model = HeisenbergXYZ1D(
        num_sites=6,
        coupling_x=1.25,
        coupling_y=0.75,
        coupling_z=1.5,
        field_z=0.2,
    )

    solver.solve_ground_state(model, observables=["magnetization_z"], subsystem=[0, 1, 2])

    assert bindings.calls
    call = bindings.calls[0]
    assert call["model"] == "heisenberg_xyz_1d"
    assert call["coupling_x"] == 1.25
    assert call["coupling_y"] == 0.75
    assert call["coupling_z"] == 1.5
    assert call["field_z"] == 0.2
    assert call["entropy_bond"] == 2


def test_rust_dmrg_bridge_receives_warm_start_state_handle():
    bindings = FakeRustBindings()
    solver = RustTensorNetworkSolver(bindings=bindings)
    model = TransverseFieldIsing1D(num_sites=6, coupling=1.0, transverse_field=0.7)
    state_handle = FakeTensorNetworkStateHandle(6, "dmrg", model_name=model.model_name)

    solver.solve_ground_state(
        model,
        observables=["magnetization_z"],
        initial_state=state_handle,
        subsystem=[0, 1, 2],
    )

    assert bindings.calls
    call = bindings.calls[0]
    assert call["state_handle"] is state_handle


def test_auto_solver_uses_rust_tdvp_when_exact_diagonalization_would_overflow():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8

    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    result = solver.time_evolve(
        model,
        [0.0, 0.2, 0.4],
        initial_state="neel",
        observables=["magnetization_z", "Z0Z1"],
        subsystem=[0, 1],
    )

    assert result.solver == "tdvp"
    assert result.times.shape == (3,)
    assert np.allclose(result.observables["magnetization_z"], [1.0, 0.75, 0.5])
    assert np.allclose(result.observables["Z0Z1"], [1.0, 0.5, 0.0])
    assert result.backend_state is not None
    assert result.entanglement_subsystem == (0, 1)
    assert np.allclose(result.entanglement_entropy, [0.3, 0.35, 0.4])


def test_model_qpu_dynamic_structure_factor_can_use_rust_tdvp_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.dynamic_structure_factor(
        model,
        [0.0, 0.2, 0.4],
        [0.0, np.pi],
        pauli="Z",
        connected=True,
        initial_state="neel",
    )

    assert result.solver == "tdvp"
    assert result.values.shape == (3, 2)
    assert np.all(np.isfinite(result.values))


def test_model_qpu_frequency_structure_factor_can_use_rust_tdvp_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    spectral = qpu.frequency_structure_factor(
        model,
        times=[0.0, 0.2, 0.4],
        momenta=[0.0, np.pi],
        pauli="Z",
        connected=True,
        initial_state="neel",
        frequencies=[0.0],
        window="none",
    )

    assert spectral.solver == "tdvp"
    assert spectral.values.shape == (1, 2)
    assert np.all(spectral.intensity >= 0.0)


def test_model_qpu_two_time_correlator_requires_exact_diagonalization_capacity():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=None),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    with pytest.raises(ValueError, match="two-time correlators currently require exact diagonalization"):
        qpu.two_time_correlator(model, [0.0, 0.2], pauli="Z")


def test_model_qpu_two_time_correlator_can_use_rust_tdvp_transition_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.two_time_correlator(
        model,
        [0.0, 0.2, 0.4],
        pauli="Z",
        connected=True,
    )

    assert result.solver == "tdvp_transition"
    assert result.values.shape == (3, 4, 4)
    assert np.all(np.isfinite(result.values.real))
    assert np.all(np.isfinite(result.values.imag))
    assert np.allclose(
        result.dynamic_single_site_expectations[0],
        result.initial_single_site_expectations,
    )


def test_model_qpu_two_time_correlator_can_limit_rust_transition_sites():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.two_time_correlator(
        model,
        [0.0, 0.2, 0.4],
        pauli="Z",
        connected=False,
        measure_sites=[0, 2],
        source_sites=[1, 3],
    )

    transition_calls = [call for call in bindings.calls if "reference_state_handle" in call]

    assert result.solver == "tdvp_transition"
    assert result.values.shape == (3, 2, 2)
    assert result.measure_sites == (0, 2)
    assert result.source_sites == (1, 3)
    assert len(transition_calls) == 3
    assert transition_calls[0]["observables"] == ["Z0", "Z2", "Z1", "Z3"]
    assert transition_calls[1]["observables"] == ["Z0", "Z2"]
    assert transition_calls[2]["observables"] == ["Z0", "Z2"]


def test_model_qpu_cross_model_two_time_requires_explicit_initial_state_model():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    final_model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    prepared = qpu.ground_state(initial_model)

    with pytest.raises(ValueError, match="initial backend state was prepared for model"):
        qpu.two_time_correlator(
            final_model,
            [0.0, 0.2],
            pauli="Z",
            initial_state=prepared,
        )

    result = qpu.two_time_correlator(
        final_model,
        [0.0, 0.2],
        pauli="Z",
        initial_state=prepared,
        initial_state_model=initial_model,
    )

    transition_calls = [call for call in bindings.calls if "reference_state_handle" in call]

    assert result.solver == "tdvp_transition"
    assert transition_calls[-1]["model"] == "heisenberg_xyz_1d"
    assert transition_calls[-1]["reference_state_handle"] is prepared.backend_state


def test_model_qpu_entanglement_spectrum_can_use_rust_ground_state_bridge():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    result = qpu.entanglement_spectrum(
        model,
        subsystem=[0, 1],
        num_levels=2,
    )

    entanglement_calls = [call for call in bindings.calls if "bond" in call]

    assert isinstance(result, EntanglementSpectrumResult)
    assert result.solver == "dmrg"
    assert result.subsystem == (0, 1)
    assert np.allclose(result.eigenvalues, [0.8, 0.2])
    assert np.isclose(result.entropy, -(0.8 * np.log2(0.8) + 0.2 * np.log2(0.2)))
    assert len(entanglement_calls) == 1
    assert entanglement_calls[0]["bond"] == 1


def test_model_qpu_entanglement_spectrum_can_use_rust_tdvp_state_handle():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    evolution = qpu.quench(
        model,
        [0.0, 0.2, 0.4],
        initial_state="neel",
        observables=["magnetization_z"],
    )
    result = qpu.entanglement_spectrum(
        model,
        subsystem=[0, 1],
        initial_state=evolution,
    )

    assert result.solver == "tdvp"
    assert np.allclose(result.eigenvalues, [0.8, 0.2])


def test_model_qpu_entanglement_spectrum_requires_supported_rust_cut_when_exact_is_too_small():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    with pytest.raises(ValueError, match="entanglement spectra currently require dense-state analysis"):
        qpu.entanglement_spectrum(model, subsystem=[0, 2])


def test_model_qpu_linear_response_spectrum_can_use_rust_tdvp_transition_bridge():
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=FakeRustBindings()),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    response = qpu.linear_response_spectrum(
        model,
        times=[0.0, 0.2, 0.4],
        momenta=[0.0, np.pi],
        pauli="Z",
        frequencies=[0.0],
        window="none",
    )

    assert response.solver == "tdvp_transition"
    assert response.values.shape == (1, 2)
    assert response.time_response.shape == (3, 2)
    assert np.all(np.isfinite(response.values.real))


def test_rust_tdvp_bridge_receives_initial_state_times_and_method():
    bindings = FakeRustBindings()
    solver = RustTensorNetworkSolver(bindings=bindings, tdvp_method="two_site")
    model = HeisenbergXXZ1D(num_sites=6, coupling_xy=1.5, anisotropy=0.75, field_z=0.2)

    solver.time_evolve(
        model,
        np.array([0.0, 0.1, 0.3]),
        initial_state="101010",
        observables=["magnetization_z"],
        subsystem=[0, 1, 2],
    )

    assert bindings.calls
    call = bindings.calls[0]
    assert call["model"] == "heisenberg_xxz_1d"
    assert call["times"] == [0.0, 0.1, 0.3]
    assert call["initial_state"] == "101010"
    assert call["method"] == "two_site"
    assert call["coupling"] == 1.5
    assert call["anisotropy"] == 0.75
    assert call["field_z"] == 0.2
    assert call["entropy_bond"] == 2


def test_rust_tdvp_bridge_receives_xyz_model_parameters():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    result = qpu.quench(
        model,
        [0.0, 0.2],
        initial_state="plus_x",
        observables=["magnetization_z"],
    )

    call = bindings.calls[-1]

    assert result.solver == "tdvp"
    assert call["model"] == "heisenberg_xyz_1d"
    assert call["coupling_x"] == 1.1
    assert call["coupling_y"] == 0.6
    assert call["coupling_z"] == 1.4
    assert call["field_z"] == 0.15
    assert call["initial_state"] == "++++"


def test_rust_tdvp_can_start_from_prior_ground_state_handle():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8

    initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    final_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=1.1)
    ground = solver.solve_ground_state(initial_model)
    evolution = solver.time_evolve(
        final_model,
        [0.0, 0.2],
        initial_state=ground,
        observables=["magnetization_z"],
    )

    assert evolution.solver == "tdvp"
    assert len(bindings.calls) == 2
    tdvp_call = bindings.calls[1]
    assert "state_handle" in tdvp_call
    assert tdvp_call["state_handle"] is ground.backend_state
    assert "initial_state" not in tdvp_call


def test_model_qpu_cross_model_rust_quench_requires_explicit_initial_state_model():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    final_model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    prepared = qpu.ground_state(initial_model)

    with pytest.raises(ValueError, match="initial backend state was prepared for model"):
        qpu.quench(
            final_model,
            [0.0, 0.2],
            initial_state=prepared,
            observables=["magnetization_z"],
        )

    result = qpu.quench(
        final_model,
        [0.0, 0.2],
        initial_state=prepared,
        initial_state_model=initial_model,
        observables=["magnetization_z"],
    )

    tdvp_call = bindings.calls[-1]

    assert result.solver == "tdvp"
    assert tdvp_call["state_handle"] is prepared.backend_state
    assert tdvp_call["model"] == "heisenberg_xyz_1d"


def test_model_qpu_cross_model_rust_loschmidt_requires_explicit_reference_state_model():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    final_model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    prepared = qpu.ground_state(initial_model)

    with pytest.raises(ValueError, match="reference backend state was prepared for model"):
        qpu.loschmidt_echo(
            final_model,
            [0.0, 0.2],
            initial_state="plus_x",
            reference_state=prepared,
        )

    result = qpu.loschmidt_echo(
        final_model,
        [0.0, 0.2],
        initial_state="plus_x",
        reference_state=prepared,
        reference_state_model=initial_model,
    )

    overlap_call = bindings.calls[-1]

    assert result.solver == "tdvp_overlap"
    assert overlap_call["reference_state_handle"] is prepared.backend_state
    assert overlap_call["model"] == "heisenberg_xyz_1d"


def test_model_qpu_cross_model_linear_response_requires_explicit_initial_state_model():
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    final_model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    prepared = qpu.ground_state(initial_model)

    with pytest.raises(ValueError, match="initial backend state was prepared for model"):
        qpu.linear_response_spectrum(
            final_model,
            times=[0.0, 0.2],
            momenta=[0.0],
            pauli="Z",
            initial_state=prepared,
            frequencies=[0.0],
            window="none",
        )

    response = qpu.linear_response_spectrum(
        final_model,
        times=[0.0, 0.2],
        momenta=[0.0],
        pauli="Z",
        initial_state=prepared,
        initial_state_model=initial_model,
        frequencies=[0.0],
        window="none",
    )

    assert response.solver == "tdvp_transition"
    assert response.values.shape == (1, 1)


def test_tensor_network_checkpoint_round_trip_supports_direct_handle_reuse(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        initial_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        final_model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=1.1)
        ground = solver.solve_ground_state(initial_model)

        checkpoint_path = tmp_path / "ground_state.json"
        save_tensor_network_state(ground, checkpoint_path)
        restored = load_tensor_network_state(checkpoint_path)

        assert restored.num_sites == 4
        assert restored.solver == "dmrg"

        evolution = solver.time_evolve(
            final_model,
            [0.0, 0.2],
            initial_state=restored,
            observables=["magnetization_z"],
        )

        assert evolution.solver == "tdvp"
        assert len(bindings.calls) == 2
        tdvp_call = bindings.calls[1]
        assert tdvp_call["state_handle"] is restored
        assert "initial_state" not in tdvp_call
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_ground_state_result_manifest_round_trip_preserves_metadata(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        result = solver.solve_ground_state(model, observables=["magnetization_z"], subsystem=[0, 1])

        manifest_path = tmp_path / "ground_state_result.json"
        save_ground_state_result(result, manifest_path)
        restored = load_ground_state_result(manifest_path)

        assert restored.model_name == result.model_name
        assert restored.solver == result.solver
        assert restored.model_metadata == result.model_metadata
        assert restored.backend_state is not None
        assert restored.observables == result.observables
        assert np.allclose(restored.eigenvalues, result.eigenvalues)
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_time_evolution_result_manifest_round_trip_preserves_traces(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        result = solver.time_evolve(
            model,
            [0.0, 0.2, 0.4],
            initial_state="neel",
            observables=["magnetization_z", "Z0Z1"],
            subsystem=[0, 1],
        )

        manifest_path = tmp_path / "time_evolution_result.json"
        save_time_evolution_result(result, manifest_path)
        restored = load_time_evolution_result(manifest_path)

        assert restored.model_name == result.model_name
        assert restored.solver == result.solver
        assert restored.model_metadata == result.model_metadata
        assert restored.entanglement_subsystem == result.entanglement_subsystem
        assert np.allclose(restored.times, result.times)
        assert np.allclose(
            restored.observables["magnetization_z"],
            result.observables["magnetization_z"],
        )
        assert np.allclose(restored.entanglement_entropy, result.entanglement_entropy)
        assert restored.backend_state is not None
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_dynamic_and_frequency_structure_factor_manifests_round_trip(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    dynamic = qpu.dynamic_structure_factor(
        model,
        [0.0, 0.2, 0.4],
        [0.0, np.pi],
        pauli="Z",
        connected=True,
        initial_state="neel",
    )
    spectral = fourier_transform_structure_factor(
        dynamic,
        frequencies=[0.0],
        window="none",
        subtract_mean=True,
    )

    dynamic_path = tmp_path / "dynamic_structure_factor.json"
    spectral_path = tmp_path / "frequency_structure_factor.json"
    save_dynamic_structure_factor_result(dynamic, dynamic_path)
    save_frequency_structure_factor_result(spectral, spectral_path)

    restored_dynamic = load_dynamic_structure_factor_result(dynamic_path)
    restored_spectral = load_frequency_structure_factor_result(spectral_path)

    assert restored_dynamic.model_name == dynamic.model_name
    assert restored_dynamic.solver == dynamic.solver
    assert restored_dynamic.pauli == dynamic.pauli
    assert restored_dynamic.connected == dynamic.connected
    assert np.allclose(restored_dynamic.times, dynamic.times)
    assert np.allclose(restored_dynamic.momenta, dynamic.momenta)
    assert np.allclose(restored_dynamic.values, dynamic.values)

    assert restored_spectral.model_name == spectral.model_name
    assert restored_spectral.solver == spectral.solver
    assert restored_spectral.window == spectral.window
    assert restored_spectral.subtract_mean == spectral.subtract_mean
    assert np.allclose(restored_spectral.frequencies, spectral.frequencies)
    assert np.allclose(restored_spectral.momenta, spectral.momenta)
    assert np.allclose(restored_spectral.values, spectral.values)


def test_entanglement_spectrum_manifest_round_trip_preserves_gap_metadata(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    spectrum = qpu.entanglement_spectrum(model, subsystem=[0, 1], num_levels=3)

    manifest_path = tmp_path / "entanglement_spectrum.json"
    save_entanglement_spectrum_result(spectrum, manifest_path)
    restored = load_entanglement_spectrum_result(manifest_path)

    assert restored.model_name == spectrum.model_name
    assert restored.solver == spectrum.solver
    assert restored.num_sites == spectrum.num_sites
    assert restored.subsystem == spectrum.subsystem
    assert np.allclose(restored.eigenvalues, spectrum.eigenvalues)
    assert np.isclose(restored.entropy, spectrum.entropy)
    assert np.isclose(restored.entanglement_gap, spectrum.entanglement_gap)


def test_loschmidt_echo_manifest_round_trip_preserves_complex_amplitudes(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        result = qpu.loschmidt_echo(model, [0.0, 0.2, 0.4], initial_state="neel")

        manifest_path = tmp_path / "loschmidt_echo.json"
        save_loschmidt_echo_result(result, manifest_path)
        restored = load_loschmidt_echo_result(manifest_path)

        assert restored.model_name == result.model_name
        assert restored.solver == result.solver
        assert restored.num_sites == result.num_sites
        assert np.allclose(restored.times, result.times)
        assert np.allclose(restored.amplitudes, result.amplitudes)
        assert np.allclose(restored.echo, result.echo)
        assert np.allclose(restored.return_rate, result.return_rate)
        assert restored.backend_state is not None
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_dqpt_diagnostics_manifest_round_trip_preserves_candidates_and_backend_state(tmp_path):
    bindings = FakeRustBindings()

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        result = DQPTDiagnosticsResult(
            model_name="synthetic",
            solver="tdvp_overlap",
            num_sites=4,
            times=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            return_rate=np.asarray([0.1, 0.8, 0.2], dtype=np.float64),
            candidates=(
                DQPTCandidate(
                    index=1,
                    time=0.1,
                    return_rate=0.8,
                    prominence=0.6,
                    cusp_strength=13.0,
                    left_slope=7.0,
                    right_slope=-6.0,
                ),
            ),
            amplitudes=np.asarray([1.0 + 0.0j, 0.8 - 0.1j, 0.6 - 0.2j], dtype=np.complex128),
            backend_state=FakeTensorNetworkStateHandle(
                4,
                "tdvp",
                model_name="heisenberg_xyz_1d",
            ),
            model_metadata={"model_name": "heisenberg_xyz_1d"},
        )

        manifest_path = tmp_path / "dqpt_diagnostics.json"
        save_dqpt_diagnostics_result(result, manifest_path)
        restored = load_dqpt_diagnostics_result(manifest_path)

        assert restored.model_name == result.model_name
        assert restored.solver == result.solver
        assert restored.num_sites == result.num_sites
        assert np.allclose(restored.times, result.times)
        assert np.allclose(restored.return_rate, result.return_rate)
        assert np.allclose(restored.amplitudes, result.amplitudes)
        assert np.allclose(restored.candidate_times, result.candidate_times)
        assert restored.candidates[0].prominence == pytest.approx(0.6)
        assert restored.backend_state is not None
        assert restored.backend_state.model_name == "heisenberg_xyz_1d"
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_dqpt_scan_manifest_round_trip_preserves_summary_and_backend_sidecars(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)
    model = HeisenbergXYZ1D(
        num_sites=4,
        coupling_x=1.1,
        coupling_y=0.6,
        coupling_z=1.4,
        field_z=0.15,
    )

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        scan = qpu.scan_dqpt_parameter(
            model,
            "field_z",
            [0.05, 0.15],
            [0.0, 0.2, 0.4],
            initial_state="plus_x",
        )

        manifest_path = tmp_path / "dqpt_scan.json"
        save_dqpt_scan_result(scan, manifest_path)
        restored = load_dqpt_scan_result(manifest_path)

        assert restored.parameter == scan.parameter
        assert restored.model_name == scan.model_name
        assert restored.solver == scan.solver
        assert np.allclose(restored.values, scan.values)
        assert np.allclose(restored.times, scan.times)
        assert np.allclose(restored.return_rates, scan.return_rates)
        assert np.allclose(restored.max_return_rates, scan.max_return_rates)
        assert np.allclose(restored.candidate_counts, scan.candidate_counts, equal_nan=True)
        assert restored.completed_points == scan.completed_points
        assert len(restored.points) == len(scan.points)
        assert restored.points[0].backend_state is not None
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_adaptive_dqpt_scan_manifest_round_trip_preserves_refinement_metadata(tmp_path):
    qpu = SyntheticAdaptiveDQPTQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    scan = qpu.adaptive_scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 0.1, 0.2],
        metric="strongest_candidate_time",
        strategy="target_crossing",
        target_value=0.8,
        max_refinement_rounds=1,
    )

    manifest_path = tmp_path / "adaptive_dqpt_scan.json"
    save_dqpt_scan_result(scan, manifest_path)
    restored = load_dqpt_scan_result(manifest_path)

    assert restored.refinement_metric == "strongest_candidate_time"
    assert restored.refinement_strategy == "target_crossing"
    assert np.isclose(restored.refinement_target_value, 0.8)
    assert restored.insertion_policy == "target_linear"
    assert restored.points_per_interval == 1
    assert restored.completed_rounds == 1
    assert np.allclose(restored.seed_values, [0.0, 0.5, 1.0, 1.5])
    assert restored.refinement_history == scan.refinement_history


def test_two_time_correlator_manifest_round_trip_preserves_subset_metadata(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    correlator = qpu.two_time_correlator(
        model,
        [0.0, 0.2, 0.4],
        pauli="Z",
        connected=True,
        measure_sites=[0, 2],
        source_sites=[1, 3],
    )

    manifest_path = tmp_path / "two_time_correlator.json"
    save_two_time_correlator_result(correlator, manifest_path)
    restored = load_two_time_correlator_result(manifest_path)

    assert restored.model_name == correlator.model_name
    assert restored.solver == correlator.solver
    assert restored.connected == correlator.connected
    assert restored.num_sites == correlator.num_sites
    assert restored.measure_sites == correlator.measure_sites
    assert restored.source_sites == correlator.source_sites
    assert np.allclose(restored.times, correlator.times)
    assert np.allclose(restored.values, correlator.values)
    assert np.allclose(
        restored.dynamic_single_site_expectations,
        correlator.dynamic_single_site_expectations,
    )
    assert np.allclose(
        restored.initial_single_site_expectations,
        correlator.initial_single_site_expectations,
    )


def test_response_spectrum_manifest_round_trip_preserves_complex_response(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    response = qpu.linear_response_spectrum(
        model,
        times=[0.0, 0.2, 0.4],
        momenta=[0.0, np.pi],
        pauli="Z",
        source_sites=[0, 2],
        frequencies=[0.0],
        window="none",
    )

    manifest_path = tmp_path / "response_spectrum.json"
    save_response_spectrum_result(response, manifest_path)
    restored = load_response_spectrum_result(manifest_path)

    assert restored.model_name == response.model_name
    assert restored.solver == response.solver
    assert restored.window == response.window
    assert restored.subtract_mean == response.subtract_mean
    assert restored.num_sites == response.num_sites
    assert restored.measure_sites == response.measure_sites
    assert restored.source_sites == response.source_sites
    assert np.allclose(restored.times, response.times)
    assert np.allclose(restored.frequencies, response.frequencies)
    assert np.allclose(restored.momenta, response.momenta)
    assert np.allclose(restored.time_response, response.time_response)
    assert np.allclose(restored.values, response.values)


def test_sweep_result_manifest_round_trip_preserves_phase_data(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    sweep = qpu.sweep_parameter(
        model,
        "transverse_field",
        np.linspace(0.5, 0.9, 3),
        observables=["magnetization_z", "Z0Z1"],
        subsystem=[0, 1],
    )

    manifest_path = tmp_path / "sweep_result.json"
    save_sweep_result(sweep, manifest_path)
    restored = load_sweep_result(manifest_path)

    assert restored.parameter == sweep.parameter
    assert restored.model_name == sweep.model_name
    assert restored.solver == sweep.solver
    assert restored.base_model_metadata == sweep.base_model_metadata
    assert restored.observables_requested == sweep.observables_requested
    assert restored.subsystem == sweep.subsystem
    assert restored.completed_points == sweep.completed_points
    assert restored.is_complete
    assert np.allclose(restored.values, sweep.values)
    assert np.allclose(restored.energies, sweep.energies)
    assert np.allclose(restored.spectral_gaps, sweep.spectral_gaps)
    assert np.allclose(restored.entanglement_entropy, sweep.entanglement_entropy)
    assert np.allclose(restored.observables["magnetization_z"], sweep.observables["magnetization_z"])
    assert restored.points[0].model_metadata == sweep.points[0].model_metadata
    assert restored.points[-1].solver == sweep.points[-1].solver


def test_adaptive_sweep_manifest_round_trip_preserves_refinement_metadata(tmp_path):
    qpu = ModelQPU(solver=SyntheticAdaptiveSolver())
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)

    sweep = qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.25, 0.75, 1.25, 1.75],
        metric="order_parameter",
        strategy="target_crossing",
        target_value=0.9,
        insertion_policy="target_linear",
        max_refinement_rounds=1,
        refinements_per_round=1,
    )

    manifest_path = tmp_path / "adaptive_sweep.json"
    save_sweep_result(sweep, manifest_path)
    restored = load_sweep_result(manifest_path)

    assert restored.refinement_metric == "order_parameter"
    assert restored.refinement_strategy == "target_crossing"
    assert np.isclose(restored.refinement_target_value, 0.9)
    assert restored.insertion_policy == "target_linear"
    assert restored.points_per_interval == 1
    assert np.allclose(restored.seed_values, [0.25, 0.75, 1.25, 1.75])
    assert restored.observables_requested == ("order_parameter",)
    assert restored.completed_rounds == 1
    assert restored.refinement_history == sweep.refinement_history
    assert restored.points[0].index is not None


def test_rust_backed_sweep_manifest_round_trip_restores_point_backend_states(tmp_path):
    bindings = FakeRustBindings()
    solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=bindings),
    )
    solver.exact_solver.max_dimension = 8
    qpu = ModelQPU(solver=solver)

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: bindings
    try:
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        sweep = qpu.sweep_parameter(
            model,
            "transverse_field",
            np.array([0.5, 0.7], dtype=np.float64),
            observables=["magnetization_z"],
            subsystem=[0, 1],
        )

        manifest_path = tmp_path / "rust_sweep.json"
        save_sweep_result(sweep, manifest_path)
        restored = load_sweep_result(manifest_path)

        assert bindings.calls[1]["state_handle"] is sweep.points[0].backend_state
        assert restored.points[0].backend_state is not None
        assert restored.points[1].backend_state is not None
        assert (tmp_path / "rust_sweep.points" / "point_000000.state.json").exists()
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_dqpt_scan_can_resume_from_partial_checkpoint(tmp_path):
    class InterruptingDQPTQPU(ModelQPU):
        def __init__(self, *, fail_on_call: int | None = None):
            super().__init__()
            self.fail_on_call = fail_on_call
            self.calls: list[float] = []

        def dqpt_diagnostics(self, model, times, **kwargs):
            del kwargs
            sweep_value = float(getattr(model, "transverse_field", 0.0))
            self.calls.append(sweep_value)
            if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
                raise RuntimeError("interrupted dqpt scan")

            return DQPTDiagnosticsResult(
                model_name=model.model_name,
                solver="fake_dqpt",
                num_sites=model.num_sites,
                times=np.asarray(list(times), dtype=np.float64),
                return_rate=np.asarray(
                    [0.1 + sweep_value, 0.3 + sweep_value, 0.2 + sweep_value],
                    dtype=np.float64,
                ),
                candidates=(
                    DQPTCandidate(
                        index=1,
                        time=0.1,
                        return_rate=0.3 + sweep_value,
                        prominence=0.1,
                        cusp_strength=2.0 + sweep_value,
                        left_slope=1.0,
                        right_slope=-1.0,
                    ),
                ),
                amplitudes=np.asarray(
                    [1.0 + 0.0j, 0.9 - 0.1j, 0.8 - 0.2j],
                    dtype=np.complex128,
                ),
                model_metadata={
                    "model_name": model.model_name,
                    "num_sites": model.num_sites,
                    "transverse_field": sweep_value,
                },
            )

    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    values = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    checkpoint_path = tmp_path / "resume_dqpt_scan.json"

    interrupted_qpu = InterruptingDQPTQPU(fail_on_call=3)
    with pytest.raises(RuntimeError, match="interrupted dqpt scan"):
        interrupted_qpu.scan_dqpt_parameter(
            model,
            "transverse_field",
            values,
            [0.0, 0.1, 0.2],
            checkpoint_path=checkpoint_path,
        )

    partial = load_dqpt_scan_result(checkpoint_path)
    assert partial.completed_points == 2
    assert np.allclose(partial.values[:2], [0.5, 0.6])

    resumed_qpu = InterruptingDQPTQPU()
    restored = resumed_qpu.scan_dqpt_parameter(
        model,
        "transverse_field",
        values,
        [0.0, 0.1, 0.2],
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed_qpu.calls == [0.7, 0.8]
    assert restored.completed_points == 4
    assert restored.is_complete


def test_sweep_parameter_can_resume_from_partial_checkpoint(tmp_path):
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    values = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    checkpoint_path = tmp_path / "nested" / "resume_sweep.json"

    interrupted_qpu = ModelQPU(solver=InterruptingSweepSolver(fail_on_call=3))
    with pytest.raises(RuntimeError, match="interrupted sweep"):
        interrupted_qpu.sweep_parameter(
            model,
            "transverse_field",
            values,
            observables=["magnetization_z"],
            subsystem=[0, 1],
            checkpoint_path=checkpoint_path,
        )

    partial = load_sweep_result(checkpoint_path)
    assert partial.completed_points == 2
    assert not partial.is_complete
    assert np.allclose(partial.values, values)
    assert np.allclose(partial.energies[:2], [-0.5, -0.6])
    assert np.isnan(partial.energies[2])
    assert np.isnan(partial.observables["magnetization_z"][3])

    resumed_solver = InterruptingSweepSolver()
    resumed_qpu = ModelQPU(solver=resumed_solver)
    restored = resumed_qpu.sweep_parameter(
        model,
        "transverse_field",
        values,
        observables=["magnetization_z"],
        subsystem=[0, 1],
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed_solver.calls == [0.7, 0.8]
    assert restored.completed_points == 4
    assert restored.is_complete
    assert np.allclose(restored.energies, [-0.5, -0.6, -0.7, -0.8])
    assert np.allclose(restored.spectral_gaps, [1.5, 1.6, 1.7, 1.8])
    assert np.allclose(restored.entanglement_entropy, [0.25, 0.25, 0.25, 0.25])


def test_sweep_resume_rejects_mismatched_checkpoint(tmp_path):
    qpu = ModelQPU()
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    checkpoint_path = tmp_path / "resume_mismatch.json"

    qpu.sweep_parameter(
        model,
        "transverse_field",
        [0.5, 0.7],
        observables=["magnetization_z"],
        checkpoint_path=checkpoint_path,
    )

    with pytest.raises(ValueError, match="checkpoint values do not match"):
        qpu.sweep_parameter(
            model,
            "transverse_field",
            [0.5, 0.6, 0.7],
            observables=["magnetization_z"],
            checkpoint_path=checkpoint_path,
            resume=True,
        )


def test_rust_backed_sweep_resume_uses_restored_backend_state_warm_start(tmp_path):
    initial_bindings = FakeRustBindings()
    initial_solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=initial_bindings),
    )
    initial_solver.exact_solver.max_dimension = 8
    initial_qpu = ModelQPU(solver=initial_solver)

    resume_bindings = FakeRustBindings()
    resumed_solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=resume_bindings),
    )
    resumed_solver.exact_solver.max_dimension = 8
    resumed_qpu = ModelQPU(solver=resumed_solver)

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: resume_bindings
    try:
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        first_value = 0.5
        first_model = replace(model, transverse_field=first_value)
        first_result = initial_qpu.ground_state(
            first_model,
            observables=["magnetization_z"],
            subsystem=[0, 1],
        )

        partial = SweepResult(
            parameter="transverse_field",
            model_name=model.model_name,
            solver=first_result.solver,
            values=np.asarray([0.5, 0.7], dtype=np.float64),
            energies=np.asarray([first_result.ground_state_energy, np.nan], dtype=np.float64),
            observables={
                "magnetization_z": np.asarray(
                    [first_result.observables["magnetization_z"], np.nan],
                    dtype=np.float64,
                )
            },
            points=[
                SweepPoint(
                    value=first_value,
                    energy=first_result.ground_state_energy,
                    observables=dict(first_result.observables),
                    solver=first_result.solver,
                    spectral_gap=first_result.spectral_gap,
                    entanglement_entropy=first_result.entanglement_entropy,
                    model_metadata=first_result.model_metadata,
                    backend_state=first_result.backend_state,
                )
            ],
            base_model_metadata={
                "num_sites": model.num_sites,
                "coupling": model.coupling,
                "transverse_field": model.transverse_field,
                "longitudinal_field": model.longitudinal_field,
                "boundary": model.boundary,
                "model_name": model.model_name,
            },
            observables_requested=("magnetization_z",),
            subsystem=(0, 1),
            spectral_gaps=np.asarray([np.nan, np.nan], dtype=np.float64),
            entanglement_entropy=np.asarray([0.5, np.nan], dtype=np.float64),
            completed_points=1,
        )

        checkpoint_path = tmp_path / "rust_resume.json"
        save_sweep_result(partial, checkpoint_path)
        restored = resumed_qpu.sweep_parameter(
            model,
            "transverse_field",
            [0.5, 0.7],
            observables=["magnetization_z"],
            subsystem=[0, 1],
            checkpoint_path=checkpoint_path,
            resume=True,
        )

        assert len(resume_bindings.calls) == 1
        call = resume_bindings.calls[0]
        assert "state_handle" in call
        assert isinstance(call["state_handle"], FakeTensorNetworkStateHandle)
        assert call["state_handle"].num_sites == 4
        assert call["state_handle"].solver == "dmrg"
        assert restored.completed_points == 2
        assert restored.is_complete
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_rust_backed_sweep_resume_prefers_nearest_backend_state_order(tmp_path):
    resume_bindings = FakeRustBindings()
    resumed_solver = AutoSolver(
        exact_solver=None,
        tensor_network_solver=RustTensorNetworkSolver(bindings=resume_bindings),
    )
    resumed_solver.exact_solver.max_dimension = 8
    resumed_qpu = ModelQPU(solver=resumed_solver)

    original_get_rust_bindings = state_io.get_rust_bindings
    state_io.get_rust_bindings = lambda: resume_bindings
    try:
        model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
        far_right_state = FakeTensorNetworkStateHandle(
            4,
            "dmrg",
            model_name=model.model_name,
        )
        partial = SweepResult(
            parameter="transverse_field",
            model_name=model.model_name,
            solver="dmrg",
            values=np.asarray([0.5, 0.7, 0.9], dtype=np.float64),
            energies=np.asarray([np.nan, np.nan, -5.25], dtype=np.float64),
            observables={
                "magnetization_z": np.asarray([np.nan, np.nan, 0.125], dtype=np.float64)
            },
            points=[
                SweepPoint(
                    value=0.9,
                    energy=-5.25,
                    observables={"magnetization_z": 0.125},
                    solver="dmrg",
                    entanglement_entropy=0.5,
                    model_metadata={
                        "model_name": model.model_name,
                        "num_sites": model.num_sites,
                        "transverse_field": 0.9,
                    },
                    backend_state=far_right_state,
                    index=2,
                )
            ],
            base_model_metadata={
                "num_sites": model.num_sites,
                "coupling": model.coupling,
                "transverse_field": model.transverse_field,
                "longitudinal_field": model.longitudinal_field,
                "boundary": model.boundary,
                "model_name": model.model_name,
            },
            observables_requested=("magnetization_z",),
            subsystem=(0, 1),
            spectral_gaps=np.asarray([np.nan, np.nan, np.nan], dtype=np.float64),
            entanglement_entropy=np.asarray([np.nan, np.nan, 0.5], dtype=np.float64),
            completed_points=1,
        )

        checkpoint_path = tmp_path / "warm_start_schedule.json"
        save_sweep_result(partial, checkpoint_path)
        restored = resumed_qpu.sweep_parameter(
            model,
            "transverse_field",
            [0.5, 0.7, 0.9],
            observables=["magnetization_z"],
            subsystem=[0, 1],
            checkpoint_path=checkpoint_path,
            resume=True,
        )

        assert [call["transverse_field"] for call in resume_bindings.calls] == [0.7, 0.5]
        assert restored.completed_points == 3
        assert restored.is_complete
    finally:
        state_io.get_rust_bindings = original_get_rust_bindings


def test_adaptive_sweep_can_resume_after_grid_expansion(tmp_path):
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    checkpoint_path = tmp_path / "adaptive_resume.json"

    interrupted_qpu = ModelQPU(solver=InterruptingSweepSolver(fail_on_call=4))
    with pytest.raises(RuntimeError, match="interrupted sweep"):
        interrupted_qpu.adaptive_sweep_parameter(
            model,
            "transverse_field",
            [0.5, 1.0, 1.5],
            metric="spectral_gap",
            max_refinement_rounds=1,
            checkpoint_path=checkpoint_path,
        )

    partial = load_sweep_result(checkpoint_path)
    assert partial.completed_points == 3
    assert not partial.is_complete
    assert partial.completed_rounds == 1
    assert np.allclose(partial.seed_values, [0.5, 1.0, 1.5])
    assert np.allclose(partial.values, [0.5, 0.75, 1.0, 1.5])
    assert np.isnan(partial.energies[1])
    assert [point.index for point in partial.points] == [0, 2, 3]

    resumed_solver = InterruptingSweepSolver()
    resumed_qpu = ModelQPU(solver=resumed_solver)
    restored = resumed_qpu.adaptive_sweep_parameter(
        model,
        "transverse_field",
        [0.5, 1.0, 1.5],
        metric="spectral_gap",
        max_refinement_rounds=1,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed_solver.calls == [0.75]
    assert restored.completed_points == 4
    assert restored.is_complete
    assert restored.completed_rounds == 1
    assert np.allclose(restored.values, [0.5, 0.75, 1.0, 1.5])
    assert np.allclose(restored.energies, [-0.5, -0.75, -1.0, -1.5])


def test_adaptive_dqpt_scan_can_resume_after_grid_expansion(tmp_path):
    model = TransverseFieldIsing1D(num_sites=4, coupling=1.0, transverse_field=0.7)
    checkpoint_path = tmp_path / "adaptive_dqpt_resume.json"

    interrupted_qpu = SyntheticAdaptiveDQPTQPU(fail_on_call=4)
    with pytest.raises(RuntimeError, match="interrupted adaptive dqpt scan"):
        interrupted_qpu.adaptive_scan_dqpt_parameter(
            model,
            "transverse_field",
            [0.5, 1.0, 1.5],
            [0.0, 0.1, 0.2],
            metric="strongest_cusp_strength",
            strategy="curvature",
            max_refinement_rounds=1,
            checkpoint_path=checkpoint_path,
        )

    partial = load_dqpt_scan_result(checkpoint_path)
    assert partial.completed_points == 3
    assert not partial.is_complete
    assert partial.completed_rounds == 1
    assert np.allclose(partial.seed_values, [0.5, 1.0, 1.5])
    assert np.allclose(partial.values, [0.5, 0.75, 1.0, 1.5])
    assert np.isnan(partial.return_rates[1]).all()
    assert [point.index for point in partial.points] == [0, 2, 3]

    resumed_qpu = SyntheticAdaptiveDQPTQPU()
    restored = resumed_qpu.adaptive_scan_dqpt_parameter(
        model,
        "transverse_field",
        [0.5, 1.0, 1.5],
        [0.0, 0.1, 0.2],
        metric="strongest_cusp_strength",
        strategy="curvature",
        max_refinement_rounds=1,
        checkpoint_path=checkpoint_path,
        resume=True,
    )

    assert resumed_qpu.calls == [0.75]
    assert restored.completed_points == 4
    assert restored.is_complete
    assert restored.completed_rounds == 1
    assert np.allclose(restored.values, [0.5, 0.75, 1.0, 1.5])
