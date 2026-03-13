"""Experiment workflows for the model-QPU research API."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import HeisenbergXXZ1D, HeisenbergXYZ1D, TransverseFieldIsing1D, pauli_string_operator
from .solvers import (
    AutoSolver,
    ExactDiagonalizationSolver,
    GroundStateResult,
    InitialStateLike,
    RustTensorNetworkSolver,
    TimeEvolutionResult,
    entanglement_spectrum as pure_state_entanglement_spectrum,
    make_initial_state,
)


@dataclass
class SweepPoint:
    value: float
    energy: float
    observables: dict[str, float]
    solver: str
    spectral_gap: float | None = None
    entanglement_entropy: float | None = None
    model_metadata: dict[str, object] | None = None
    backend_state: object | None = None
    backend_state_checkpoint: str | None = None
    index: int | None = None


@dataclass
class SweepResult:
    parameter: str
    model_name: str
    solver: str
    values: np.ndarray
    energies: np.ndarray
    observables: dict[str, np.ndarray]
    points: list[SweepPoint]
    base_model_metadata: dict[str, object] | None = None
    observables_requested: tuple[str, ...] = ()
    subsystem: tuple[int, ...] | None = None
    spectral_gaps: np.ndarray | None = None
    entanglement_entropy: np.ndarray | None = None
    completed_points: int = 0
    seed_values: np.ndarray | None = None
    refinement_metric: str | None = None
    refinement_strategy: str | None = None
    refinement_target_value: float | None = None
    insertion_policy: str | None = None
    points_per_interval: int = 1
    max_refinement_rounds: int = 0
    refinements_per_round: int = 0
    min_spacing: float | None = None
    refinement_history: list[dict[str, object]] | None = None

    @property
    def is_complete(self) -> bool:
        return self.completed_points >= len(self.values)

    @property
    def completed_rounds(self) -> int:
        return len(self.refinement_history or [])


@dataclass
class CorrelationMatrixResult:
    model_name: str
    solver: str
    pauli: str
    connected: bool
    matrix: np.ndarray
    single_site_expectations: np.ndarray
    model_metadata: dict[str, object] | None = None


@dataclass
class StructureFactorResult:
    model_name: str
    solver: str
    pauli: str
    connected: bool
    momenta: np.ndarray
    values: np.ndarray
    correlation_matrix: np.ndarray | None = None
    single_site_expectations: np.ndarray | None = None
    model_metadata: dict[str, object] | None = None


@dataclass
class DynamicStructureFactorResult:
    model_name: str
    solver: str
    pauli: str
    connected: bool
    times: np.ndarray
    momenta: np.ndarray
    values: np.ndarray
    model_metadata: dict[str, object] | None = None


@dataclass
class FrequencyStructureFactorResult:
    model_name: str
    solver: str
    pauli: str
    connected: bool
    times: np.ndarray
    frequencies: np.ndarray
    momenta: np.ndarray
    values: np.ndarray
    window: str
    subtract_mean: bool
    model_metadata: dict[str, object] | None = None

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.values)


@dataclass
class TwoTimeCorrelatorResult:
    model_name: str
    solver: str
    pauli: str
    connected: bool
    num_sites: int
    times: np.ndarray
    values: np.ndarray
    dynamic_single_site_expectations: np.ndarray
    initial_single_site_expectations: np.ndarray
    measure_sites: tuple[int, ...]
    source_sites: tuple[int, ...]
    model_metadata: dict[str, object] | None = None


@dataclass
class ResponseSpectrumResult:
    model_name: str
    solver: str
    pauli: str
    num_sites: int
    times: np.ndarray
    frequencies: np.ndarray
    momenta: np.ndarray
    time_response: np.ndarray
    values: np.ndarray
    measure_sites: tuple[int, ...]
    source_sites: tuple[int, ...]
    window: str
    subtract_mean: bool
    model_metadata: dict[str, object] | None = None

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.values)


@dataclass
class EntanglementSpectrumResult:
    model_name: str
    solver: str
    num_sites: int
    subsystem: tuple[int, ...]
    eigenvalues: np.ndarray
    entropy: float
    model_metadata: dict[str, object] | None = None

    @property
    def schmidt_values(self) -> np.ndarray:
        return np.sqrt(np.clip(self.eigenvalues, 0.0, None))

    @property
    def entanglement_energies(self) -> np.ndarray:
        values = np.asarray(self.eigenvalues, dtype=np.float64)
        energies = np.full(values.shape, np.inf, dtype=np.float64)
        mask = values > 1e-300
        energies[mask] = -np.log(values[mask])
        return energies

    @property
    def entanglement_gap(self) -> float | None:
        energies = self.entanglement_energies
        finite = energies[np.isfinite(energies)]
        if finite.size < 2:
            return None
        return float(finite[1] - finite[0])


@dataclass
class LoschmidtEchoResult:
    model_name: str
    solver: str
    num_sites: int
    times: np.ndarray
    amplitudes: np.ndarray
    backend_state: object | None = None
    model_metadata: dict[str, object] | None = None

    @property
    def echo(self) -> np.ndarray:
        amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        return np.abs(amplitudes) ** 2

    @property
    def return_rate(self) -> np.ndarray:
        echo = np.clip(self.echo, 1e-300, None)
        return -np.log(echo) / float(self.num_sites)


@dataclass(frozen=True)
class DQPTCandidate:
    index: int
    time: float
    return_rate: float
    prominence: float
    cusp_strength: float
    left_slope: float
    right_slope: float


@dataclass
class DQPTDiagnosticsResult:
    model_name: str
    solver: str
    num_sites: int
    times: np.ndarray
    return_rate: np.ndarray
    candidates: tuple[DQPTCandidate, ...]
    amplitudes: np.ndarray | None = None
    backend_state: object | None = None
    model_metadata: dict[str, object] | None = None

    @property
    def candidate_times(self) -> np.ndarray:
        return np.asarray([candidate.time for candidate in self.candidates], dtype=np.float64)


@dataclass
class DQPTScanPoint:
    value: float
    solver: str
    return_rate: np.ndarray
    candidate_times: np.ndarray
    candidate_return_rates: np.ndarray
    candidate_prominences: np.ndarray
    candidate_cusp_strengths: np.ndarray
    strongest_candidate_time: float | None = None
    strongest_candidate_return_rate: float | None = None
    strongest_prominence: float | None = None
    strongest_cusp_strength: float | None = None
    model_metadata: dict[str, object] | None = None
    backend_state: object | None = None
    backend_state_checkpoint: str | None = None
    index: int | None = None

    @property
    def num_candidates(self) -> int:
        return int(np.asarray(self.candidate_times, dtype=np.float64).size)

    @property
    def max_return_rate(self) -> float:
        return float(np.max(np.asarray(self.return_rate, dtype=np.float64)))


@dataclass
class DQPTScanResult:
    parameter: str
    model_name: str
    solver: str
    values: np.ndarray
    times: np.ndarray
    return_rates: np.ndarray
    candidate_counts: np.ndarray
    max_return_rates: np.ndarray
    strongest_candidate_times: np.ndarray
    strongest_candidate_return_rates: np.ndarray
    strongest_prominences: np.ndarray
    strongest_cusp_strengths: np.ndarray
    points: list[DQPTScanPoint]
    base_model_metadata: dict[str, object] | None = None
    initial_state_descriptor: dict[str, object] | None = None
    reference_state_descriptor: dict[str, object] | None = None
    initial_state_model_metadata: dict[str, object] | None = None
    reference_state_model_metadata: dict[str, object] | None = None
    min_prominence: float = 0.0
    min_cusp: float = 0.0
    mode: str = "maxima"
    completed_points: int = 0
    seed_values: np.ndarray | None = None
    refinement_metric: str | None = None
    refinement_strategy: str | None = None
    refinement_target_value: float | None = None
    insertion_policy: str | None = None
    points_per_interval: int = 1
    max_refinement_rounds: int = 0
    refinements_per_round: int = 0
    min_spacing: float | None = None
    refinement_history: list[dict[str, object]] | None = None

    @property
    def is_complete(self) -> bool:
        return self.completed_points >= len(self.values)

    @property
    def completed_rounds(self) -> int:
        return len(self.refinement_history or [])


class ModelQPU:
    """Research-oriented front end for model Hamiltonians and sweeps."""

    def __init__(self, solver: object | None = None):
        self.solver = solver or AutoSolver()

    def ground_state(
        self,
        model: object,
        observables: Iterable[str] = (),
        *,
        initial_state: InitialStateLike | None = None,
        subsystem: Iterable[int] | None = None,
    ) -> GroundStateResult:
        if initial_state is None:
            return self.solver.solve_ground_state(
                model,
                observables=observables,
                subsystem=subsystem,
            )
        return self.solver.solve_ground_state(
            model,
            observables=observables,
            initial_state=initial_state,
            subsystem=subsystem,
        )

    def spectrum(self, model: object, num_eigenvalues: int | None = None) -> np.ndarray:
        return self.solver.spectrum(model, num_eigenvalues=num_eigenvalues)

    def quench(
        self,
        model: object,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike = "all_up",
        initial_state_model: object | None = None,
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
    ) -> TimeEvolutionResult:
        _validate_prepared_state_model(
            target_model=model,
            state=initial_state,
            state_model=initial_state_model,
            role="initial",
        )
        return self.solver.time_evolve(
            model,
            times,
            initial_state=initial_state,
            observables=observables,
            subsystem=subsystem,
        )

    def loschmidt_echo(
        self,
        model: object,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike | None = "all_up",
        initial_state_model: object | None = None,
        reference_state: InitialStateLike | None = None,
        reference_state_model: object | None = None,
    ) -> LoschmidtEchoResult:
        _validate_prepared_state_model(
            target_model=model,
            state=initial_state,
            state_model=initial_state_model,
            role="initial",
        )
        _validate_prepared_state_model(
            target_model=model,
            state=reference_state,
            state_model=reference_state_model,
            role="reference",
        )
        if _should_use_rust_loschmidt_echo(
            solver=self.solver,
            model=model,
            times=times,
            initial_state=initial_state,
            reference_state=reference_state,
        ):
            tensor_solver = _resolve_tensor_network_solver(self.solver)
            assert tensor_solver is not None
            return _rust_loschmidt_echo(
                self,
                model,
                times,
                initial_state=initial_state,
                reference_state=reference_state,
                tensor_solver=tensor_solver,
            )

        exact_solver = _resolve_exact_solver(self.solver)
        dimension = _model_dimension(model)
        if dimension > exact_solver.max_dimension:
            raise ValueError(
                "Loschmidt echo currently requires exact diagonalization, "
                f"limited to dimension {exact_solver.max_dimension}, got {dimension}"
            )

        times_array = _normalize_correlation_times(times)
        hamiltonian = np.asarray(model.hamiltonian(), dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

        ket = _resolve_exact_loschmidt_state(
            model,
            state=initial_state,
            max_dimension=exact_solver.max_dimension,
            role="initial",
        )
        bra = (
            ket
            if reference_state is None
            else _resolve_exact_loschmidt_state(
                model,
                state=reference_state,
                max_dimension=exact_solver.max_dimension,
                role="reference",
            )
        )

        ket_coefficients = eigenvectors.conj().T @ ket
        bra_coefficients = eigenvectors.conj().T @ bra
        amplitudes = np.zeros(times_array.shape, dtype=np.complex128)
        for index, time in enumerate(times_array):
            phases = np.exp(-1.0j * eigenvalues * time)
            amplitudes[index] = np.vdot(bra_coefficients, phases * ket_coefficients)

        return LoschmidtEchoResult(
            model_name=getattr(model, "model_name", type(model).__name__),
            solver="exact_diagonalization",
            num_sites=getattr(model, "num_sites"),
            times=times_array,
            amplitudes=amplitudes,
            backend_state=None,
            model_metadata=_model_metadata(model),
        )

    def dqpt_diagnostics(
        self,
        model: object,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike | None = "all_up",
        initial_state_model: object | None = None,
        reference_state: InitialStateLike | None = None,
        reference_state_model: object | None = None,
        min_prominence: float = 0.0,
        min_cusp: float = 0.0,
        mode: str = "maxima",
    ) -> DQPTDiagnosticsResult:
        echo = self.loschmidt_echo(
            model,
            times,
            initial_state=initial_state,
            initial_state_model=initial_state_model,
            reference_state=reference_state,
            reference_state_model=reference_state_model,
        )
        return analyze_dqpt_from_loschmidt(
            echo,
            min_prominence=min_prominence,
            min_cusp=min_cusp,
            mode=mode,
        )

    def entanglement_spectrum(
        self,
        model: object,
        *,
        subsystem: Iterable[int],
        initial_state: InitialStateLike | None = None,
        num_levels: int | None = None,
    ) -> EntanglementSpectrumResult:
        num_sites = getattr(model, "num_sites")
        subsystem_sites = _normalize_subsystem(num_sites, subsystem)
        if not subsystem_sites:
            raise ValueError("subsystem must contain at least one site")
        if num_levels is not None and num_levels < 1:
            raise ValueError("num_levels must be at least 1 when provided")

        if _should_use_rust_entanglement_spectrum(
            solver=self.solver,
            model=model,
            subsystem=subsystem_sites,
            initial_state=initial_state,
        ):
            tensor_solver = _resolve_tensor_network_solver(self.solver)
            assert tensor_solver is not None
            return _rust_entanglement_spectrum(
                self,
                model,
                subsystem=subsystem_sites,
                initial_state=initial_state,
                tensor_solver=tensor_solver,
                num_levels=num_levels,
            )

        max_dimension = _resolve_exact_dimension_limit(self.solver)
        state = _resolve_exact_entanglement_state(
            model,
            initial_state=initial_state,
            max_dimension=max_dimension,
        )
        full_spectrum = pure_state_entanglement_spectrum(state, num_sites, list(subsystem_sites))
        entropy = float(-np.sum(full_spectrum * np.log2(full_spectrum)))
        spectrum = full_spectrum
        if num_levels is not None:
            spectrum = spectrum[:num_levels]
        return EntanglementSpectrumResult(
            model_name=getattr(model, "model_name", type(model).__name__),
            solver="exact_diagonalization",
            num_sites=num_sites,
            subsystem=subsystem_sites,
            eigenvalues=np.asarray(spectrum, dtype=np.float64),
            entropy=entropy,
            model_metadata=_model_metadata(model),
        )

    def correlation_matrix(
        self,
        model: object,
        *,
        pauli: str = "Z",
        connected: bool = False,
        initial_state: InitialStateLike | None = None,
    ) -> CorrelationMatrixResult:
        axis = _normalize_pauli_axis(pauli)
        observables = _correlation_observable_labels(
            getattr(model, "num_sites"),
            axis,
            include_single_site=True,
        )
        result = self.ground_state(
            model,
            observables=observables,
            initial_state=initial_state,
        )
        single_site = _single_site_expectations_from_observables(
            result.observables,
            getattr(model, "num_sites"),
            axis,
        )
        matrix = _correlation_matrix_from_observables(
            result.observables,
            getattr(model, "num_sites"),
            axis,
            connected=connected,
            single_site_expectations=single_site,
        )
        return CorrelationMatrixResult(
            model_name=result.model_name,
            solver=result.solver,
            pauli=axis,
            connected=connected,
            matrix=matrix,
            single_site_expectations=single_site,
            model_metadata=result.model_metadata,
        )

    def structure_factor(
        self,
        model: object,
        momenta: Iterable[float] | float | None = None,
        *,
        pauli: str = "Z",
        connected: bool = False,
        initial_state: InitialStateLike | None = None,
    ) -> StructureFactorResult:
        correlations = self.correlation_matrix(
            model,
            pauli=pauli,
            connected=connected,
            initial_state=initial_state,
        )
        momenta_array = _normalize_momenta(getattr(model, "num_sites"), momenta)
        values = _static_structure_factor_from_matrix(correlations.matrix, momenta_array)
        return StructureFactorResult(
            model_name=correlations.model_name,
            solver=correlations.solver,
            pauli=correlations.pauli,
            connected=correlations.connected,
            momenta=momenta_array,
            values=values,
            correlation_matrix=correlations.matrix,
            single_site_expectations=correlations.single_site_expectations,
            model_metadata=correlations.model_metadata,
        )

    def dynamic_structure_factor(
        self,
        model: object,
        times: Iterable[float],
        momenta: Iterable[float] | float | None = None,
        *,
        pauli: str = "Z",
        connected: bool = False,
        initial_state: InitialStateLike = "all_up",
    ) -> DynamicStructureFactorResult:
        axis = _normalize_pauli_axis(pauli)
        observables = _correlation_observable_labels(
            getattr(model, "num_sites"),
            axis,
            include_single_site=True,
        )
        evolution = self.quench(
            model,
            times,
            initial_state=initial_state,
            observables=observables,
        )
        momenta_array = _normalize_momenta(getattr(model, "num_sites"), momenta)
        values = _dynamic_structure_factor_from_traces(
            evolution.observables,
            getattr(model, "num_sites"),
            momenta_array,
            axis,
            connected=connected,
        )
        return DynamicStructureFactorResult(
            model_name=evolution.model_name,
            solver=evolution.solver,
            pauli=axis,
            connected=connected,
            times=np.asarray(evolution.times, dtype=np.float64),
            momenta=momenta_array,
            values=values,
            model_metadata=evolution.model_metadata,
        )

    def frequency_structure_factor(
        self,
        model: object | DynamicStructureFactorResult,
        times: Iterable[float] | None = None,
        momenta: Iterable[float] | float | None = None,
        *,
        pauli: str = "Z",
        connected: bool = False,
        initial_state: InitialStateLike = "all_up",
        frequencies: Iterable[float] | float | None = None,
        window: str = "hann",
        subtract_mean: bool = False,
    ) -> FrequencyStructureFactorResult:
        if isinstance(model, DynamicStructureFactorResult):
            dynamic = model
        else:
            if times is None:
                raise ValueError("times are required when building a frequency structure factor from a model")
            dynamic = self.dynamic_structure_factor(
                model,
                times,
                momenta,
                pauli=pauli,
                connected=connected,
                initial_state=initial_state,
            )
        return fourier_transform_structure_factor(
            dynamic,
            frequencies=frequencies,
            window=window,
            subtract_mean=subtract_mean,
        )

    def two_time_correlator(
        self,
        model: object,
        times: Iterable[float],
        *,
        pauli: str = "Z",
        connected: bool = False,
        initial_state: InitialStateLike | None = None,
        initial_state_model: object | None = None,
        source_sites: Iterable[int] | None = None,
        measure_sites: Iterable[int] | None = None,
    ) -> TwoTimeCorrelatorResult:
        axis = _normalize_pauli_axis(pauli)
        num_sites = getattr(model, "num_sites")
        _validate_prepared_state_model(
            target_model=model,
            state=initial_state,
            state_model=initial_state_model,
            role="initial",
        )
        resolved_measure_sites = _normalize_site_selection(
            num_sites,
            measure_sites,
            name="measure_sites",
        )
        resolved_source_sites = _normalize_site_selection(
            num_sites,
            source_sites,
            name="source_sites",
        )
        exact_solver = _resolve_exact_solver(self.solver)
        dimension = _model_dimension(model)
        tensor_solver = _resolve_tensor_network_solver(self.solver)
        if _should_use_rust_two_time_correlator(
            solver=self.solver,
            model=model,
            times=times,
            pauli=axis,
            initial_state=initial_state,
        ):
            assert tensor_solver is not None
            return _rust_two_time_correlator(
                self,
                model,
                times,
                pauli=axis,
                connected=connected,
                initial_state=initial_state,
                tensor_solver=tensor_solver,
                measure_sites=resolved_measure_sites,
                source_sites=resolved_source_sites,
            )
        if dimension > exact_solver.max_dimension:
            raise ValueError(
                "two-time correlators currently require exact diagonalization, "
                f"limited to dimension {exact_solver.max_dimension}, got {dimension}"
            )

        hamiltonian = np.asarray(model.hamiltonian(), dtype=np.complex128)
        times_array = _normalize_correlation_times(times)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        if initial_state is None:
            reference_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
        else:
            reference_state = make_initial_state(num_sites, initial_state)

        measure_operators = [
            pauli_string_operator(num_sites, {site: axis})
            for site in resolved_measure_sites
        ]
        source_operators = [
            pauli_string_operator(num_sites, {site: axis})
            for site in resolved_source_sites
        ]
        initial_single = np.asarray(
            [np.real(np.vdot(reference_state, operator @ reference_state)) for operator in source_operators],
            dtype=np.float64,
        )
        source_vectors = np.column_stack([operator @ reference_state for operator in source_operators])
        reference_coefficients = eigenvectors.conj().T @ reference_state
        source_coefficients = eigenvectors.conj().T @ source_vectors

        values = np.zeros(
            (times_array.size, len(resolved_measure_sites), len(resolved_source_sites)),
            dtype=np.complex128,
        )
        dynamic_single = np.zeros((times_array.size, len(resolved_measure_sites)), dtype=np.float64)

        for index, time in enumerate(times_array):
            phases = np.exp(-1.0j * eigenvalues * time)
            evolved_reference = eigenvectors @ (phases * reference_coefficients)
            evolved_sources = eigenvectors @ (phases[:, None] * source_coefficients)
            acted_reference = np.column_stack(
                [operator @ evolved_reference for operator in measure_operators]
            )
            values[index] = acted_reference.conj().T @ evolved_sources
            dynamic_single[index] = np.real(acted_reference.conj().T @ evolved_reference)

        if connected:
            values = values - dynamic_single[:, :, None] * initial_single[None, None, :]

        return TwoTimeCorrelatorResult(
            model_name=getattr(model, "model_name", type(model).__name__),
            solver="exact_diagonalization",
            pauli=axis,
            connected=connected,
            num_sites=num_sites,
            times=times_array,
            values=values,
            dynamic_single_site_expectations=dynamic_single,
            initial_single_site_expectations=initial_single,
            measure_sites=resolved_measure_sites,
            source_sites=resolved_source_sites,
            model_metadata=_model_metadata(model),
        )

    def linear_response_spectrum(
        self,
        model: object | TwoTimeCorrelatorResult,
        times: Iterable[float] | None = None,
        momenta: Iterable[float] | float | None = None,
        *,
        pauli: str = "Z",
        initial_state: InitialStateLike | None = None,
        initial_state_model: object | None = None,
        source_sites: Iterable[int] | None = None,
        measure_sites: Iterable[int] | None = None,
        frequencies: Iterable[float] | float | None = None,
        window: str = "hann",
        subtract_mean: bool = False,
    ) -> ResponseSpectrumResult:
        if isinstance(model, TwoTimeCorrelatorResult):
            if source_sites is not None or measure_sites is not None:
                raise ValueError(
                    "source_sites and measure_sites can only be provided when "
                    "building a response spectrum directly from a model"
                )
            correlator = model
        else:
            if times is None:
                raise ValueError("times are required when building a response spectrum from a model")
            if source_sites is None and measure_sites is not None:
                source_sites = tuple(int(site) for site in measure_sites)
            elif measure_sites is None and source_sites is not None:
                measure_sites = tuple(int(site) for site in source_sites)
            correlator = self.two_time_correlator(
                model,
                times,
                pauli=pauli,
                connected=False,
                initial_state=initial_state,
                initial_state_model=initial_state_model,
                source_sites=source_sites,
                measure_sites=measure_sites,
            )
        return response_spectrum_from_correlator(
            correlator,
            momenta=momenta,
            frequencies=frequencies,
            window=window,
            subtract_mean=subtract_mean,
        )

    def sweep_parameter(
        self,
        model: object,
        parameter: str,
        values: Iterable[float],
        *,
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
        warm_start: bool = True,
    ) -> SweepResult:
        if not is_dataclass(model):
            raise TypeError("parameter sweeps currently require dataclass-based model specs")
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")

        values_array = np.asarray(list(values), dtype=np.float64)
        requested_observables = tuple(observables)
        model_name = getattr(model, "model_name", type(model).__name__)
        base_metadata = _model_metadata(model)
        subsystem_sites = _normalize_subsystem(getattr(model, "num_sites"), subsystem)
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        restored = None
        if resume and checkpoint is not None and checkpoint.exists():
            from .state_io import load_sweep_result

            restored = load_sweep_result(checkpoint)
            _validate_sweep_checkpoint(
                restored,
                parameter=parameter,
                model_name=model_name,
                values=values_array,
                observables=requested_observables,
                subsystem=subsystem_sites,
                base_model_metadata=base_metadata,
            )

        if restored is None:
            sweep = _empty_sweep_result(
                parameter=parameter,
                model_name=model_name,
                values=values_array,
                observables=requested_observables,
                base_model_metadata=base_metadata,
                subsystem=subsystem_sites,
            )
        else:
            sweep = _coerce_sweep_result(restored)

        return self._run_sweep_points(
            sweep,
            model=model,
            parameter=parameter,
            observables=requested_observables,
            subsystem=subsystem,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            warm_start=warm_start,
        )

    def adaptive_sweep_parameter(
        self,
        model: object,
        parameter: str,
        values: Iterable[float],
        *,
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
        metric: str = "spectral_gap",
        strategy: str | None = None,
        target_value: float | None = None,
        insertion_policy: str | None = None,
        points_per_interval: int = 1,
        max_refinement_rounds: int = 2,
        refinements_per_round: int = 1,
        min_spacing: float | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
        warm_start: bool = True,
    ) -> SweepResult:
        if not is_dataclass(model):
            raise TypeError("adaptive sweeps currently require dataclass-based model specs")
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if max_refinement_rounds < 0:
            raise ValueError("max_refinement_rounds must be non-negative")
        if refinements_per_round < 1:
            raise ValueError("refinements_per_round must be at least 1")
        if points_per_interval < 1:
            raise ValueError("points_per_interval must be at least 1")
        if min_spacing is not None and min_spacing <= 0.0:
            raise ValueError("min_spacing must be positive")

        seed_values = _sorted_unique_values(values)
        resolved_strategy = _resolve_refinement_strategy(
            metric,
            strategy=strategy,
            target_value=target_value,
        )
        resolved_insertion_policy = _resolve_insertion_policy(
            insertion_policy=insertion_policy,
            refinement_strategy=resolved_strategy,
            target_value=target_value,
            points_per_interval=points_per_interval,
        )
        requested_observables = _refinement_observables(tuple(observables), metric)
        model_name = getattr(model, "model_name", type(model).__name__)
        base_metadata = _model_metadata(model)
        subsystem_sites = _normalize_subsystem(getattr(model, "num_sites"), subsystem)
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        restored = None
        if resume and checkpoint is not None and checkpoint.exists():
            from .state_io import load_sweep_result

            restored = load_sweep_result(checkpoint)
            _validate_adaptive_sweep_checkpoint(
                restored,
                parameter=parameter,
                model_name=model_name,
                seed_values=seed_values,
                observables=requested_observables,
                subsystem=subsystem_sites,
                base_model_metadata=base_metadata,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                max_refinement_rounds=max_refinement_rounds,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
            )

        if restored is None:
            sweep = _empty_sweep_result(
                parameter=parameter,
                model_name=model_name,
                values=seed_values,
                observables=requested_observables,
                base_model_metadata=base_metadata,
                subsystem=subsystem_sites,
                seed_values=seed_values,
                refinement_metric=metric,
                refinement_strategy=resolved_strategy,
                refinement_target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                max_refinement_rounds=max_refinement_rounds,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
                refinement_history=[],
            )
        else:
            sweep = _coerce_sweep_result(restored)

        while True:
            sweep = self._run_sweep_points(
                sweep,
                model=model,
                parameter=parameter,
                observables=requested_observables,
                subsystem=subsystem,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
                warm_start=warm_start,
            )
            if sweep.completed_rounds >= max_refinement_rounds:
                return sweep

            inserted = _plan_refinement_points(
                sweep,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
            )
            if not inserted:
                return sweep

            sweep = _extend_sweep_result(
                sweep,
                inserted,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
            )
            if checkpoint is not None:
                from .state_io import save_sweep_result

                save_sweep_result(sweep, checkpoint)

    def scan_dqpt_parameter(
        self,
        model: object,
        parameter: str,
        values: Iterable[float],
        times: Iterable[float],
        *,
        initial_state: InitialStateLike | None = "all_up",
        initial_state_model: object | None = None,
        reference_state: InitialStateLike | None = None,
        reference_state_model: object | None = None,
        min_prominence: float = 0.0,
        min_cusp: float = 0.0,
        mode: str = "maxima",
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
    ) -> DQPTScanResult:
        if not is_dataclass(model):
            raise TypeError("DQPT parameter scans currently require dataclass-based model specs")
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")

        values_array = np.asarray(list(values), dtype=np.float64)
        times_array = _normalize_correlation_times(times)
        model_name = getattr(model, "model_name", type(model).__name__)
        base_metadata = _model_metadata(model)
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        restored = None
        initial_descriptor = _state_descriptor(initial_state)
        reference_descriptor = _state_descriptor(reference_state)
        initial_model_metadata = (
            None if initial_state_model is None else _model_metadata(initial_state_model)
        )
        reference_model_metadata = (
            None if reference_state_model is None else _model_metadata(reference_state_model)
        )
        if resume and checkpoint is not None and checkpoint.exists():
            from .state_io import load_dqpt_scan_result

            restored = load_dqpt_scan_result(checkpoint)
            _validate_dqpt_scan_checkpoint(
                restored,
                parameter=parameter,
                model_name=model_name,
                values=values_array,
                times=times_array,
                base_model_metadata=base_metadata,
                initial_state_descriptor=initial_descriptor,
                reference_state_descriptor=reference_descriptor,
                initial_state_model_metadata=initial_model_metadata,
                reference_state_model_metadata=reference_model_metadata,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
            )

        if restored is None:
            scan = _empty_dqpt_scan_result(
                parameter=parameter,
                model_name=model_name,
                values=values_array,
                times=times_array,
                base_model_metadata=base_metadata,
                initial_state_descriptor=initial_descriptor,
                reference_state_descriptor=reference_descriptor,
                initial_state_model_metadata=initial_model_metadata,
                reference_state_model_metadata=reference_model_metadata,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
            )
        else:
            scan = _coerce_dqpt_scan_result(restored)

        return self._run_dqpt_scan_points(
            scan,
            model=model,
            parameter=parameter,
            initial_state=initial_state,
            initial_state_model=initial_state_model,
            reference_state=reference_state,
            reference_state_model=reference_state_model,
            min_prominence=min_prominence,
            min_cusp=min_cusp,
            mode=mode,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
        )

    def adaptive_scan_dqpt_parameter(
        self,
        model: object,
        parameter: str,
        values: Iterable[float],
        times: Iterable[float],
        *,
        initial_state: InitialStateLike | None = "all_up",
        initial_state_model: object | None = None,
        reference_state: InitialStateLike | None = None,
        reference_state_model: object | None = None,
        min_prominence: float = 0.0,
        min_cusp: float = 0.0,
        mode: str = "maxima",
        metric: str = "strongest_cusp_strength",
        strategy: str | None = None,
        target_value: float | None = None,
        insertion_policy: str | None = None,
        points_per_interval: int = 1,
        max_refinement_rounds: int = 2,
        refinements_per_round: int = 1,
        min_spacing: float | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        checkpoint_every: int = 1,
    ) -> DQPTScanResult:
        if not is_dataclass(model):
            raise TypeError("adaptive DQPT scans currently require dataclass-based model specs")
        if resume and checkpoint_path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        if max_refinement_rounds < 0:
            raise ValueError("max_refinement_rounds must be non-negative")
        if refinements_per_round < 1:
            raise ValueError("refinements_per_round must be at least 1")
        if points_per_interval < 1:
            raise ValueError("points_per_interval must be at least 1")
        if min_spacing is not None and min_spacing <= 0.0:
            raise ValueError("min_spacing must be positive")

        seed_values = _sorted_unique_values(values)
        times_array = _normalize_correlation_times(times)
        resolved_strategy = _resolve_refinement_strategy(
            metric,
            strategy=strategy,
            target_value=target_value,
        )
        resolved_insertion_policy = _resolve_insertion_policy(
            insertion_policy=insertion_policy,
            refinement_strategy=resolved_strategy,
            target_value=target_value,
            points_per_interval=points_per_interval,
        )
        model_name = getattr(model, "model_name", type(model).__name__)
        base_metadata = _model_metadata(model)
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        restored = None
        initial_descriptor = _state_descriptor(initial_state)
        reference_descriptor = _state_descriptor(reference_state)
        initial_model_metadata = (
            None if initial_state_model is None else _model_metadata(initial_state_model)
        )
        reference_model_metadata = (
            None if reference_state_model is None else _model_metadata(reference_state_model)
        )
        if resume and checkpoint is not None and checkpoint.exists():
            from .state_io import load_dqpt_scan_result

            restored = load_dqpt_scan_result(checkpoint)
            _validate_adaptive_dqpt_scan_checkpoint(
                restored,
                parameter=parameter,
                model_name=model_name,
                seed_values=seed_values,
                times=times_array,
                base_model_metadata=base_metadata,
                initial_state_descriptor=initial_descriptor,
                reference_state_descriptor=reference_descriptor,
                initial_state_model_metadata=initial_model_metadata,
                reference_state_model_metadata=reference_model_metadata,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                max_refinement_rounds=max_refinement_rounds,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
            )

        if restored is None:
            scan = _empty_dqpt_scan_result(
                parameter=parameter,
                model_name=model_name,
                values=seed_values,
                times=times_array,
                base_model_metadata=base_metadata,
                initial_state_descriptor=initial_descriptor,
                reference_state_descriptor=reference_descriptor,
                initial_state_model_metadata=initial_model_metadata,
                reference_state_model_metadata=reference_model_metadata,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
                seed_values=seed_values,
                refinement_metric=metric,
                refinement_strategy=resolved_strategy,
                refinement_target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                max_refinement_rounds=max_refinement_rounds,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
                refinement_history=[],
            )
        else:
            scan = _coerce_dqpt_scan_result(restored)

        while True:
            scan = self._run_dqpt_scan_points(
                scan,
                model=model,
                parameter=parameter,
                initial_state=initial_state,
                initial_state_model=initial_state_model,
                reference_state=reference_state,
                reference_state_model=reference_state_model,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
            )
            if scan.completed_rounds >= max_refinement_rounds:
                return scan

            inserted = _plan_dqpt_scan_refinement_points(
                scan,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
                refinements_per_round=refinements_per_round,
                min_spacing=min_spacing,
            )
            if not inserted:
                return scan

            scan = _extend_dqpt_scan_result(
                scan,
                inserted,
                metric=metric,
                strategy=resolved_strategy,
                target_value=target_value,
                insertion_policy=resolved_insertion_policy,
                points_per_interval=points_per_interval,
            )
            if checkpoint is not None:
                from .state_io import save_dqpt_scan_result

                save_dqpt_scan_result(scan, checkpoint)

    def _run_sweep_points(
        self,
        sweep: SweepResult,
        *,
        model: object,
        parameter: str,
        observables: tuple[str, ...],
        subsystem: Iterable[int] | None,
        checkpoint: Path | None,
        checkpoint_every: int,
        warm_start: bool,
    ) -> SweepResult:
        point_map = _point_map(sweep.points, sweep.values)
        energies = np.asarray(sweep.energies, dtype=np.float64)
        observable_series = {
            label: np.asarray(
                sweep.observables.get(
                    label,
                    np.full_like(sweep.values, np.nan, dtype=np.float64),
                ),
                dtype=np.float64,
            )
            for label in observables
        }
        spectral_gaps = _optional_trace(sweep.spectral_gaps, sweep.values)
        entanglement_series = _optional_trace(sweep.entanglement_entropy, sweep.values)

        while len(point_map) < len(sweep.values):
            index = _next_pending_index(
                point_map,
                sweep.values,
                warm_start=warm_start,
            )
            value = float(sweep.values[index])
            swept_model = replace(model, **{parameter: value})
            initial_state = None
            if warm_start:
                initial_state = _warm_start_state_for_index(
                    point_map,
                    index=index,
                    values=sweep.values,
                )
            result = self.ground_state(
                swept_model,
                observables=observables,
                initial_state=initial_state,
                subsystem=subsystem,
            )
            energies[index] = result.ground_state_energy
            for label, observable_value in result.observables.items():
                if label in observable_series:
                    observable_series[label][index] = observable_value
            if result.spectral_gap is not None:
                spectral_gaps[index] = result.spectral_gap
            if result.entanglement_entropy is not None:
                entanglement_series[index] = result.entanglement_entropy

            point_map[index] = SweepPoint(
                value=value,
                energy=result.ground_state_energy,
                observables=dict(result.observables),
                solver=result.solver,
                spectral_gap=result.spectral_gap,
                entanglement_entropy=result.entanglement_entropy,
                model_metadata=result.model_metadata,
                backend_state=result.backend_state,
                index=index,
            )
            sweep = _build_sweep_result(
                sweep,
                energies=energies,
                observables=observable_series,
                points=point_map,
                spectral_gaps=spectral_gaps,
                entanglement_entropy=entanglement_series,
            )
            if checkpoint is not None and (
                sweep.completed_points % checkpoint_every == 0 or sweep.is_complete
            ):
                from .state_io import save_sweep_result

                save_sweep_result(sweep, checkpoint)

        return sweep

    def _run_dqpt_scan_points(
        self,
        scan: DQPTScanResult,
        *,
        model: object,
        parameter: str,
        initial_state: InitialStateLike | None,
        initial_state_model: object | None,
        reference_state: InitialStateLike | None,
        reference_state_model: object | None,
        min_prominence: float,
        min_cusp: float,
        mode: str,
        checkpoint: Path | None,
        checkpoint_every: int,
    ) -> DQPTScanResult:
        point_map = _dqpt_scan_point_map(scan.points, scan.values)
        return_rates = np.asarray(scan.return_rates, dtype=np.float64)
        candidate_counts = np.asarray(scan.candidate_counts, dtype=np.float64)
        max_return_rates = np.asarray(scan.max_return_rates, dtype=np.float64)
        strongest_candidate_times = np.asarray(scan.strongest_candidate_times, dtype=np.float64)
        strongest_candidate_return_rates = np.asarray(
            scan.strongest_candidate_return_rates,
            dtype=np.float64,
        )
        strongest_prominences = np.asarray(scan.strongest_prominences, dtype=np.float64)
        strongest_cusp_strengths = np.asarray(scan.strongest_cusp_strengths, dtype=np.float64)

        while len(point_map) < len(scan.values):
            index = _next_pending_index(point_map, scan.values, warm_start=False)
            value = float(scan.values[index])
            swept_model = replace(model, **{parameter: value})
            diagnostics = self.dqpt_diagnostics(
                swept_model,
                scan.times,
                initial_state=initial_state,
                initial_state_model=initial_state_model,
                reference_state=reference_state,
                reference_state_model=reference_state_model,
                min_prominence=min_prominence,
                min_cusp=min_cusp,
                mode=mode,
            )
            strongest = _strongest_dqpt_candidate(diagnostics.candidates)
            candidate_times = np.asarray(
                [candidate.time for candidate in diagnostics.candidates],
                dtype=np.float64,
            )
            candidate_return_rates = np.asarray(
                [candidate.return_rate for candidate in diagnostics.candidates],
                dtype=np.float64,
            )
            candidate_prominences = np.asarray(
                [candidate.prominence for candidate in diagnostics.candidates],
                dtype=np.float64,
            )
            candidate_cusp_strengths = np.asarray(
                [candidate.cusp_strength for candidate in diagnostics.candidates],
                dtype=np.float64,
            )
            return_rates[index] = diagnostics.return_rate
            candidate_counts[index] = float(len(diagnostics.candidates))
            max_return_rates[index] = float(np.max(diagnostics.return_rate))
            if strongest is None:
                strongest_candidate_times[index] = np.nan
                strongest_candidate_return_rates[index] = np.nan
                strongest_prominences[index] = np.nan
                strongest_cusp_strengths[index] = np.nan
            else:
                strongest_candidate_times[index] = strongest.time
                strongest_candidate_return_rates[index] = strongest.return_rate
                strongest_prominences[index] = strongest.prominence
                strongest_cusp_strengths[index] = strongest.cusp_strength

            point_map[index] = DQPTScanPoint(
                value=value,
                solver=diagnostics.solver,
                return_rate=np.asarray(diagnostics.return_rate, dtype=np.float64),
                candidate_times=candidate_times,
                candidate_return_rates=candidate_return_rates,
                candidate_prominences=candidate_prominences,
                candidate_cusp_strengths=candidate_cusp_strengths,
                strongest_candidate_time=None if strongest is None else float(strongest.time),
                strongest_candidate_return_rate=None
                if strongest is None
                else float(strongest.return_rate),
                strongest_prominence=None if strongest is None else float(strongest.prominence),
                strongest_cusp_strength=None if strongest is None else float(strongest.cusp_strength),
                model_metadata=diagnostics.model_metadata,
                backend_state=diagnostics.backend_state,
                index=index,
            )
            scan = _build_dqpt_scan_result(
                scan,
                return_rates=return_rates,
                candidate_counts=candidate_counts,
                max_return_rates=max_return_rates,
                strongest_candidate_times=strongest_candidate_times,
                strongest_candidate_return_rates=strongest_candidate_return_rates,
                strongest_prominences=strongest_prominences,
                strongest_cusp_strengths=strongest_cusp_strengths,
                points=point_map,
            )
            if checkpoint is not None and (
                scan.completed_points % checkpoint_every == 0 or scan.is_complete
            ):
                from .state_io import save_dqpt_scan_result

                save_dqpt_scan_result(scan, checkpoint)

        return scan


def _normalize_subsystem(num_sites: int, subsystem: Iterable[int] | None) -> tuple[int, ...] | None:
    if subsystem is None:
        return None
    ordered = sorted(set(int(site) for site in subsystem))
    if any(site < 0 or site >= num_sites for site in ordered):
        raise ValueError("subsystem indices out of range")
    return tuple(ordered)


def _normalize_site_selection(
    num_sites: int,
    sites: Iterable[int] | None,
    *,
    name: str,
) -> tuple[int, ...]:
    if sites is None:
        return tuple(range(num_sites))
    resolved = tuple(int(site) for site in sites)
    if not resolved:
        raise ValueError(f"{name} must contain at least one site")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate sites")
    if any(site < 0 or site >= num_sites for site in resolved):
        raise ValueError(f"{name} indices out of range")
    return resolved


def _model_metadata(model: object) -> dict[str, object] | None:
    if is_dataclass(model):
        metadata = asdict(model)
        metadata["model_name"] = getattr(model, "model_name", type(model).__name__)
        return metadata
    return {"model_name": getattr(model, "model_name", type(model).__name__)}


def _empty_sweep_result(
    *,
    parameter: str,
    model_name: str,
    values: np.ndarray,
    observables: tuple[str, ...],
    base_model_metadata: dict[str, object] | None,
    subsystem: tuple[int, ...] | None,
    seed_values: np.ndarray | None = None,
    refinement_metric: str | None = None,
    refinement_strategy: str | None = None,
    refinement_target_value: float | None = None,
    insertion_policy: str | None = None,
    points_per_interval: int = 1,
    max_refinement_rounds: int = 0,
    refinements_per_round: int = 0,
    min_spacing: float | None = None,
    refinement_history: list[dict[str, object]] | None = None,
) -> SweepResult:
    values_array = np.asarray(values, dtype=np.float64)
    return SweepResult(
        parameter=parameter,
        model_name=model_name,
        solver="unknown",
        values=values_array,
        energies=np.full_like(values_array, np.nan, dtype=np.float64),
        observables={
            label: np.full_like(values_array, np.nan, dtype=np.float64)
            for label in observables
        },
        points=[],
        base_model_metadata=base_model_metadata,
        observables_requested=observables,
        subsystem=subsystem,
        spectral_gaps=None,
        entanglement_entropy=None,
        completed_points=0,
        seed_values=None if seed_values is None else np.asarray(seed_values, dtype=np.float64),
        refinement_metric=refinement_metric,
        refinement_strategy=refinement_strategy,
        refinement_target_value=refinement_target_value,
        insertion_policy=insertion_policy,
        points_per_interval=points_per_interval,
        max_refinement_rounds=max_refinement_rounds,
        refinements_per_round=refinements_per_round,
        min_spacing=min_spacing,
        refinement_history=list(refinement_history or []),
    )


def _coerce_sweep_result(result: SweepResult) -> SweepResult:
    point_map = _point_map(result.points, result.values)
    return _build_sweep_result(
        replace(result, points=[]),
        energies=np.asarray(result.energies, dtype=np.float64),
        observables={
            label: np.asarray(values, dtype=np.float64)
            for label, values in result.observables.items()
        },
        points=point_map,
        spectral_gaps=_optional_trace(result.spectral_gaps, result.values),
        entanglement_entropy=_optional_trace(result.entanglement_entropy, result.values),
    )


def _empty_dqpt_scan_result(
    *,
    parameter: str,
    model_name: str,
    values: np.ndarray,
    times: np.ndarray,
    base_model_metadata: dict[str, object] | None,
    initial_state_descriptor: dict[str, object] | None,
    reference_state_descriptor: dict[str, object] | None,
    initial_state_model_metadata: dict[str, object] | None,
    reference_state_model_metadata: dict[str, object] | None,
    min_prominence: float,
    min_cusp: float,
    mode: str,
    seed_values: np.ndarray | None = None,
    refinement_metric: str | None = None,
    refinement_strategy: str | None = None,
    refinement_target_value: float | None = None,
    insertion_policy: str | None = None,
    points_per_interval: int = 1,
    max_refinement_rounds: int = 0,
    refinements_per_round: int = 0,
    min_spacing: float | None = None,
    refinement_history: list[dict[str, object]] | None = None,
) -> DQPTScanResult:
    values_array = np.asarray(values, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    return DQPTScanResult(
        parameter=parameter,
        model_name=model_name,
        solver="unknown",
        values=values_array,
        times=times_array,
        return_rates=np.full((len(values_array), len(times_array)), np.nan, dtype=np.float64),
        candidate_counts=np.full(len(values_array), np.nan, dtype=np.float64),
        max_return_rates=np.full(len(values_array), np.nan, dtype=np.float64),
        strongest_candidate_times=np.full(len(values_array), np.nan, dtype=np.float64),
        strongest_candidate_return_rates=np.full(len(values_array), np.nan, dtype=np.float64),
        strongest_prominences=np.full(len(values_array), np.nan, dtype=np.float64),
        strongest_cusp_strengths=np.full(len(values_array), np.nan, dtype=np.float64),
        points=[],
        base_model_metadata=base_model_metadata,
        initial_state_descriptor=initial_state_descriptor,
        reference_state_descriptor=reference_state_descriptor,
        initial_state_model_metadata=initial_state_model_metadata,
        reference_state_model_metadata=reference_state_model_metadata,
        min_prominence=float(min_prominence),
        min_cusp=float(min_cusp),
        mode=mode,
        completed_points=0,
        seed_values=None if seed_values is None else np.asarray(seed_values, dtype=np.float64),
        refinement_metric=refinement_metric,
        refinement_strategy=refinement_strategy,
        refinement_target_value=refinement_target_value,
        insertion_policy=insertion_policy,
        points_per_interval=int(points_per_interval),
        max_refinement_rounds=int(max_refinement_rounds),
        refinements_per_round=int(refinements_per_round),
        min_spacing=min_spacing,
        refinement_history=list(refinement_history or []),
    )


def _coerce_dqpt_scan_result(result: DQPTScanResult) -> DQPTScanResult:
    point_map = _dqpt_scan_point_map(result.points, result.values)
    return _build_dqpt_scan_result(
        replace(result, points=[]),
        return_rates=np.asarray(result.return_rates, dtype=np.float64),
        candidate_counts=np.asarray(result.candidate_counts, dtype=np.float64),
        max_return_rates=np.asarray(result.max_return_rates, dtype=np.float64),
        strongest_candidate_times=np.asarray(result.strongest_candidate_times, dtype=np.float64),
        strongest_candidate_return_rates=np.asarray(
            result.strongest_candidate_return_rates,
            dtype=np.float64,
        ),
        strongest_prominences=np.asarray(result.strongest_prominences, dtype=np.float64),
        strongest_cusp_strengths=np.asarray(result.strongest_cusp_strengths, dtype=np.float64),
        points=point_map,
    )


def _build_dqpt_scan_result(
    template: DQPTScanResult,
    *,
    return_rates: np.ndarray,
    candidate_counts: np.ndarray,
    max_return_rates: np.ndarray,
    strongest_candidate_times: np.ndarray,
    strongest_candidate_return_rates: np.ndarray,
    strongest_prominences: np.ndarray,
    strongest_cusp_strengths: np.ndarray,
    points: dict[int, DQPTScanPoint],
) -> DQPTScanResult:
    ordered_points = [points[index] for index in sorted(points)]
    scan_solver = _resolve_sweep_solver(ordered_points)
    return replace(
        template,
        solver=scan_solver,
        return_rates=np.asarray(return_rates, dtype=np.float64),
        candidate_counts=np.asarray(candidate_counts, dtype=np.float64),
        max_return_rates=np.asarray(max_return_rates, dtype=np.float64),
        strongest_candidate_times=np.asarray(strongest_candidate_times, dtype=np.float64),
        strongest_candidate_return_rates=np.asarray(
            strongest_candidate_return_rates,
            dtype=np.float64,
        ),
        strongest_prominences=np.asarray(strongest_prominences, dtype=np.float64),
        strongest_cusp_strengths=np.asarray(strongest_cusp_strengths, dtype=np.float64),
        points=ordered_points,
        completed_points=len(ordered_points),
    )


def _build_sweep_result(
    template: SweepResult,
    *,
    energies: np.ndarray,
    observables: dict[str, np.ndarray],
    points: dict[int, SweepPoint],
    spectral_gaps: np.ndarray,
    entanglement_entropy: np.ndarray,
) -> SweepResult:
    ordered_points = [points[index] for index in sorted(points)]
    sweep_solver = _resolve_sweep_solver(ordered_points)
    return replace(
        template,
        solver=sweep_solver,
        energies=np.asarray(energies, dtype=np.float64),
        observables={
            label: np.asarray(values, dtype=np.float64)
            for label, values in observables.items()
        },
        points=ordered_points,
        spectral_gaps=_compact_optional_trace(spectral_gaps),
        entanglement_entropy=_compact_optional_trace(entanglement_entropy),
        completed_points=len(ordered_points),
    )


def _resolve_sweep_solver(points: list[SweepPoint]) -> str:
    if not points:
        return "unknown"
    solver = points[0].solver
    if any(point.solver != solver for point in points[1:]):
        return "mixed"
    return solver


def _optional_trace(values: np.ndarray | None, reference: np.ndarray) -> np.ndarray:
    if values is None:
        return np.full_like(reference, np.nan, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _compact_optional_trace(values: np.ndarray) -> np.ndarray | None:
    if not np.any(np.isfinite(values)):
        return None
    return np.asarray(values, dtype=np.float64)


def _point_map(points: list[SweepPoint], values: np.ndarray) -> dict[int, SweepPoint]:
    point_map: dict[int, SweepPoint] = {}
    for ordinal, point in enumerate(points):
        index = point.index
        if index is None:
            index = _infer_point_index(point.value, values, point_map, ordinal)
        if index < 0 or index >= len(values):
            raise ValueError("checkpoint point index is out of range")
        if index in point_map:
            raise ValueError("checkpoint contains duplicate sweep point indices")
        point_map[index] = replace(point, index=index)
    return point_map


def _dqpt_scan_point_map(points: list[DQPTScanPoint], values: np.ndarray) -> dict[int, DQPTScanPoint]:
    point_map: dict[int, DQPTScanPoint] = {}
    for ordinal, point in enumerate(points):
        index = point.index
        if index is None:
            index = _infer_point_index(point.value, values, point_map, ordinal)
        if index < 0 or index >= len(values):
            raise ValueError("checkpoint DQPT-scan point index is out of range")
        if index in point_map:
            raise ValueError("checkpoint contains duplicate DQPT-scan point indices")
        point_map[index] = replace(point, index=index)
    return point_map


def _infer_point_index(
    value: float,
    values: np.ndarray,
    existing_points: dict[int, SweepPoint],
    fallback_index: int,
) -> int:
    matches = [int(index) for index in np.flatnonzero(np.isclose(values, value, atol=1e-12, rtol=1e-9))]
    for index in matches:
        if index not in existing_points:
            return index
    if fallback_index < len(values) and fallback_index not in existing_points:
        return fallback_index
    raise ValueError("unable to reconcile checkpoint sweep point with the requested grid")


def _warm_start_state_for_index(
    points: dict[int, SweepPoint],
    *,
    index: int,
    values: np.ndarray,
) -> object | None:
    anchor = _nearest_backend_state_index(points, values, index)
    if anchor is not None:
        return points[anchor].backend_state
    return None


def _next_pending_index(
    points: dict[int, SweepPoint],
    values: np.ndarray,
    *,
    warm_start: bool,
) -> int:
    pending = [index for index in range(len(values)) if index not in points]
    if not pending:
        raise ValueError("no pending sweep indices remain")
    if not warm_start:
        return pending[0]

    anchored = [index for index, point in points.items() if point.backend_state is not None]
    if not anchored:
        return pending[0]

    return min(
        pending,
        key=lambda index: (
            _nearest_backend_state_distance(points, values, index),
            abs(index - _nearest_completed_index(points, values, index)),
            index,
        ),
    )


def _nearest_backend_state_index(
    points: dict[int, SweepPoint],
    values: np.ndarray,
    index: int,
) -> int | None:
    anchored = [candidate for candidate, point in points.items() if point.backend_state is not None]
    if not anchored:
        return None
    return min(
        anchored,
        key=lambda candidate: (
            abs(float(values[candidate]) - float(values[index])),
            abs(candidate - index),
            candidate,
        ),
    )


def _nearest_backend_state_distance(
    points: dict[int, SweepPoint],
    values: np.ndarray,
    index: int,
) -> float:
    anchor = _nearest_backend_state_index(points, values, index)
    if anchor is None:
        return float("inf")
    return abs(float(values[anchor]) - float(values[index]))


def _nearest_completed_index(
    points: dict[int, SweepPoint],
    values: np.ndarray,
    index: int,
) -> int:
    return min(
        points,
        key=lambda candidate: (
            abs(float(values[candidate]) - float(values[index])),
            abs(candidate - index),
            candidate,
        ),
    )


def _sorted_unique_values(values: Iterable[float]) -> np.ndarray:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError("adaptive sweeps require a one-dimensional sweep grid")
    if values_array.size < 2:
        raise ValueError("adaptive sweeps require at least two seed values")
    if not np.all(np.isfinite(values_array)):
        raise ValueError("adaptive sweep values must be finite")
    unique = np.unique(values_array)
    if unique.size < 2:
        raise ValueError("adaptive sweeps require at least two distinct seed values")
    return unique


def _refinement_observables(observables: tuple[str, ...], metric: str) -> tuple[str, ...]:
    built_in_metrics = {"energy", "spectral_gap", "entanglement_entropy"}
    if metric in built_in_metrics or metric in observables:
        return observables
    return observables + (metric,)


def _resolve_refinement_strategy(
    metric: str,
    *,
    strategy: str | None,
    target_value: float | None,
) -> str:
    resolved = strategy or ("minimum_average" if metric == "spectral_gap" else "gradient_magnitude")
    valid = {"minimum_average", "gradient_magnitude", "curvature", "target_crossing"}
    if resolved not in valid:
        raise ValueError(f"unsupported adaptive refinement strategy: {resolved}")
    if resolved == "target_crossing":
        if target_value is None or not np.isfinite(target_value):
            raise ValueError("target_crossing refinement requires a finite target_value")
    elif target_value is not None and not np.isfinite(target_value):
        raise ValueError("target_value must be finite when provided")
    return resolved


def _resolve_insertion_policy(
    *,
    insertion_policy: str | None,
    refinement_strategy: str,
    target_value: float | None,
    points_per_interval: int,
) -> str:
    resolved = insertion_policy or (
        "target_linear" if refinement_strategy == "target_crossing" else "equal_spacing"
    )
    valid = {"equal_spacing", "target_linear"}
    if resolved not in valid:
        raise ValueError(f"unsupported adaptive insertion policy: {resolved}")
    if resolved == "target_linear":
        if target_value is None or not np.isfinite(target_value):
            raise ValueError("target_linear insertion requires a finite target_value")
        if points_per_interval != 1:
            raise ValueError("target_linear insertion currently supports only points_per_interval=1")
    return resolved


def _plan_refinement_points(
    sweep: SweepResult,
    *,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
    refinements_per_round: int,
    min_spacing: float | None,
) -> list[dict[str, object]]:
    return _plan_refinement_candidates(
        np.asarray(sweep.values, dtype=np.float64),
        _metric_trace(sweep, metric),
        strategy=strategy,
        target_value=target_value,
        insertion_policy=insertion_policy,
        points_per_interval=points_per_interval,
        refinements_per_round=refinements_per_round,
        min_spacing=min_spacing,
    )


def _plan_dqpt_scan_refinement_points(
    scan: DQPTScanResult,
    *,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
    refinements_per_round: int,
    min_spacing: float | None,
) -> list[dict[str, object]]:
    return _plan_refinement_candidates(
        np.asarray(scan.values, dtype=np.float64),
        _dqpt_scan_metric_trace(scan, metric),
        strategy=strategy,
        target_value=target_value,
        insertion_policy=insertion_policy,
        points_per_interval=points_per_interval,
        refinements_per_round=refinements_per_round,
        min_spacing=min_spacing,
    )


def _plan_refinement_candidates(
    values: np.ndarray,
    metric_values: np.ndarray,
    *,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
    refinements_per_round: int,
    min_spacing: float | None,
) -> list[dict[str, object]]:
    candidates = []
    for index in range(len(values) - 1):
        left = float(values[index])
        right = float(values[index + 1])
        left_metric = float(metric_values[index])
        right_metric = float(metric_values[index + 1])
        spacing = right - left
        if spacing <= 0.0:
            continue
        if min_spacing is not None and spacing <= min_spacing:
            continue
        score = _refinement_score(
            metric_values,
            values,
            index=index,
            strategy=strategy,
            target_value=target_value,
        )
        if not np.isfinite(score):
            continue
        if not np.isfinite(left_metric) or not np.isfinite(right_metric):
            continue
        inserted_values = _inserted_values_for_interval(
            left=left,
            right=right,
            left_metric=left_metric,
            right_metric=right_metric,
            target_value=target_value,
            insertion_policy=insertion_policy,
            points_per_interval=points_per_interval,
        )
        if not inserted_values:
            continue
        candidates.append(
            {
                "left": left,
                "right": right,
                "inserted_values": inserted_values,
                "score": float(score),
            }
        )

    if strategy == "minimum_average":
        candidates.sort(key=lambda candidate: (candidate["score"], candidate["left"]))
    else:
        candidates.sort(key=lambda candidate: (-candidate["score"], candidate["left"]))
    return candidates[:refinements_per_round]


def _inserted_values_for_interval(
    *,
    left: float,
    right: float,
    left_metric: float,
    right_metric: float,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
) -> list[float]:
    spacing = right - left
    if spacing <= 0.0:
        return []
    if insertion_policy == "equal_spacing":
        return [
            float(left + spacing * fraction)
            for fraction in (
                point / (points_per_interval + 1)
                for point in range(1, points_per_interval + 1)
            )
        ]
    if insertion_policy == "target_linear":
        assert target_value is not None
        fraction = _target_linear_fraction(
            left_metric=left_metric,
            right_metric=right_metric,
            target_value=target_value,
        )
        return [float(left + spacing * fraction)]
    raise ValueError(f"unsupported insertion policy: {insertion_policy}")


def _target_linear_fraction(
    *,
    left_metric: float,
    right_metric: float,
    target_value: float,
) -> float:
    delta = right_metric - left_metric
    if abs(delta) < 1e-12:
        return 0.5
    fraction = (target_value - left_metric) / delta
    if not np.isfinite(fraction) or fraction <= 1e-6 or fraction >= 1.0 - 1e-6:
        return 0.5
    return float(fraction)


def _metric_trace(sweep: SweepResult, metric: str) -> np.ndarray:
    if metric == "energy":
        trace = np.asarray(sweep.energies, dtype=np.float64)
    elif metric == "spectral_gap":
        if sweep.spectral_gaps is None:
            raise ValueError(
                "adaptive spectral-gap refinement requires finite spectral gaps on the current sweep grid"
            )
        trace = np.asarray(sweep.spectral_gaps, dtype=np.float64)
    elif metric == "entanglement_entropy":
        if sweep.entanglement_entropy is None:
            raise ValueError(
                "adaptive entanglement refinement requires subsystem entropy on the current sweep grid"
            )
        trace = np.asarray(sweep.entanglement_entropy, dtype=np.float64)
    else:
        if metric not in sweep.observables:
            raise ValueError(f"adaptive refinement metric '{metric}' is not available in the sweep data")
        trace = np.asarray(sweep.observables[metric], dtype=np.float64)

    if trace.shape != sweep.values.shape or not np.all(np.isfinite(trace)):
        raise ValueError(
            f"adaptive refinement metric '{metric}' must be finite across the evaluated sweep grid"
        )
    return trace


def _dqpt_scan_metric_trace(scan: DQPTScanResult, metric: str) -> np.ndarray:
    if metric == "candidate_count":
        trace = np.asarray(scan.candidate_counts, dtype=np.float64)
    elif metric == "max_return_rate":
        trace = np.asarray(scan.max_return_rates, dtype=np.float64)
    elif metric == "strongest_candidate_time":
        trace = np.asarray(scan.strongest_candidate_times, dtype=np.float64)
    elif metric == "strongest_candidate_return_rate":
        trace = np.asarray(scan.strongest_candidate_return_rates, dtype=np.float64)
    elif metric == "strongest_prominence":
        trace = np.asarray(scan.strongest_prominences, dtype=np.float64)
    elif metric == "strongest_cusp_strength":
        trace = np.asarray(scan.strongest_cusp_strengths, dtype=np.float64)
    else:
        raise ValueError(f"adaptive DQPT refinement metric '{metric}' is not available")

    if trace.shape != scan.values.shape:
        raise ValueError(f"adaptive DQPT refinement metric '{metric}' has an invalid shape")
    return trace


def _refinement_score(
    metric_values: np.ndarray,
    values: np.ndarray,
    *,
    index: int,
    strategy: str,
    target_value: float | None,
) -> float:
    if strategy == "minimum_average":
        return 0.5 * float(metric_values[index] + metric_values[index + 1])
    if strategy == "gradient_magnitude":
        spacing = float(values[index + 1] - values[index])
        return abs(float(metric_values[index + 1] - metric_values[index])) / spacing
    if strategy == "curvature":
        return _interval_curvature_score(metric_values, values, index)
    if strategy == "target_crossing":
        assert target_value is not None
        return _target_crossing_score(metric_values, index=index, target_value=target_value)
    spacing = float(values[index + 1] - values[index])
    return abs(float(metric_values[index + 1] - metric_values[index])) / spacing


def _interval_curvature_score(metric_values: np.ndarray, values: np.ndarray, index: int) -> float:
    scores = []
    if 0 < index < len(values) - 1:
        scores.append(_point_curvature_score(metric_values, values, index))
    if 0 < index + 1 < len(values) - 1:
        scores.append(_point_curvature_score(metric_values, values, index + 1))
    if not scores:
        return 0.0
    return max(scores)


def _point_curvature_score(metric_values: np.ndarray, values: np.ndarray, index: int) -> float:
    left_spacing = float(values[index] - values[index - 1])
    right_spacing = float(values[index + 1] - values[index])
    if left_spacing <= 0.0 or right_spacing <= 0.0:
        return 0.0
    left_slope = float(metric_values[index] - metric_values[index - 1]) / left_spacing
    right_slope = float(metric_values[index + 1] - metric_values[index]) / right_spacing
    center_spacing = 0.5 * (left_spacing + right_spacing)
    return abs(right_slope - left_slope) / max(center_spacing, 1e-12)


def _target_crossing_score(
    metric_values: np.ndarray,
    *,
    index: int,
    target_value: float,
) -> float:
    left = float(metric_values[index])
    right = float(metric_values[index + 1])
    midpoint = 0.5 * (left + right)
    spans_target = (left - target_value) * (right - target_value) <= 0.0
    bonus = 1_000_000.0 if spans_target else 0.0
    return bonus + 1.0 / (abs(midpoint - target_value) + 1e-12)


def _extend_sweep_result(
    sweep: SweepResult,
    inserted: list[dict[str, object]],
    *,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
) -> SweepResult:
    inserted_values = np.asarray(
        [
            value
            for candidate in inserted
            for value in candidate["inserted_values"]
        ],
        dtype=np.float64,
    )
    new_values = np.unique(np.concatenate([np.asarray(sweep.values, dtype=np.float64), inserted_values]))
    energies = np.full_like(new_values, np.nan, dtype=np.float64)
    observables = {
        label: np.full_like(new_values, np.nan, dtype=np.float64)
        for label in sweep.observables_requested
    }
    spectral_gaps = np.full_like(new_values, np.nan, dtype=np.float64)
    entanglement_entropy = np.full_like(new_values, np.nan, dtype=np.float64)

    old_points = _point_map(sweep.points, sweep.values)
    remapped_points: dict[int, SweepPoint] = {}
    old_spectral = _optional_trace(sweep.spectral_gaps, sweep.values)
    old_entanglement = _optional_trace(sweep.entanglement_entropy, sweep.values)
    for old_index, point in old_points.items():
        new_index = _infer_point_index(point.value, new_values, remapped_points, old_index)
        energies[new_index] = sweep.energies[old_index]
        for label in observables:
            observables[label][new_index] = sweep.observables[label][old_index]
        spectral_gaps[new_index] = old_spectral[old_index]
        entanglement_entropy[new_index] = old_entanglement[old_index]
        remapped_points[new_index] = replace(point, index=new_index)

    history = list(sweep.refinement_history or [])
    history.append(
        {
            "round": len(history) + 1,
            "metric": metric,
            "strategy": strategy,
            "target_value": target_value,
            "insertion_policy": insertion_policy,
            "points_per_interval": points_per_interval,
            "inserted_values": inserted_values.tolist(),
            "inserted_by_interval": [
                [float(value) for value in candidate["inserted_values"]]
                for candidate in inserted
            ],
            "selected_intervals": [
                [float(candidate["left"]), float(candidate["right"])]
                for candidate in inserted
            ],
            "scores": [float(candidate["score"]) for candidate in inserted],
        }
    )
    return _build_sweep_result(
        replace(
            sweep,
            values=new_values,
            refinement_history=history,
        ),
        energies=energies,
        observables=observables,
        points=remapped_points,
        spectral_gaps=spectral_gaps,
        entanglement_entropy=entanglement_entropy,
    )


def _extend_dqpt_scan_result(
    scan: DQPTScanResult,
    inserted: list[dict[str, object]],
    *,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
) -> DQPTScanResult:
    inserted_values = np.asarray(
        [
            value
            for candidate in inserted
            for value in candidate["inserted_values"]
        ],
        dtype=np.float64,
    )
    new_values = np.unique(np.concatenate([np.asarray(scan.values, dtype=np.float64), inserted_values]))
    return_rates = np.full((len(new_values), len(scan.times)), np.nan, dtype=np.float64)
    candidate_counts = np.full(len(new_values), np.nan, dtype=np.float64)
    max_return_rates = np.full(len(new_values), np.nan, dtype=np.float64)
    strongest_candidate_times = np.full(len(new_values), np.nan, dtype=np.float64)
    strongest_candidate_return_rates = np.full(len(new_values), np.nan, dtype=np.float64)
    strongest_prominences = np.full(len(new_values), np.nan, dtype=np.float64)
    strongest_cusp_strengths = np.full(len(new_values), np.nan, dtype=np.float64)

    old_points = _dqpt_scan_point_map(scan.points, scan.values)
    remapped_points: dict[int, DQPTScanPoint] = {}
    for old_index, point in old_points.items():
        new_index = _infer_point_index(point.value, new_values, remapped_points, old_index)
        return_rates[new_index] = np.asarray(scan.return_rates[old_index], dtype=np.float64)
        candidate_counts[new_index] = float(scan.candidate_counts[old_index])
        max_return_rates[new_index] = float(scan.max_return_rates[old_index])
        strongest_candidate_times[new_index] = float(scan.strongest_candidate_times[old_index])
        strongest_candidate_return_rates[new_index] = float(
            scan.strongest_candidate_return_rates[old_index]
        )
        strongest_prominences[new_index] = float(scan.strongest_prominences[old_index])
        strongest_cusp_strengths[new_index] = float(scan.strongest_cusp_strengths[old_index])
        remapped_points[new_index] = replace(point, index=new_index)

    history = list(scan.refinement_history or [])
    history.append(
        {
            "round": len(history) + 1,
            "metric": metric,
            "strategy": strategy,
            "target_value": target_value,
            "insertion_policy": insertion_policy,
            "points_per_interval": points_per_interval,
            "inserted_values": inserted_values.tolist(),
            "inserted_by_interval": [
                [float(value) for value in candidate["inserted_values"]]
                for candidate in inserted
            ],
            "selected_intervals": [
                [float(candidate["left"]), float(candidate["right"])]
                for candidate in inserted
            ],
            "scores": [float(candidate["score"]) for candidate in inserted],
        }
    )
    return _build_dqpt_scan_result(
        replace(
            scan,
            values=new_values,
            refinement_history=history,
        ),
        return_rates=return_rates,
        candidate_counts=candidate_counts,
        max_return_rates=max_return_rates,
        strongest_candidate_times=strongest_candidate_times,
        strongest_candidate_return_rates=strongest_candidate_return_rates,
        strongest_prominences=strongest_prominences,
        strongest_cusp_strengths=strongest_cusp_strengths,
        points=remapped_points,
    )


def _validate_sweep_checkpoint(
    checkpoint: SweepResult,
    *,
    parameter: str,
    model_name: str,
    values: np.ndarray,
    observables: tuple[str, ...],
    subsystem: tuple[int, ...] | None,
    base_model_metadata: dict[str, object] | None,
) -> None:
    if checkpoint.parameter != parameter:
        raise ValueError("checkpoint parameter does not match the requested sweep")
    if checkpoint.model_name != model_name:
        raise ValueError("checkpoint model does not match the requested sweep")
    if checkpoint.observables_requested != observables:
        raise ValueError("checkpoint observables do not match the requested sweep")
    if checkpoint.subsystem != subsystem:
        raise ValueError("checkpoint subsystem does not match the requested sweep")
    if checkpoint.base_model_metadata != base_model_metadata:
        raise ValueError("checkpoint model metadata does not match the requested sweep")
    if checkpoint.values.shape != values.shape or not np.allclose(checkpoint.values, values):
        raise ValueError("checkpoint values do not match the requested sweep grid")
    if checkpoint.energies.shape != values.shape:
        raise ValueError("checkpoint energies shape does not match the requested sweep grid")
    for label in observables:
        trace = checkpoint.observables.get(label)
        if trace is None or trace.shape != values.shape:
            raise ValueError(f"checkpoint observable '{label}' does not match the requested sweep grid")
    if checkpoint.spectral_gaps is not None and checkpoint.spectral_gaps.shape != values.shape:
        raise ValueError("checkpoint spectral_gaps shape does not match the requested sweep grid")
    if checkpoint.entanglement_entropy is not None and checkpoint.entanglement_entropy.shape != values.shape:
        raise ValueError("checkpoint entanglement_entropy shape does not match the requested sweep grid")
    point_map = _point_map(checkpoint.points, values)
    if checkpoint.completed_points < 0 or checkpoint.completed_points > len(values):
        raise ValueError("checkpoint completed_points is out of range")
    if len(point_map) != checkpoint.completed_points:
        raise ValueError("checkpoint point list does not match completed_points")
    for index, point in point_map.items():
        if not np.isclose(values[index], point.value, atol=1e-12, rtol=1e-9):
            raise ValueError("checkpoint point values do not match the requested sweep grid")


def _validate_adaptive_sweep_checkpoint(
    checkpoint: SweepResult,
    *,
    parameter: str,
    model_name: str,
    seed_values: np.ndarray,
    observables: tuple[str, ...],
    subsystem: tuple[int, ...] | None,
    base_model_metadata: dict[str, object] | None,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
    max_refinement_rounds: int,
    refinements_per_round: int,
    min_spacing: float | None,
) -> None:
    _validate_sweep_checkpoint(
        checkpoint,
        parameter=parameter,
        model_name=model_name,
        values=np.asarray(checkpoint.values, dtype=np.float64),
        observables=observables,
        subsystem=subsystem,
        base_model_metadata=base_model_metadata,
    )
    if checkpoint.seed_values is None:
        raise ValueError("checkpoint does not include adaptive sweep seed values")
    if checkpoint.seed_values.shape != seed_values.shape or not np.allclose(
        checkpoint.seed_values,
        seed_values,
    ):
        raise ValueError("checkpoint seed values do not match the requested adaptive sweep")
    if checkpoint.refinement_metric != metric:
        raise ValueError("checkpoint adaptive metric does not match the requested sweep")
    if checkpoint.refinement_strategy != strategy:
        raise ValueError("checkpoint adaptive strategy does not match the requested sweep")
    if checkpoint.refinement_target_value != target_value:
        raise ValueError("checkpoint adaptive target_value does not match the requested sweep")
    if checkpoint.insertion_policy != insertion_policy:
        raise ValueError("checkpoint adaptive insertion policy does not match the requested sweep")
    if checkpoint.points_per_interval != points_per_interval:
        raise ValueError("checkpoint adaptive points_per_interval does not match the requested sweep")
    if checkpoint.max_refinement_rounds != max_refinement_rounds:
        raise ValueError("checkpoint refinement-round limit does not match the requested sweep")
    if checkpoint.refinements_per_round != refinements_per_round:
        raise ValueError("checkpoint refinements-per-round does not match the requested sweep")
    if checkpoint.min_spacing != min_spacing:
        raise ValueError("checkpoint min_spacing does not match the requested sweep")


def _validate_adaptive_dqpt_scan_checkpoint(
    checkpoint: DQPTScanResult,
    *,
    parameter: str,
    model_name: str,
    seed_values: np.ndarray,
    times: np.ndarray,
    base_model_metadata: dict[str, object] | None,
    initial_state_descriptor: dict[str, object] | None,
    reference_state_descriptor: dict[str, object] | None,
    initial_state_model_metadata: dict[str, object] | None,
    reference_state_model_metadata: dict[str, object] | None,
    min_prominence: float,
    min_cusp: float,
    mode: str,
    metric: str,
    strategy: str,
    target_value: float | None,
    insertion_policy: str,
    points_per_interval: int,
    max_refinement_rounds: int,
    refinements_per_round: int,
    min_spacing: float | None,
) -> None:
    _validate_dqpt_scan_checkpoint(
        checkpoint,
        parameter=parameter,
        model_name=model_name,
        values=np.asarray(checkpoint.values, dtype=np.float64),
        times=times,
        base_model_metadata=base_model_metadata,
        initial_state_descriptor=initial_state_descriptor,
        reference_state_descriptor=reference_state_descriptor,
        initial_state_model_metadata=initial_state_model_metadata,
        reference_state_model_metadata=reference_state_model_metadata,
        min_prominence=min_prominence,
        min_cusp=min_cusp,
        mode=mode,
    )
    if checkpoint.seed_values is None:
        raise ValueError("checkpoint does not include adaptive DQPT seed values")
    if checkpoint.seed_values.shape != seed_values.shape or not np.allclose(
        checkpoint.seed_values,
        seed_values,
    ):
        raise ValueError("checkpoint seed values do not match the requested adaptive DQPT scan")
    if checkpoint.refinement_metric != metric:
        raise ValueError("checkpoint adaptive DQPT metric does not match the requested scan")
    if checkpoint.refinement_strategy != strategy:
        raise ValueError("checkpoint adaptive DQPT strategy does not match the requested scan")
    if checkpoint.refinement_target_value != target_value:
        raise ValueError("checkpoint adaptive DQPT target_value does not match the requested scan")
    if checkpoint.insertion_policy != insertion_policy:
        raise ValueError(
            "checkpoint adaptive DQPT insertion policy does not match the requested scan"
        )
    if checkpoint.points_per_interval != points_per_interval:
        raise ValueError(
            "checkpoint adaptive DQPT points_per_interval does not match the requested scan"
        )
    if checkpoint.max_refinement_rounds != max_refinement_rounds:
        raise ValueError(
            "checkpoint adaptive DQPT refinement-round limit does not match the requested scan"
        )
    if checkpoint.refinements_per_round != refinements_per_round:
        raise ValueError(
            "checkpoint adaptive DQPT refinements-per-round does not match the requested scan"
        )
    if checkpoint.min_spacing != min_spacing:
        raise ValueError("checkpoint adaptive DQPT min_spacing does not match the requested scan")


def _validate_dqpt_scan_checkpoint(
    checkpoint: DQPTScanResult,
    *,
    parameter: str,
    model_name: str,
    values: np.ndarray,
    times: np.ndarray,
    base_model_metadata: dict[str, object] | None,
    initial_state_descriptor: dict[str, object] | None,
    reference_state_descriptor: dict[str, object] | None,
    initial_state_model_metadata: dict[str, object] | None,
    reference_state_model_metadata: dict[str, object] | None,
    min_prominence: float,
    min_cusp: float,
    mode: str,
) -> None:
    if checkpoint.parameter != parameter:
        raise ValueError("checkpoint parameter does not match the requested DQPT scan")
    if checkpoint.model_name != model_name:
        raise ValueError("checkpoint model does not match the requested DQPT scan")
    if checkpoint.base_model_metadata != base_model_metadata:
        raise ValueError("checkpoint model metadata does not match the requested DQPT scan")
    if checkpoint.values.shape != values.shape or not np.allclose(checkpoint.values, values):
        raise ValueError("checkpoint values do not match the requested DQPT scan grid")
    if checkpoint.times.shape != times.shape or not np.allclose(checkpoint.times, times):
        raise ValueError("checkpoint times do not match the requested DQPT scan time grid")
    if checkpoint.initial_state_descriptor != initial_state_descriptor:
        raise ValueError("checkpoint initial_state does not match the requested DQPT scan")
    if checkpoint.reference_state_descriptor != reference_state_descriptor:
        raise ValueError("checkpoint reference_state does not match the requested DQPT scan")
    if checkpoint.initial_state_model_metadata != initial_state_model_metadata:
        raise ValueError("checkpoint initial_state_model does not match the requested DQPT scan")
    if checkpoint.reference_state_model_metadata != reference_state_model_metadata:
        raise ValueError("checkpoint reference_state_model does not match the requested DQPT scan")
    if not np.isclose(checkpoint.min_prominence, min_prominence):
        raise ValueError("checkpoint min_prominence does not match the requested DQPT scan")
    if not np.isclose(checkpoint.min_cusp, min_cusp):
        raise ValueError("checkpoint min_cusp does not match the requested DQPT scan")
    if checkpoint.mode != mode:
        raise ValueError("checkpoint mode does not match the requested DQPT scan")
    expected_shape = (len(values), len(times))
    if checkpoint.return_rates.shape != expected_shape:
        raise ValueError("checkpoint return_rates shape does not match the requested DQPT scan")
    for trace in (
        checkpoint.candidate_counts,
        checkpoint.max_return_rates,
        checkpoint.strongest_candidate_times,
        checkpoint.strongest_candidate_return_rates,
        checkpoint.strongest_prominences,
        checkpoint.strongest_cusp_strengths,
    ):
        if trace.shape != values.shape:
            raise ValueError("checkpoint DQPT summary trace does not match the requested scan grid")
    point_map = _dqpt_scan_point_map(checkpoint.points, values)
    if checkpoint.completed_points < 0 or checkpoint.completed_points > len(values):
        raise ValueError("checkpoint completed_points is out of range")
    if len(point_map) != checkpoint.completed_points:
        raise ValueError("checkpoint point list does not match completed_points")
    for index, point in point_map.items():
        if not np.isclose(values[index], point.value, atol=1e-12, rtol=1e-9):
            raise ValueError("checkpoint point values do not match the requested DQPT scan grid")
        if np.asarray(point.return_rate, dtype=np.float64).shape != times.shape:
            raise ValueError("checkpoint point return_rate does not match the requested time grid")


def fourier_transform_structure_factor(
    result: DynamicStructureFactorResult,
    *,
    frequencies: Iterable[float] | float | None = None,
    window: str = "hann",
    subtract_mean: bool = False,
) -> FrequencyStructureFactorResult:
    times = _validated_time_grid(result.times)
    values = np.asarray(result.values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("dynamic structure-factor data must have shape (num_times, num_momenta)")
    if values.shape[0] != times.size:
        raise ValueError("dynamic structure-factor time axis does not match the provided times")

    frequency_grid = _normalize_frequency_grid(times, frequencies)
    relative_times = times - times[0]
    signal = values.copy()
    if subtract_mean:
        signal -= np.mean(signal, axis=0, keepdims=True)

    weights = _window_weights(times.size, window)
    signal *= weights[:, None]
    phase = np.exp(1.0j * frequency_grid[:, None] * relative_times[None, :])
    spectrum = np.trapezoid(
        phase[:, :, None] * signal[None, :, :],
        x=relative_times,
        axis=1,
    )
    return FrequencyStructureFactorResult(
        model_name=result.model_name,
        solver=result.solver,
        pauli=result.pauli,
        connected=result.connected,
        times=times,
        frequencies=frequency_grid,
        momenta=np.asarray(result.momenta, dtype=np.float64),
        values=np.asarray(spectrum, dtype=np.complex128),
        window=window,
        subtract_mean=subtract_mean,
        model_metadata=result.model_metadata,
    )


def response_spectrum_from_correlator(
    result: TwoTimeCorrelatorResult,
    *,
    momenta: Iterable[float] | float | None = None,
    frequencies: Iterable[float] | float | None = None,
    window: str = "hann",
    subtract_mean: bool = False,
) -> ResponseSpectrumResult:
    if result.connected:
        raise ValueError("linear response requires raw two-time correlator data, not connected correlators")

    times = _validated_time_grid(result.times)
    if times[0] < -1e-12:
        raise ValueError("linear-response analysis requires non-negative times")
    values = np.asarray(result.values, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != times.size:
        raise ValueError(
            "two-time correlator data must have shape "
            "(num_times, num_measure_sites, num_source_sites)"
        )

    measure_sites = tuple(
        int(site)
        for site in getattr(result, "measure_sites", tuple(range(values.shape[1])))
    )
    source_sites = tuple(
        int(site)
        for site in getattr(result, "source_sites", tuple(range(values.shape[2])))
    )
    if values.shape[1] != len(measure_sites) or values.shape[2] != len(source_sites):
        raise ValueError("two-time correlator site metadata does not match the data tensor shape")
    if measure_sites != source_sites:
        raise ValueError(
            "linear response currently requires matching measure_sites and source_sites"
        )

    num_sites = int(getattr(result, "num_sites", max(measure_sites, default=-1) + 1))
    momenta_array = _normalize_momenta(num_sites, momenta)
    frequency_grid = _normalize_frequency_grid(times, frequencies)
    commutator = values - np.swapaxes(values.conj(), 1, 2)
    time_response = -1.0j * _momentum_resolve_tensor(
        commutator,
        momenta_array,
        measure_positions=np.asarray(measure_sites, dtype=np.float64),
        source_positions=np.asarray(source_sites, dtype=np.float64),
    )

    relative_times = times - times[0]
    signal = time_response.copy()
    if subtract_mean:
        signal -= np.mean(signal, axis=0, keepdims=True)

    weights = _window_weights(times.size, window)
    signal *= weights[:, None]
    phase = np.exp(1.0j * frequency_grid[:, None] * relative_times[None, :])
    spectrum = np.trapezoid(
        phase[:, :, None] * signal[None, :, :],
        x=relative_times,
        axis=1,
    )
    return ResponseSpectrumResult(
        model_name=result.model_name,
        solver=result.solver,
        pauli=result.pauli,
        num_sites=num_sites,
        times=times,
        frequencies=frequency_grid,
        momenta=momenta_array,
        time_response=time_response,
        values=np.asarray(spectrum, dtype=np.complex128),
        measure_sites=measure_sites,
        source_sites=source_sites,
        window=window,
        subtract_mean=subtract_mean,
        model_metadata=result.model_metadata,
    )


def _normalize_pauli_axis(pauli: str) -> str:
    axis = str(pauli).upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("pauli axis must be one of 'X', 'Y', or 'Z'")
    return axis


def _resolve_exact_solver(solver: object) -> ExactDiagonalizationSolver:
    if isinstance(solver, ExactDiagonalizationSolver):
        return solver
    exact_solver = getattr(solver, "exact_solver", None)
    if isinstance(exact_solver, ExactDiagonalizationSolver):
        return exact_solver
    raise ValueError("two-time correlators currently require an exact-diagonalization solver backend")


def _try_resolve_exact_solver(solver: object) -> ExactDiagonalizationSolver | None:
    try:
        return _resolve_exact_solver(solver)
    except ValueError:
        return None


def _resolve_exact_dimension_limit(solver: object) -> int:
    exact_solver = _try_resolve_exact_solver(solver)
    if exact_solver is not None:
        return exact_solver.max_dimension
    limit = getattr(solver, "max_dimension", None)
    if isinstance(limit, int) and limit > 0:
        return limit
    return 4096


def _resolve_tensor_network_solver(solver: object) -> RustTensorNetworkSolver | None:
    if isinstance(solver, RustTensorNetworkSolver):
        return solver
    tensor_solver = getattr(solver, "tensor_network_solver", None)
    if isinstance(tensor_solver, RustTensorNetworkSolver):
        return tensor_solver
    return None


def _supports_exact_two_time_initial_state(initial_state: InitialStateLike | None) -> bool:
    if initial_state is None:
        return True
    if isinstance(initial_state, str):
        return True
    if isinstance(initial_state, np.ndarray):
        return True
    if isinstance(initial_state, GroundStateResult):
        return initial_state.ground_state is not None
    if isinstance(initial_state, TimeEvolutionResult):
        return initial_state.final_state is not None
    return False


def _extract_backend_state_for_analysis(initial_state: InitialStateLike | None) -> object | None:
    if isinstance(initial_state, (GroundStateResult, TimeEvolutionResult)):
        return initial_state.backend_state
    if initial_state is None or isinstance(initial_state, (str, np.ndarray)):
        return None
    if hasattr(initial_state, "num_sites") and hasattr(initial_state, "solver"):
        return initial_state
    return None


def _prepared_state_model_name(state: InitialStateLike | None) -> str | None:
    backend_state = _extract_backend_state_for_analysis(state)
    if backend_state is not None:
        return getattr(backend_state, "model_name", None)
    if isinstance(state, (GroundStateResult, TimeEvolutionResult)):
        return str(state.model_name)
    return None


def _validate_prepared_state_model(
    *,
    target_model: object,
    state: InitialStateLike | None,
    state_model: object | None,
    role: str,
) -> None:
    if state is None or isinstance(state, (str, np.ndarray)):
        return

    target_num_sites = int(getattr(target_model, "num_sites"))
    target_model_name = str(getattr(target_model, "model_name", type(target_model).__name__))
    backend_state = _extract_backend_state_for_analysis(state)
    prepared_model_name = _prepared_state_model_name(state)

    if state_model is not None:
        declared_num_sites = int(getattr(state_model, "num_sites"))
        if declared_num_sites != target_num_sites:
            raise ValueError(
                f"{role} state model has {declared_num_sites} sites but the target model has "
                f"{target_num_sites}"
            )
        declared_model_name = str(getattr(state_model, "model_name", type(state_model).__name__))
        if (
            prepared_model_name is not None
            and prepared_model_name not in {declared_model_name, "compressed_state"}
        ):
            raise ValueError(
                f"{role} state reports model '{prepared_model_name}' but "
                f"{role}_state_model declares '{declared_model_name}'"
            )
        return

    if backend_state is None:
        return
    if prepared_model_name is None or prepared_model_name in {target_model_name, "compressed_state"}:
        return
    raise ValueError(
        f"{role} backend state was prepared for model '{prepared_model_name}', not "
        f"target model '{target_model_name}'; pass {role}_state_model=... to explicitly "
        "allow cross-model reuse"
    )


def _strongest_dqpt_candidate(candidates: tuple[DQPTCandidate, ...]) -> DQPTCandidate | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate.cusp_strength),
            float(candidate.prominence),
            -float(candidate.time),
        ),
    )


def _state_descriptor(state: InitialStateLike | None) -> dict[str, object] | None:
    if state is None:
        return {"kind": "none"}
    if isinstance(state, str):
        return {"kind": "product_label", "label": state}
    if isinstance(state, np.ndarray):
        dense = np.asarray(state, dtype=np.complex128)
        return {
            "kind": "dense_state",
            "dimension": int(dense.size),
            "hash": _dense_state_hash(dense),
        }
    if isinstance(state, GroundStateResult):
        descriptor: dict[str, object] = {
            "kind": "ground_state_result",
            "model_name": state.model_name,
            "solver": state.solver,
            "has_dense_state": state.ground_state is not None,
            "has_backend_state": state.backend_state is not None,
        }
        if state.ground_state is not None:
            descriptor["dense_hash"] = _dense_state_hash(state.ground_state)
            descriptor["dimension"] = int(np.asarray(state.ground_state).size)
        if state.backend_state is not None:
            descriptor["backend_model_name"] = getattr(state.backend_state, "model_name", None)
            descriptor["backend_solver"] = getattr(state.backend_state, "solver", None)
            descriptor["backend_num_sites"] = getattr(state.backend_state, "num_sites", None)
        return descriptor
    if isinstance(state, TimeEvolutionResult):
        descriptor = {
            "kind": "time_evolution_result",
            "model_name": state.model_name,
            "solver": state.solver,
            "has_dense_state": state.final_state is not None,
            "has_backend_state": state.backend_state is not None,
        }
        if state.final_state is not None:
            descriptor["dense_hash"] = _dense_state_hash(state.final_state)
            descriptor["dimension"] = int(np.asarray(state.final_state).size)
        if state.backend_state is not None:
            descriptor["backend_model_name"] = getattr(state.backend_state, "model_name", None)
            descriptor["backend_solver"] = getattr(state.backend_state, "solver", None)
            descriptor["backend_num_sites"] = getattr(state.backend_state, "num_sites", None)
        return descriptor
    backend_state = _extract_backend_state_for_analysis(state)
    if backend_state is not None:
        return {
            "kind": "backend_state",
            "model_name": getattr(backend_state, "model_name", None),
            "solver": getattr(backend_state, "solver", None),
            "num_sites": getattr(backend_state, "num_sites", None),
        }
    return {"kind": type(state).__name__}


def _dense_state_hash(state: np.ndarray) -> str:
    values = np.asarray(state, dtype=np.complex128).ravel()
    norm = np.linalg.norm(values)
    if norm > 1e-30:
        values = values / norm
    return hashlib.sha256(values.view(np.float64).tobytes()).hexdigest()


def _should_use_rust_two_time_correlator(
    *,
    solver: object,
    model: object,
    times: Iterable[float],
    pauli: str,
    initial_state: InitialStateLike | None,
) -> bool:
    tensor_solver = _resolve_tensor_network_solver(solver)
    if tensor_solver is None:
        return False
    exact_solver = _resolve_exact_solver(solver)
    dimension = _model_dimension(model)
    if isinstance(initial_state, GroundStateResult) and initial_state.backend_state is not None:
        if not _supports_exact_two_time_initial_state(initial_state):
            return tensor_solver.supports_two_time_ground_state_correlator(
                model,
                times,
                pauli=pauli,
                reference_state=initial_state,
            )
    if initial_state is None and dimension > exact_solver.max_dimension:
        return tensor_solver.is_available() and tensor_solver.has_transition_bindings()
    return False


def _should_use_rust_entanglement_spectrum(
    *,
    solver: object,
    model: object,
    subsystem: tuple[int, ...],
    initial_state: InitialStateLike | None,
) -> bool:
    tensor_solver = _resolve_tensor_network_solver(solver)
    if tensor_solver is None:
        return False
    backend_state = _extract_backend_state_for_analysis(initial_state)
    if backend_state is not None:
        return tensor_solver.supports_entanglement_spectrum(
            backend_state,
            subsystem=subsystem,
        )
    if initial_state is not None:
        return False
    if not _supports_dmrg_model_for_entanglement(model):
        return False
    if _entropy_bond_for_subsystem_for_entanglement(
        getattr(model, "num_sites"),
        subsystem,
    ) is None:
        return False
    max_dimension = _resolve_exact_dimension_limit(solver)
    return _model_dimension(model) > max_dimension and tensor_solver.has_entanglement_spectrum_bindings()


def _should_use_rust_loschmidt_echo(
    *,
    solver: object,
    model: object,
    times: Iterable[float],
    initial_state: InitialStateLike | None,
    reference_state: InitialStateLike | None,
) -> bool:
    tensor_solver = _resolve_tensor_network_solver(solver)
    if tensor_solver is None:
        return False

    exact_limit = _resolve_exact_dimension_limit(solver)
    dimension = _model_dimension(model)
    needs_rust = (
        dimension > exact_limit
        or _extract_backend_state_for_analysis(initial_state) is not None
        or _extract_backend_state_for_analysis(reference_state) is not None
    )
    if not needs_rust:
        return False

    resolved_initial = initial_state
    if resolved_initial is None:
        if not tensor_solver.is_available():
            return False
        resolved_initial = "all_up"

    return tensor_solver.supports_loschmidt_echo(
        model,
        times,
        initial_state=resolved_initial,
        reference_state=reference_state,
    )


def _supports_dmrg_model_for_entanglement(model: object) -> bool:
    if isinstance(model, TransverseFieldIsing1D):
        return model.num_sites >= 2 and model.boundary == "open"
    if isinstance(model, HeisenbergXXZ1D):
        return model.num_sites >= 2 and model.boundary == "open"
    if isinstance(model, HeisenbergXYZ1D):
        return model.num_sites >= 2 and model.boundary == "open"
    return False


def _entropy_bond_for_subsystem_for_entanglement(
    num_sites: int,
    subsystem: tuple[int, ...],
) -> int | None:
    if not subsystem:
        return None
    if subsystem != tuple(range(subsystem[-1] + 1)):
        return None
    if subsystem[-1] >= num_sites - 1:
        return None
    return subsystem[-1]


def _resolve_exact_entanglement_state(
    model: object,
    *,
    initial_state: InitialStateLike | None,
    max_dimension: int,
) -> np.ndarray:
    dimension = _model_dimension(model)
    if initial_state is None:
        if dimension > max_dimension:
            raise ValueError(
                "entanglement spectra currently require dense-state analysis, "
                f"limited to dimension {max_dimension}, got {dimension}"
            )
        hamiltonian = np.asarray(model.hamiltonian(), dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        del eigenvalues
        return np.asarray(eigenvectors[:, 0], dtype=np.complex128)

    if isinstance(initial_state, GroundStateResult):
        if initial_state.ground_state is None:
            raise ValueError(
                "the provided ground-state result does not carry a dense state; "
                "use a supported Rust backend state or a dense exact result"
            )
        state = np.asarray(initial_state.ground_state, dtype=np.complex128)
    elif isinstance(initial_state, TimeEvolutionResult):
        if initial_state.final_state is None:
            raise ValueError(
                "the provided time-evolution result does not carry a dense final state; "
                "use a supported Rust backend state or a dense exact result"
            )
        state = np.asarray(initial_state.final_state, dtype=np.complex128)
    elif isinstance(initial_state, np.ndarray):
        state = np.asarray(initial_state, dtype=np.complex128)
    elif isinstance(initial_state, str):
        if dimension > max_dimension:
            raise ValueError(
                "dense-state entanglement spectra for string-labeled states are "
                f"limited to dimension {max_dimension}, got {dimension}"
            )
        state = make_initial_state(getattr(model, "num_sites"), initial_state)
    else:
        raise ValueError(
            "entanglement spectra currently require a dense state, a dense result, "
            "or a supported Rust tensor-network backend state"
        )

    if state.ndim != 1:
        raise ValueError("state vectors for entanglement-spectrum analysis must be one-dimensional")
    if state.size > max_dimension:
        raise ValueError(
            "dense-state entanglement spectra are limited to "
            f"dimension {max_dimension}, got {state.size}"
        )
    return state


def _resolve_exact_loschmidt_state(
    model: object,
    *,
    state: InitialStateLike | None,
    max_dimension: int,
    role: str,
) -> np.ndarray:
    num_sites = int(getattr(model, "num_sites"))
    dimension = _model_dimension(model)

    if state is None:
        hamiltonian = np.asarray(model.hamiltonian(), dtype=np.complex128)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        del eigenvalues
        vector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    elif isinstance(state, GroundStateResult):
        if state.ground_state is None:
            raise ValueError(
                f"the provided {role} ground-state result does not carry a dense state; "
                "Loschmidt echo currently requires dense-state analysis"
            )
        vector = np.asarray(state.ground_state, dtype=np.complex128)
    elif isinstance(state, TimeEvolutionResult):
        if state.final_state is None:
            raise ValueError(
                f"the provided {role} time-evolution result does not carry a dense final state; "
                "Loschmidt echo currently requires dense-state analysis"
            )
        vector = np.asarray(state.final_state, dtype=np.complex128)
    elif isinstance(state, np.ndarray):
        vector = np.asarray(state, dtype=np.complex128)
    elif isinstance(state, str):
        vector = make_initial_state(num_sites, state)
    else:
        raise ValueError(
            f"{role} state for Loschmidt echo must be a dense statevector, "
            "a dense result, or a string-labeled product state"
        )

    if vector.ndim != 1:
        raise ValueError(f"{role} state for Loschmidt echo must be one-dimensional")
    if vector.size != dimension:
        raise ValueError(
            f"{role} state dimension {vector.size} does not match model dimension {dimension}"
        )
    if vector.size > max_dimension:
        raise ValueError(
            "Loschmidt echo currently requires exact diagonalization, "
            f"limited to dimension {max_dimension}, got {vector.size}"
        )
    norm = np.linalg.norm(vector)
    if norm <= 1e-30:
        raise ValueError(f"{role} state for Loschmidt echo must have non-zero norm")
    return vector / norm


def _rust_entanglement_spectrum(
    qpu: ModelQPU,
    model: object,
    *,
    subsystem: tuple[int, ...],
    initial_state: InitialStateLike | None,
    tensor_solver: RustTensorNetworkSolver,
    num_levels: int | None,
) -> EntanglementSpectrumResult:
    backend_state = _extract_backend_state_for_analysis(initial_state)
    solver_name = getattr(initial_state, "solver", "tensor_network")
    if backend_state is None:
        reference = qpu.ground_state(model)
        backend_state = reference.backend_state
        solver_name = reference.solver
    if backend_state is None:
        raise ValueError(
            "Rust entanglement spectra require a tensor-network backend state or "
            "a supported ground-state model that can prepare one"
        )

    full_spectrum = tensor_solver.entanglement_spectrum(
        backend_state,
        subsystem=subsystem,
    )
    entropy = float(-np.sum(full_spectrum * np.log2(full_spectrum)))
    spectrum = full_spectrum
    if num_levels is not None:
        spectrum = spectrum[:num_levels]
    return EntanglementSpectrumResult(
        model_name=getattr(model, "model_name", type(model).__name__),
        solver=str(solver_name),
        num_sites=getattr(model, "num_sites"),
        subsystem=subsystem,
        eigenvalues=np.asarray(spectrum, dtype=np.float64),
        entropy=entropy,
        model_metadata=_model_metadata(model),
    )


def _rust_loschmidt_echo(
    qpu: ModelQPU,
    model: object,
    times: Iterable[float],
    *,
    initial_state: InitialStateLike | None,
    reference_state: InitialStateLike | None,
    tensor_solver: RustTensorNetworkSolver,
) -> LoschmidtEchoResult:
    resolved_initial = initial_state
    if resolved_initial is None:
        resolved_initial = qpu.ground_state(model)

    raw = tensor_solver.loschmidt_echo(
        model,
        times,
        initial_state=resolved_initial,
        reference_state=reference_state,
    )
    return LoschmidtEchoResult(
        model_name=str(raw.get("model_name", getattr(model, "model_name", type(model).__name__))),
        solver=str(raw.get("solver", "tdvp_overlap")),
        num_sites=getattr(model, "num_sites"),
        times=np.asarray(raw.get("times", list(times)), dtype=np.float64),
        amplitudes=np.asarray(raw.get("amplitudes", []), dtype=np.complex128),
        backend_state=raw.get("state_handle"),
        model_metadata=_model_metadata(model),
    )


def analyze_dqpt_from_loschmidt(
    result: LoschmidtEchoResult,
    *,
    min_prominence: float = 0.0,
    min_cusp: float = 0.0,
    mode: str = "maxima",
) -> DQPTDiagnosticsResult:
    if min_prominence < 0.0:
        raise ValueError("min_prominence must be non-negative")
    if min_cusp < 0.0:
        raise ValueError("min_cusp must be non-negative")
    if mode not in {"maxima", "extrema"}:
        raise ValueError("mode must be 'maxima' or 'extrema'")

    times = np.asarray(result.times, dtype=np.float64)
    return_rate = np.asarray(result.return_rate, dtype=np.float64)
    candidates: list[DQPTCandidate] = []

    for index in range(1, len(times) - 1):
        left_time = float(times[index - 1])
        center_time = float(times[index])
        right_time = float(times[index + 1])
        left_dt = center_time - left_time
        right_dt = right_time - center_time
        if left_dt <= 0.0 or right_dt <= 0.0:
            continue

        left_value = float(return_rate[index - 1])
        center_value = float(return_rate[index])
        right_value = float(return_rate[index + 1])

        is_maximum = center_value >= left_value and center_value >= right_value
        is_minimum = center_value <= left_value and center_value <= right_value
        if mode == "maxima":
            if not is_maximum or (center_value == left_value and center_value == right_value):
                continue
            prominence = center_value - max(left_value, right_value)
        else:
            if not (is_maximum or is_minimum):
                continue
            prominence = min(
                abs(center_value - left_value),
                abs(center_value - right_value),
            )

        left_slope = (center_value - left_value) / left_dt
        right_slope = (right_value - center_value) / right_dt
        cusp_strength = abs(right_slope - left_slope)
        if prominence < min_prominence or cusp_strength < min_cusp:
            continue

        candidates.append(
            DQPTCandidate(
                index=index,
                time=center_time,
                return_rate=center_value,
                prominence=float(prominence),
                cusp_strength=float(cusp_strength),
                left_slope=float(left_slope),
                right_slope=float(right_slope),
            )
        )

    amplitudes = None
    if result.amplitudes is not None:
        amplitudes = np.asarray(result.amplitudes, dtype=np.complex128)
    return DQPTDiagnosticsResult(
        model_name=result.model_name,
        solver=result.solver,
        num_sites=result.num_sites,
        times=times,
        return_rate=return_rate,
        candidates=tuple(candidates),
        amplitudes=amplitudes,
        backend_state=result.backend_state,
        model_metadata=result.model_metadata,
    )


def _rust_two_time_correlator(
    qpu: ModelQPU,
    model: object,
    times: Iterable[float],
    *,
    pauli: str,
    connected: bool,
    initial_state: InitialStateLike | None,
    tensor_solver: RustTensorNetworkSolver,
    measure_sites: tuple[int, ...],
    source_sites: tuple[int, ...],
) -> TwoTimeCorrelatorResult:
    times_array = _normalize_correlation_times(times)
    if initial_state is None:
        reference = qpu.ground_state(model)
    elif isinstance(initial_state, GroundStateResult):
        reference = initial_state
    else:
        raise ValueError(
            "Rust two-time correlators currently support only ground-state references "
            "from the Rust DMRG path"
        )
    if reference.backend_state is None:
        raise ValueError(
            "Rust two-time correlators require a ground-state result with a tensor-network backend state"
        )
    if not tensor_solver.supports_two_time_ground_state_correlator(
        model,
        times_array,
        pauli=pauli,
        reference_state=reference,
    ):
        raise ValueError(
            "requested model/time grid is not supported by the Rust two-time correlator path"
        )

    num_sites = getattr(model, "num_sites")
    measure_labels = tuple(f"{pauli}{site}" for site in measure_sites)
    source_labels = tuple(f"{pauli}{site}" for site in source_sites)
    base_labels = tuple(dict.fromkeys((*measure_labels, *source_labels)))
    base_measurement = tensor_solver.transition_observable_evolution(
        model,
        [0.0],
        reference_state=reference.backend_state,
        source_state=reference.backend_state,
        observables=base_labels,
    )
    base_observables = base_measurement["observables"]
    initial_single = np.asarray(
        [np.real(base_observables[label][0]) for label in source_labels],
        dtype=np.float64,
    )
    dynamic_single = np.repeat(
        np.asarray(
            [np.real(base_observables[label][0]) for label in measure_labels],
            dtype=np.float64,
        )[None, :],
        times_array.size,
        axis=0,
    )
    phase = np.exp(1.0j * reference.ground_state_energy * times_array)

    values = np.zeros((times_array.size, len(measure_sites), len(source_sites)), dtype=np.complex128)
    for source_index, source_site in enumerate(source_sites):
        source_state = tensor_solver.apply_local_pauli(
            reference.backend_state,
            pauli=pauli,
            site=source_site,
        )
        transition = tensor_solver.transition_observable_evolution(
            model,
            times_array,
            reference_state=reference.backend_state,
            source_state=source_state,
            observables=measure_labels,
        )
        for measure_index, label in enumerate(measure_labels):
            values[:, measure_index, source_index] = phase * transition["observables"][label]

    if connected:
        values = values - dynamic_single[:, :, None] * initial_single[None, None, :]

    return TwoTimeCorrelatorResult(
        model_name=getattr(model, "model_name", type(model).__name__),
        solver="tdvp_transition",
        pauli=pauli,
        connected=connected,
        num_sites=num_sites,
        times=times_array,
        values=values,
        dynamic_single_site_expectations=dynamic_single,
        initial_single_site_expectations=initial_single,
        measure_sites=measure_sites,
        source_sites=source_sites,
        model_metadata=_model_metadata(model),
    )


def _normalize_correlation_times(times: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(times), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("times must be a non-empty one-dimensional grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("times must be finite")
    return values


def _model_dimension(model: object) -> int:
    dimension = getattr(model, "dimension", None)
    if dimension is not None:
        return int(dimension)
    return 1 << int(getattr(model, "num_sites"))


def _validated_time_grid(times: np.ndarray | Iterable[float]) -> np.ndarray:
    values = np.asarray(list(times) if not isinstance(times, np.ndarray) else times, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("frequency-domain structure factors require at least two time points")
    if not np.all(np.isfinite(values)):
        raise ValueError("time grid must be finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("time grid must be strictly increasing for Fourier analysis")
    return values


def _normalize_frequency_grid(
    times: np.ndarray,
    frequencies: Iterable[float] | float | None,
) -> np.ndarray:
    if frequencies is None:
        deltas = np.diff(times)
        spacing = float(np.median(deltas))
        return np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(times.size, d=spacing)).astype(np.float64)
    if np.isscalar(frequencies):
        values = np.asarray([frequencies], dtype=np.float64)
    else:
        values = np.asarray(list(frequencies), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("frequencies must be a non-empty one-dimensional grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("frequencies must be finite")
    return values


def _window_weights(num_times: int, window: str) -> np.ndarray:
    window_name = str(window).lower()
    if window_name == "none":
        return np.ones(num_times, dtype=np.float64)
    if window_name == "hann":
        return np.hanning(num_times).astype(np.float64)
    if window_name == "hamming":
        return np.hamming(num_times).astype(np.float64)
    raise ValueError("window must be 'none', 'hann', or 'hamming'")


def _single_site_label(pauli: str, site: int) -> str:
    return f"{pauli}{site}"


def _two_site_label(pauli: str, left: int, right: int) -> str:
    return f"{pauli}{left}{pauli}{right}"


def _correlation_observable_labels(
    num_sites: int,
    pauli: str,
    *,
    include_single_site: bool,
) -> tuple[str, ...]:
    labels: list[str] = []
    if include_single_site:
        labels.extend(_single_site_label(pauli, site) for site in range(num_sites))
    labels.extend(
        _two_site_label(pauli, left, right)
        for left in range(num_sites)
        for right in range(left + 1, num_sites)
    )
    return tuple(labels)


def _single_site_expectations_from_observables(
    observables: dict[str, float],
    num_sites: int,
    pauli: str,
) -> np.ndarray:
    return np.asarray(
        [float(observables[_single_site_label(pauli, site)]) for site in range(num_sites)],
        dtype=np.float64,
    )


def _correlation_matrix_from_observables(
    observables: dict[str, float],
    num_sites: int,
    pauli: str,
    *,
    connected: bool,
    single_site_expectations: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((num_sites, num_sites), dtype=np.float64)
    for site in range(num_sites):
        matrix[site, site] = 1.0 - single_site_expectations[site] ** 2 if connected else 1.0

    for left in range(num_sites):
        for right in range(left + 1, num_sites):
            value = float(observables[_two_site_label(pauli, left, right)])
            if connected:
                value -= single_site_expectations[left] * single_site_expectations[right]
            matrix[left, right] = value
            matrix[right, left] = value
    return matrix


def _normalize_momenta(num_sites: int, momenta: Iterable[float] | float | None) -> np.ndarray:
    if num_sites < 1:
        raise ValueError("structure-factor calculations require at least one site")
    if momenta is None:
        return (2.0 * np.pi * np.arange(num_sites, dtype=np.float64)) / float(num_sites)
    if np.isscalar(momenta):
        values = np.asarray([momenta], dtype=np.float64)
    else:
        values = np.asarray(list(momenta), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("momenta must be a non-empty one-dimensional grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("momenta must be finite")
    return values


def _static_structure_factor_from_matrix(
    matrix: np.ndarray,
    momenta: np.ndarray,
) -> np.ndarray:
    num_sites = matrix.shape[0]
    positions = np.arange(num_sites, dtype=np.float64)
    values = np.zeros_like(momenta, dtype=np.complex128)
    for index, momentum in enumerate(momenta):
        phase = np.exp(1.0j * momentum * (positions[:, None] - positions[None, :]))
        values[index] = np.sum(phase * matrix) / float(num_sites)
    return np.asarray(np.real_if_close(values), dtype=np.float64)


def _momentum_resolve_tensor(
    values: np.ndarray,
    momenta: np.ndarray,
    *,
    measure_positions: np.ndarray | None = None,
    source_positions: np.ndarray | None = None,
) -> np.ndarray:
    num_times, num_measure_sites, num_source_sites = values.shape
    if measure_positions is None:
        measure_positions = np.arange(num_measure_sites, dtype=np.float64)
    else:
        measure_positions = np.asarray(measure_positions, dtype=np.float64)
    if source_positions is None:
        source_positions = np.arange(num_source_sites, dtype=np.float64)
    else:
        source_positions = np.asarray(source_positions, dtype=np.float64)
    if measure_positions.shape != (num_measure_sites,):
        raise ValueError("measure_positions must match the measure-site axis of the tensor")
    if source_positions.shape != (num_source_sites,):
        raise ValueError("source_positions must match the source-site axis of the tensor")

    resolved = np.zeros((num_times, len(momenta)), dtype=np.complex128)
    normalization = np.sqrt(float(num_measure_sites * num_source_sites))
    for index, momentum in enumerate(momenta):
        phase = np.exp(
            1.0j
            * momentum
            * (measure_positions[:, None] - source_positions[None, :])
        )
        resolved[:, index] = np.sum(values * phase[None, :, :], axis=(1, 2)) / normalization
    return resolved


def _dynamic_structure_factor_from_traces(
    observables: dict[str, np.ndarray],
    num_sites: int,
    momenta: np.ndarray,
    pauli: str,
    *,
    connected: bool,
) -> np.ndarray:
    if observables:
        reference = next(iter(observables.values()))
        num_times = len(np.asarray(reference, dtype=np.float64))
    else:
        num_times = 0
    values = np.zeros((num_times, len(momenta)), dtype=np.float64)
    single_site = None
    if connected:
        single_site = np.stack(
            [
                np.asarray(observables[_single_site_label(pauli, site)], dtype=np.float64)
                for site in range(num_sites)
            ],
            axis=1,
        )
        diagonal = 1.0 - single_site**2
    else:
        diagonal = np.ones((num_times, num_sites), dtype=np.float64)

    values += np.sum(diagonal, axis=1, keepdims=True) / float(num_sites)
    for left in range(num_sites):
        for right in range(left + 1, num_sites):
            trace = np.asarray(observables[_two_site_label(pauli, left, right)], dtype=np.float64)
            if connected:
                assert single_site is not None
                trace = trace - single_site[:, left] * single_site[:, right]
            weight = 2.0 * np.cos(momenta * float(left - right)) / float(num_sites)
            values += trace[:, None] * weight[None, :]
    return values
