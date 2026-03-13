"""Persistence helpers for Rust-backed tensor-network states and research results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nqpu._compat import get_rust_bindings

from .experiments import (
    DQPTCandidate,
    DQPTDiagnosticsResult,
    DQPTScanPoint,
    DQPTScanResult,
    DynamicStructureFactorResult,
    EntanglementSpectrumResult,
    FrequencyStructureFactorResult,
    LoschmidtEchoResult,
    ResponseSpectrumResult,
    SweepPoint,
    SweepResult,
    TwoTimeCorrelatorResult,
)
from .solvers import GroundStateResult, TimeEvolutionResult


def _resolve_backend_state(
    state_or_result: GroundStateResult | TimeEvolutionResult | LoschmidtEchoResult | DQPTDiagnosticsResult | object,
) -> object:
    if isinstance(
        state_or_result,
        (GroundStateResult, TimeEvolutionResult, LoschmidtEchoResult, DQPTDiagnosticsResult),
    ):
        backend_state = state_or_result.backend_state
    else:
        backend_state = state_or_result

    if backend_state is None:
        raise ValueError("tensor-network checkpointing requires a Rust-backed state handle")
    if not hasattr(backend_state, "save_json"):
        raise ValueError("object does not expose Rust tensor-network checkpoint methods")
    return backend_state


def dump_tensor_network_state(
    state_or_result: GroundStateResult | TimeEvolutionResult | LoschmidtEchoResult | DQPTDiagnosticsResult | object,
) -> str:
    backend_state = _resolve_backend_state(state_or_result)
    return str(backend_state.to_json())


def save_tensor_network_state(
    state_or_result: GroundStateResult | TimeEvolutionResult | LoschmidtEchoResult | DQPTDiagnosticsResult | object,
    path: str | Path,
) -> Path:
    backend_state = _resolve_backend_state(state_or_result)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    backend_state.save_json(str(destination))
    return destination


def load_tensor_network_state(path: str | Path) -> object:
    bindings = get_rust_bindings()
    if bindings is None or not hasattr(bindings, "TensorNetworkState1D"):
        raise RuntimeError(
            "tensor-network checkpoint loading requires the optional Rust bindings"
        )
    source = Path(path)
    return bindings.TensorNetworkState1D.load_json(str(source))


def restore_tensor_network_state(payload: str) -> object:
    bindings = get_rust_bindings()
    if bindings is None or not hasattr(bindings, "TensorNetworkState1D"):
        raise RuntimeError(
            "tensor-network checkpoint loading requires the optional Rust bindings"
        )
    return bindings.TensorNetworkState1D.from_json(payload)


def _encode_complex_array(array: np.ndarray | None) -> dict[str, object] | None:
    if array is None:
        return None
    values = np.asarray(array, dtype=np.complex128)
    return {
        "shape": list(values.shape),
        "real": values.real.ravel().tolist(),
        "imag": values.imag.ravel().tolist(),
    }


def _decode_complex_array(payload: dict[str, object] | None) -> np.ndarray | None:
    if payload is None:
        return None
    shape = tuple(int(value) for value in payload["shape"])
    real = np.asarray(payload["real"], dtype=np.float64)
    imag = np.asarray(payload["imag"], dtype=np.float64)
    return (real + 1.0j * imag).reshape(shape)


def _sidecar_checkpoint_path(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.state.json")
    return path.with_name(f"{path.name}.state.json")


def _dqpt_scan_sidecar_dir(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.dqpt_points")
    return path.with_name(f"{path.name}.dqpt_points")


def _dqpt_scan_point_checkpoint_path(path: Path, index: int) -> Path:
    return _dqpt_scan_sidecar_dir(path) / f"point_{index:06d}.state.json"


def _sweep_sidecar_dir(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.points")
    return path.with_name(f"{path.name}.points")


def _sweep_point_checkpoint_path(path: Path, index: int) -> Path:
    return _sweep_sidecar_dir(path) / f"point_{index:06d}.state.json"


def _encode_real_array(array: np.ndarray | None) -> list[float] | None:
    if array is None:
        return None
    return np.asarray(array, dtype=np.float64).tolist()


def _decode_real_array(payload: object) -> np.ndarray | None:
    if payload is None:
        return None
    return np.asarray(payload, dtype=np.float64)


def _encode_optional_real_array(array: np.ndarray | None) -> list[float | None] | None:
    if array is None:
        return None
    values = np.asarray(array, dtype=np.float64)
    return [None if not np.isfinite(value) else float(value) for value in values]


def _decode_optional_real_array(payload: object) -> np.ndarray | None:
    if payload is None:
        return None
    values = [
        np.nan if value is None else float(value)
        for value in payload
    ]
    return np.asarray(values, dtype=np.float64)


def _encode_optional_real_ndarray(array: np.ndarray | None) -> dict[str, object] | None:
    if array is None:
        return None
    values = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "data": [None if not np.isfinite(value) else float(value) for value in values.ravel()],
    }


def _decode_optional_real_ndarray(payload: dict[str, object] | None) -> np.ndarray | None:
    if payload is None:
        return None
    shape = tuple(int(value) for value in payload["shape"])
    data = [
        np.nan if value is None else float(value)
        for value in payload["data"]
    ]
    return np.asarray(data, dtype=np.float64).reshape(shape)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_ground_state_result(
    result: GroundStateResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    backend_state_checkpoint = None
    if include_backend_state and result.backend_state is not None:
        state_path = _sidecar_checkpoint_path(destination)
        save_tensor_network_state(result, state_path)
        backend_state_checkpoint = state_path.name

    payload = {
        "format_version": 1,
        "result_type": "ground_state",
        "model_name": result.model_name,
        "solver": result.solver,
        "dimension": int(result.dimension),
        "ground_state_energy": float(result.ground_state_energy),
        "spectral_gap": None if result.spectral_gap is None else float(result.spectral_gap),
        "eigenvalues": np.asarray(result.eigenvalues, dtype=np.float64).tolist(),
        "ground_state": _encode_complex_array(result.ground_state),
        "observables": {label: float(value) for label, value in result.observables.items()},
        "entanglement_entropy": result.entanglement_entropy,
        "model_metadata": result.model_metadata,
        "backend_state_checkpoint": backend_state_checkpoint,
    }
    _write_json(destination, payload)
    return destination


def load_ground_state_result(path: str | Path) -> GroundStateResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "ground_state":
        raise ValueError("checkpoint does not contain a ground-state result")

    backend_state = None
    checkpoint_name = payload.get("backend_state_checkpoint")
    if checkpoint_name:
        backend_state = load_tensor_network_state(source.with_name(str(checkpoint_name)))

    return GroundStateResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        dimension=int(payload["dimension"]),
        ground_state_energy=float(payload["ground_state_energy"]),
        spectral_gap=None
        if payload["spectral_gap"] is None
        else float(payload["spectral_gap"]),
        eigenvalues=np.asarray(payload["eigenvalues"], dtype=np.float64),
        ground_state=_decode_complex_array(payload.get("ground_state")),
        observables={label: float(value) for label, value in dict(payload["observables"]).items()},
        entanglement_entropy=None
        if payload["entanglement_entropy"] is None
        else float(payload["entanglement_entropy"]),
        backend_state=backend_state,
        model_metadata=payload.get("model_metadata"),
    )


def save_time_evolution_result(
    result: TimeEvolutionResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    backend_state_checkpoint = None
    if include_backend_state and result.backend_state is not None:
        state_path = _sidecar_checkpoint_path(destination)
        save_tensor_network_state(result, state_path)
        backend_state_checkpoint = state_path.name

    payload = {
        "format_version": 1,
        "result_type": "time_evolution",
        "model_name": result.model_name,
        "solver": result.solver,
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "observables": {
            label: np.asarray(values, dtype=np.float64).tolist()
            for label, values in result.observables.items()
        },
        "final_state": _encode_complex_array(result.final_state),
        "entanglement_entropy": _encode_real_array(result.entanglement_entropy),
        "entanglement_subsystem": None
        if result.entanglement_subsystem is None
        else list(result.entanglement_subsystem),
        "model_metadata": result.model_metadata,
        "backend_state_checkpoint": backend_state_checkpoint,
    }
    _write_json(destination, payload)
    return destination


def load_time_evolution_result(path: str | Path) -> TimeEvolutionResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "time_evolution":
        raise ValueError("checkpoint does not contain a time-evolution result")

    backend_state = None
    checkpoint_name = payload.get("backend_state_checkpoint")
    if checkpoint_name:
        backend_state = load_tensor_network_state(source.with_name(str(checkpoint_name)))

    entanglement_subsystem = payload.get("entanglement_subsystem")
    return TimeEvolutionResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        observables={
            label: np.asarray(values, dtype=np.float64)
            for label, values in dict(payload["observables"]).items()
        },
        final_state=_decode_complex_array(payload.get("final_state")),
        backend_state=backend_state,
        entanglement_entropy=_decode_real_array(payload.get("entanglement_entropy")),
        entanglement_subsystem=None
        if entanglement_subsystem is None
        else tuple(int(site) for site in entanglement_subsystem),
        model_metadata=payload.get("model_metadata"),
    )


def save_entanglement_spectrum_result(
    result: EntanglementSpectrumResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    payload = {
        "format_version": 1,
        "result_type": "entanglement_spectrum",
        "model_name": result.model_name,
        "solver": result.solver,
        "num_sites": int(result.num_sites),
        "subsystem": list(result.subsystem),
        "eigenvalues": np.asarray(result.eigenvalues, dtype=np.float64).tolist(),
        "entropy": float(result.entropy),
        "model_metadata": result.model_metadata,
    }
    _write_json(destination, payload)
    return destination


def load_entanglement_spectrum_result(path: str | Path) -> EntanglementSpectrumResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "entanglement_spectrum":
        raise ValueError("checkpoint does not contain an entanglement-spectrum result")

    return EntanglementSpectrumResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        num_sites=int(payload["num_sites"]),
        subsystem=tuple(int(site) for site in payload["subsystem"]),
        eigenvalues=np.asarray(payload["eigenvalues"], dtype=np.float64),
        entropy=float(payload["entropy"]),
        model_metadata=payload.get("model_metadata"),
    )


def save_loschmidt_echo_result(
    result: LoschmidtEchoResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    backend_state_checkpoint = None
    if include_backend_state and result.backend_state is not None:
        state_path = _sidecar_checkpoint_path(destination)
        save_tensor_network_state(result, state_path)
        backend_state_checkpoint = state_path.name

    payload = {
        "format_version": 1,
        "result_type": "loschmidt_echo",
        "model_name": result.model_name,
        "solver": result.solver,
        "num_sites": int(result.num_sites),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "amplitudes": _encode_complex_array(result.amplitudes),
        "model_metadata": result.model_metadata,
        "backend_state_checkpoint": backend_state_checkpoint,
    }
    _write_json(destination, payload)
    return destination


def load_loschmidt_echo_result(path: str | Path) -> LoschmidtEchoResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "loschmidt_echo":
        raise ValueError("checkpoint does not contain a Loschmidt-echo result")

    amplitudes = _decode_complex_array(payload.get("amplitudes"))
    if amplitudes is None:
        raise ValueError("Loschmidt-echo checkpoint is missing amplitudes")
    backend_state = None
    checkpoint_name = payload.get("backend_state_checkpoint")
    if checkpoint_name:
        backend_state = load_tensor_network_state(source.with_name(str(checkpoint_name)))
    return LoschmidtEchoResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        num_sites=int(payload["num_sites"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        amplitudes=np.asarray(amplitudes, dtype=np.complex128),
        backend_state=backend_state,
        model_metadata=payload.get("model_metadata"),
    )


def save_dqpt_diagnostics_result(
    result: DQPTDiagnosticsResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    backend_state_checkpoint = None
    if include_backend_state and result.backend_state is not None:
        state_path = _sidecar_checkpoint_path(destination)
        save_tensor_network_state(result, state_path)
        backend_state_checkpoint = state_path.name

    payload = {
        "format_version": 1,
        "result_type": "dqpt_diagnostics",
        "model_name": result.model_name,
        "solver": result.solver,
        "num_sites": int(result.num_sites),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "return_rate": np.asarray(result.return_rate, dtype=np.float64).tolist(),
        "amplitudes": _encode_complex_array(result.amplitudes),
        "candidates": [
            {
                "index": int(candidate.index),
                "time": float(candidate.time),
                "return_rate": float(candidate.return_rate),
                "prominence": float(candidate.prominence),
                "cusp_strength": float(candidate.cusp_strength),
                "left_slope": float(candidate.left_slope),
                "right_slope": float(candidate.right_slope),
            }
            for candidate in result.candidates
        ],
        "model_metadata": result.model_metadata,
        "backend_state_checkpoint": backend_state_checkpoint,
    }
    _write_json(destination, payload)
    return destination


def load_dqpt_diagnostics_result(path: str | Path) -> DQPTDiagnosticsResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "dqpt_diagnostics":
        raise ValueError("checkpoint does not contain a DQPT-diagnostics result")

    backend_state = None
    checkpoint_name = payload.get("backend_state_checkpoint")
    if checkpoint_name:
        backend_state = load_tensor_network_state(source.with_name(str(checkpoint_name)))

    candidates = tuple(
        DQPTCandidate(
            index=int(candidate["index"]),
            time=float(candidate["time"]),
            return_rate=float(candidate["return_rate"]),
            prominence=float(candidate["prominence"]),
            cusp_strength=float(candidate["cusp_strength"]),
            left_slope=float(candidate["left_slope"]),
            right_slope=float(candidate["right_slope"]),
        )
        for candidate in payload.get("candidates", [])
    )
    amplitudes = _decode_complex_array(payload.get("amplitudes"))
    return DQPTDiagnosticsResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        num_sites=int(payload["num_sites"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        return_rate=np.asarray(payload["return_rate"], dtype=np.float64),
        candidates=candidates,
        amplitudes=None if amplitudes is None else np.asarray(amplitudes, dtype=np.complex128),
        backend_state=backend_state,
        model_metadata=payload.get("model_metadata"),
    )


def save_dqpt_scan_result(
    result: DQPTScanResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    serialized_points = []
    for index, point in enumerate(result.points):
        backend_state_checkpoint = point.backend_state_checkpoint if include_backend_state else None
        if include_backend_state and point.backend_state is not None:
            if backend_state_checkpoint is None:
                state_path = _dqpt_scan_point_checkpoint_path(destination, index)
                backend_state_checkpoint = str(state_path.relative_to(destination.parent))
            state_path = destination.parent / backend_state_checkpoint
            if not state_path.exists():
                save_tensor_network_state(point.backend_state, state_path)
            point.backend_state_checkpoint = backend_state_checkpoint

        serialized_points.append(
            {
                "index": point.index,
                "value": float(point.value),
                "solver": point.solver,
                "return_rate": np.asarray(point.return_rate, dtype=np.float64).tolist(),
                "candidate_times": np.asarray(point.candidate_times, dtype=np.float64).tolist(),
                "candidate_return_rates": np.asarray(
                    point.candidate_return_rates,
                    dtype=np.float64,
                ).tolist(),
                "candidate_prominences": np.asarray(
                    point.candidate_prominences,
                    dtype=np.float64,
                ).tolist(),
                "candidate_cusp_strengths": np.asarray(
                    point.candidate_cusp_strengths,
                    dtype=np.float64,
                ).tolist(),
                "strongest_candidate_time": point.strongest_candidate_time,
                "strongest_candidate_return_rate": point.strongest_candidate_return_rate,
                "strongest_prominence": point.strongest_prominence,
                "strongest_cusp_strength": point.strongest_cusp_strength,
                "model_metadata": point.model_metadata,
                "backend_state_checkpoint": backend_state_checkpoint,
            }
        )

    payload = {
        "format_version": 1,
        "result_type": "dqpt_scan",
        "parameter": result.parameter,
        "model_name": result.model_name,
        "solver": result.solver,
        "values": np.asarray(result.values, dtype=np.float64).tolist(),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "return_rates": _encode_optional_real_ndarray(result.return_rates),
        "candidate_counts": _encode_optional_real_array(result.candidate_counts),
        "max_return_rates": _encode_optional_real_array(result.max_return_rates),
        "strongest_candidate_times": _encode_optional_real_array(result.strongest_candidate_times),
        "strongest_candidate_return_rates": _encode_optional_real_array(
            result.strongest_candidate_return_rates
        ),
        "strongest_prominences": _encode_optional_real_array(result.strongest_prominences),
        "strongest_cusp_strengths": _encode_optional_real_array(result.strongest_cusp_strengths),
        "points": serialized_points,
        "base_model_metadata": result.base_model_metadata,
        "initial_state_descriptor": result.initial_state_descriptor,
        "reference_state_descriptor": result.reference_state_descriptor,
        "initial_state_model_metadata": result.initial_state_model_metadata,
        "reference_state_model_metadata": result.reference_state_model_metadata,
        "min_prominence": float(result.min_prominence),
        "min_cusp": float(result.min_cusp),
        "mode": result.mode,
        "completed_points": int(result.completed_points),
        "seed_values": None
        if result.seed_values is None
        else np.asarray(result.seed_values, dtype=np.float64).tolist(),
        "refinement_metric": result.refinement_metric,
        "refinement_strategy": result.refinement_strategy,
        "refinement_target_value": result.refinement_target_value,
        "insertion_policy": result.insertion_policy,
        "points_per_interval": int(result.points_per_interval),
        "max_refinement_rounds": int(result.max_refinement_rounds),
        "refinements_per_round": int(result.refinements_per_round),
        "min_spacing": result.min_spacing,
        "refinement_history": result.refinement_history,
    }
    _write_json(destination, payload)
    return destination


def load_dqpt_scan_result(path: str | Path) -> DQPTScanResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "dqpt_scan":
        raise ValueError("checkpoint does not contain a DQPT scan result")

    points = []
    for ordinal, point in enumerate(payload.get("points", [])):
        points.append(
            DQPTScanPoint(
                value=float(point["value"]),
                solver=str(point["solver"]),
                return_rate=np.asarray(point["return_rate"], dtype=np.float64),
                candidate_times=np.asarray(point.get("candidate_times", []), dtype=np.float64),
                candidate_return_rates=np.asarray(
                    point.get("candidate_return_rates", []),
                    dtype=np.float64,
                ),
                candidate_prominences=np.asarray(
                    point.get("candidate_prominences", []),
                    dtype=np.float64,
                ),
                candidate_cusp_strengths=np.asarray(
                    point.get("candidate_cusp_strengths", []),
                    dtype=np.float64,
                ),
                strongest_candidate_time=None
                if point.get("strongest_candidate_time") is None
                else float(point["strongest_candidate_time"]),
                strongest_candidate_return_rate=None
                if point.get("strongest_candidate_return_rate") is None
                else float(point["strongest_candidate_return_rate"]),
                strongest_prominence=None
                if point.get("strongest_prominence") is None
                else float(point["strongest_prominence"]),
                strongest_cusp_strength=None
                if point.get("strongest_cusp_strength") is None
                else float(point["strongest_cusp_strength"]),
                model_metadata=point.get("model_metadata"),
                backend_state=None
                if point.get("backend_state_checkpoint") is None
                else load_tensor_network_state(source.parent / str(point["backend_state_checkpoint"])),
                backend_state_checkpoint=point.get("backend_state_checkpoint"),
                index=ordinal if point.get("index") is None else int(point["index"]),
            )
        )

    return DQPTScanResult(
        parameter=str(payload["parameter"]),
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        values=np.asarray(payload["values"], dtype=np.float64),
        times=np.asarray(payload["times"], dtype=np.float64),
        return_rates=np.asarray(_decode_optional_real_ndarray(payload.get("return_rates")), dtype=np.float64),
        candidate_counts=np.asarray(_decode_optional_real_array(payload.get("candidate_counts")), dtype=np.float64),
        max_return_rates=np.asarray(_decode_optional_real_array(payload.get("max_return_rates")), dtype=np.float64),
        strongest_candidate_times=np.asarray(
            _decode_optional_real_array(payload.get("strongest_candidate_times")),
            dtype=np.float64,
        ),
        strongest_candidate_return_rates=np.asarray(
            _decode_optional_real_array(payload.get("strongest_candidate_return_rates")),
            dtype=np.float64,
        ),
        strongest_prominences=np.asarray(
            _decode_optional_real_array(payload.get("strongest_prominences")),
            dtype=np.float64,
        ),
        strongest_cusp_strengths=np.asarray(
            _decode_optional_real_array(payload.get("strongest_cusp_strengths")),
            dtype=np.float64,
        ),
        points=points,
        base_model_metadata=payload.get("base_model_metadata"),
        initial_state_descriptor=payload.get("initial_state_descriptor"),
        reference_state_descriptor=payload.get("reference_state_descriptor"),
        initial_state_model_metadata=payload.get("initial_state_model_metadata"),
        reference_state_model_metadata=payload.get("reference_state_model_metadata"),
        min_prominence=float(payload.get("min_prominence", 0.0)),
        min_cusp=float(payload.get("min_cusp", 0.0)),
        mode=str(payload.get("mode", "maxima")),
        completed_points=int(payload.get("completed_points", len(points))),
        seed_values=None
        if payload.get("seed_values") is None
        else np.asarray(payload.get("seed_values"), dtype=np.float64),
        refinement_metric=payload.get("refinement_metric"),
        refinement_strategy=payload.get("refinement_strategy"),
        refinement_target_value=payload.get("refinement_target_value"),
        insertion_policy=payload.get("insertion_policy"),
        points_per_interval=int(payload.get("points_per_interval", 1)),
        max_refinement_rounds=int(payload.get("max_refinement_rounds", 0)),
        refinements_per_round=int(payload.get("refinements_per_round", 0)),
        min_spacing=None
        if payload.get("min_spacing") is None
        else float(payload.get("min_spacing")),
        refinement_history=list(payload.get("refinement_history") or []),
    )


def save_dynamic_structure_factor_result(
    result: DynamicStructureFactorResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    payload = {
        "format_version": 1,
        "result_type": "dynamic_structure_factor",
        "model_name": result.model_name,
        "solver": result.solver,
        "pauli": result.pauli,
        "connected": bool(result.connected),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "momenta": np.asarray(result.momenta, dtype=np.float64).tolist(),
        "values": np.asarray(result.values, dtype=np.float64).tolist(),
        "model_metadata": result.model_metadata,
    }
    _write_json(destination, payload)
    return destination


def load_dynamic_structure_factor_result(path: str | Path) -> DynamicStructureFactorResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "dynamic_structure_factor":
        raise ValueError("checkpoint does not contain a dynamic-structure-factor result")

    return DynamicStructureFactorResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        pauli=str(payload["pauli"]),
        connected=bool(payload["connected"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        momenta=np.asarray(payload["momenta"], dtype=np.float64),
        values=np.asarray(payload["values"], dtype=np.float64),
        model_metadata=payload.get("model_metadata"),
    )


def save_frequency_structure_factor_result(
    result: FrequencyStructureFactorResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    payload = {
        "format_version": 1,
        "result_type": "frequency_structure_factor",
        "model_name": result.model_name,
        "solver": result.solver,
        "pauli": result.pauli,
        "connected": bool(result.connected),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "frequencies": np.asarray(result.frequencies, dtype=np.float64).tolist(),
        "momenta": np.asarray(result.momenta, dtype=np.float64).tolist(),
        "values": _encode_complex_array(result.values),
        "window": str(result.window),
        "subtract_mean": bool(result.subtract_mean),
        "model_metadata": result.model_metadata,
    }
    _write_json(destination, payload)
    return destination


def load_frequency_structure_factor_result(path: str | Path) -> FrequencyStructureFactorResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "frequency_structure_factor":
        raise ValueError("checkpoint does not contain a frequency-structure-factor result")

    return FrequencyStructureFactorResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        pauli=str(payload["pauli"]),
        connected=bool(payload["connected"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        frequencies=np.asarray(payload["frequencies"], dtype=np.float64),
        momenta=np.asarray(payload["momenta"], dtype=np.float64),
        values=np.asarray(_decode_complex_array(payload.get("values")), dtype=np.complex128),
        window=str(payload["window"]),
        subtract_mean=bool(payload["subtract_mean"]),
        model_metadata=payload.get("model_metadata"),
    )


def save_two_time_correlator_result(
    result: TwoTimeCorrelatorResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    payload = {
        "format_version": 1,
        "result_type": "two_time_correlator",
        "model_name": result.model_name,
        "solver": result.solver,
        "pauli": result.pauli,
        "connected": bool(result.connected),
        "num_sites": int(result.num_sites),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "values": _encode_complex_array(result.values),
        "dynamic_single_site_expectations": np.asarray(
            result.dynamic_single_site_expectations,
            dtype=np.float64,
        ).tolist(),
        "initial_single_site_expectations": np.asarray(
            result.initial_single_site_expectations,
            dtype=np.float64,
        ).tolist(),
        "measure_sites": list(result.measure_sites),
        "source_sites": list(result.source_sites),
        "model_metadata": result.model_metadata,
    }
    _write_json(destination, payload)
    return destination


def load_two_time_correlator_result(path: str | Path) -> TwoTimeCorrelatorResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "two_time_correlator":
        raise ValueError("checkpoint does not contain a two-time correlator result")

    return TwoTimeCorrelatorResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        pauli=str(payload["pauli"]),
        connected=bool(payload["connected"]),
        num_sites=int(payload["num_sites"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        values=np.asarray(_decode_complex_array(payload.get("values")), dtype=np.complex128),
        dynamic_single_site_expectations=np.asarray(
            payload["dynamic_single_site_expectations"],
            dtype=np.float64,
        ),
        initial_single_site_expectations=np.asarray(
            payload["initial_single_site_expectations"],
            dtype=np.float64,
        ),
        measure_sites=tuple(int(site) for site in payload.get("measure_sites", [])),
        source_sites=tuple(int(site) for site in payload.get("source_sites", [])),
        model_metadata=payload.get("model_metadata"),
    )


def save_response_spectrum_result(
    result: ResponseSpectrumResult,
    path: str | Path,
) -> Path:
    destination = Path(path)
    payload = {
        "format_version": 1,
        "result_type": "response_spectrum",
        "model_name": result.model_name,
        "solver": result.solver,
        "pauli": result.pauli,
        "num_sites": int(result.num_sites),
        "times": np.asarray(result.times, dtype=np.float64).tolist(),
        "frequencies": np.asarray(result.frequencies, dtype=np.float64).tolist(),
        "momenta": np.asarray(result.momenta, dtype=np.float64).tolist(),
        "time_response": _encode_complex_array(result.time_response),
        "values": _encode_complex_array(result.values),
        "measure_sites": list(result.measure_sites),
        "source_sites": list(result.source_sites),
        "window": str(result.window),
        "subtract_mean": bool(result.subtract_mean),
        "model_metadata": result.model_metadata,
    }
    _write_json(destination, payload)
    return destination


def load_response_spectrum_result(path: str | Path) -> ResponseSpectrumResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "response_spectrum":
        raise ValueError("checkpoint does not contain a response-spectrum result")

    return ResponseSpectrumResult(
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        pauli=str(payload["pauli"]),
        num_sites=int(payload["num_sites"]),
        times=np.asarray(payload["times"], dtype=np.float64),
        frequencies=np.asarray(payload["frequencies"], dtype=np.float64),
        momenta=np.asarray(payload["momenta"], dtype=np.float64),
        time_response=np.asarray(_decode_complex_array(payload.get("time_response")), dtype=np.complex128),
        values=np.asarray(_decode_complex_array(payload.get("values")), dtype=np.complex128),
        measure_sites=tuple(int(site) for site in payload.get("measure_sites", [])),
        source_sites=tuple(int(site) for site in payload.get("source_sites", [])),
        window=str(payload["window"]),
        subtract_mean=bool(payload["subtract_mean"]),
        model_metadata=payload.get("model_metadata"),
    )


def save_sweep_result(
    result: SweepResult,
    path: str | Path,
    *,
    include_backend_state: bool = True,
) -> Path:
    destination = Path(path)
    serialized_points = []
    for index, point in enumerate(result.points):
        backend_state_checkpoint = point.backend_state_checkpoint if include_backend_state else None
        if include_backend_state and point.backend_state is not None:
            if backend_state_checkpoint is None:
                state_path = _sweep_point_checkpoint_path(destination, index)
                backend_state_checkpoint = str(state_path.relative_to(destination.parent))
            state_path = destination.parent / backend_state_checkpoint
            if not state_path.exists():
                save_tensor_network_state(point.backend_state, state_path)
            point.backend_state_checkpoint = backend_state_checkpoint

        serialized_points.append(
            {
                "index": point.index,
                "value": float(point.value),
                "energy": float(point.energy),
                "observables": {
                    label: float(value) for label, value in point.observables.items()
                },
                "solver": point.solver,
                "spectral_gap": None
                if point.spectral_gap is None
                else float(point.spectral_gap),
                "entanglement_entropy": None
                if point.entanglement_entropy is None
                else float(point.entanglement_entropy),
                "model_metadata": point.model_metadata,
                "backend_state_checkpoint": backend_state_checkpoint,
            }
        )

    payload = {
        "format_version": 1,
        "result_type": "parameter_sweep",
        "parameter": result.parameter,
        "model_name": result.model_name,
        "solver": result.solver,
        "completed_points": int(result.completed_points),
        "values": np.asarray(result.values, dtype=np.float64).tolist(),
        "energies": _encode_optional_real_array(result.energies),
        "observables": {
            label: _encode_optional_real_array(values)
            for label, values in result.observables.items()
        },
        "base_model_metadata": result.base_model_metadata,
        "observables_requested": list(result.observables_requested),
        "subsystem": None if result.subsystem is None else list(result.subsystem),
        "spectral_gaps": _encode_optional_real_array(result.spectral_gaps),
        "entanglement_entropy": _encode_optional_real_array(result.entanglement_entropy),
        "seed_values": _encode_real_array(result.seed_values),
        "refinement_metric": result.refinement_metric,
        "refinement_strategy": result.refinement_strategy,
        "refinement_target_value": result.refinement_target_value,
        "insertion_policy": result.insertion_policy,
        "points_per_interval": int(result.points_per_interval),
        "max_refinement_rounds": int(result.max_refinement_rounds),
        "refinements_per_round": int(result.refinements_per_round),
        "min_spacing": result.min_spacing,
        "refinement_history": result.refinement_history,
        "points": serialized_points,
    }
    _write_json(destination, payload)
    return destination


def load_sweep_result(path: str | Path) -> SweepResult:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_type") != "parameter_sweep":
        raise ValueError("checkpoint does not contain a parameter-sweep result")

    subsystem = payload.get("subsystem")
    points = []
    for ordinal, point in enumerate(payload.get("points", [])):
        points.append(
            SweepPoint(
                value=float(point["value"]),
                energy=float(point["energy"]),
                observables={
                    label: float(value)
                    for label, value in dict(point["observables"]).items()
                },
                solver=str(point["solver"]),
                spectral_gap=None
                if point.get("spectral_gap") is None
                else float(point["spectral_gap"]),
                entanglement_entropy=None
                if point.get("entanglement_entropy") is None
                else float(point["entanglement_entropy"]),
                model_metadata=point.get("model_metadata"),
                backend_state=None
                if point.get("backend_state_checkpoint") is None
                else load_tensor_network_state(source.parent / str(point["backend_state_checkpoint"])),
                backend_state_checkpoint=point.get("backend_state_checkpoint"),
                index=ordinal if point.get("index") is None else int(point["index"]),
            )
        )
    return SweepResult(
        parameter=str(payload["parameter"]),
        model_name=str(payload["model_name"]),
        solver=str(payload["solver"]),
        values=np.asarray(payload["values"], dtype=np.float64),
        energies=_decode_optional_real_array(payload.get("energies")),
        observables={
            label: _decode_optional_real_array(values)
            for label, values in dict(payload["observables"]).items()
        },
        points=points,
        base_model_metadata=payload.get("base_model_metadata"),
        observables_requested=tuple(str(label) for label in payload.get("observables_requested", [])),
        subsystem=None if subsystem is None else tuple(int(site) for site in subsystem),
        spectral_gaps=_decode_optional_real_array(payload.get("spectral_gaps")),
        entanglement_entropy=_decode_optional_real_array(payload.get("entanglement_entropy")),
        completed_points=int(payload.get("completed_points", len(points))),
        seed_values=_decode_real_array(payload.get("seed_values")),
        refinement_metric=payload.get("refinement_metric"),
        refinement_strategy=payload.get("refinement_strategy"),
        refinement_target_value=None
        if payload.get("refinement_target_value") is None
        else float(payload.get("refinement_target_value")),
        insertion_policy=payload.get("insertion_policy"),
        points_per_interval=int(payload.get("points_per_interval", 1)),
        max_refinement_rounds=int(payload.get("max_refinement_rounds", 0)),
        refinements_per_round=int(payload.get("refinements_per_round", 0)),
        min_spacing=None
        if payload.get("min_spacing") is None
        else float(payload.get("min_spacing")),
        refinement_history=list(payload.get("refinement_history") or []),
    )
