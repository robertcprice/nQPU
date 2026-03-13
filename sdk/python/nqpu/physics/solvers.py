"""Research solvers and analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import re
from typing import Iterable, Protocol

import numpy as np

from nqpu._compat import get_rust_bindings

from .models import (
    CustomHamiltonian,
    HeisenbergXXZ1D,
    HeisenbergXYZ1D,
    TransverseFieldIsing1D,
    pauli_string_operator,
)


PAULI_TOKEN = re.compile(r"([IXYZ])(\d+)")
PRODUCT_STATE_CHARS = frozenset({"0", "1", "+", "-", "R", "L"})


class TensorNetworkStateHandle(Protocol):
    num_sites: int
    solver: str


@dataclass
class GroundStateResult:
    model_name: str
    solver: str
    dimension: int
    ground_state_energy: float
    spectral_gap: float | None
    eigenvalues: np.ndarray
    ground_state: np.ndarray | None
    observables: dict[str, float]
    entanglement_entropy: float | None = None
    backend_state: object | None = None
    model_metadata: dict[str, object] | None = None


@dataclass
class TimeEvolutionResult:
    model_name: str
    solver: str
    times: np.ndarray
    observables: dict[str, np.ndarray]
    final_state: np.ndarray | None = None
    backend_state: object | None = None
    entanglement_entropy: np.ndarray | None = None
    entanglement_subsystem: tuple[int, ...] | None = None
    model_metadata: dict[str, object] | None = None


InitialStateLike = GroundStateResult | TimeEvolutionResult | TensorNetworkStateHandle | str | np.ndarray


class ExactDiagonalizationSolver:
    """Dense exact diagonalization for small spin models."""

    def __init__(self, max_dimension: int = 4096):
        self.max_dimension = max_dimension

    def solve_ground_state(
        self,
        model: AnyModel,
        observables: Iterable[str] = (),
        *,
        initial_state: InitialStateLike | None = None,
        subsystem: Iterable[int] | None = None,
        num_eigenvalues: int = 8,
    ) -> GroundStateResult:
        del initial_state
        hamiltonian = model.hamiltonian()
        dimension = hamiltonian.shape[0]
        self._validate_dimension(dimension)

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        ground_state = eigenvectors[:, 0]
        ground_energy = float(np.real(eigenvalues[0]))
        gap = float(np.real(eigenvalues[1] - eigenvalues[0])) if len(eigenvalues) > 1 else None

        resolved_observables = {
            label: _expectation_value(
                ground_state,
                _resolve_observable(model, label),
            )
            for label in observables
        }

        subsystem_sites = _normalize_subsystem(model.num_sites, subsystem)
        if not subsystem_sites:
            subsystem_sites = None
        entanglement = None
        if subsystem_sites is not None:
            entanglement = entanglement_entropy(ground_state, model.num_sites, list(subsystem_sites))

        return GroundStateResult(
            model_name=getattr(model, "model_name", type(model).__name__),
            solver="exact_diagonalization",
            dimension=dimension,
            ground_state_energy=ground_energy,
            spectral_gap=gap,
            eigenvalues=np.asarray(eigenvalues[:num_eigenvalues], dtype=np.float64),
            ground_state=ground_state,
            observables=resolved_observables,
            entanglement_entropy=entanglement,
            model_metadata=_model_metadata(model),
        )

    def spectrum(self, model: AnyModel, num_eigenvalues: int | None = None) -> np.ndarray:
        hamiltonian = model.hamiltonian()
        self._validate_dimension(hamiltonian.shape[0])
        eigenvalues = np.linalg.eigvalsh(hamiltonian)
        if num_eigenvalues is None:
            return np.asarray(eigenvalues, dtype=np.float64)
        return np.asarray(eigenvalues[:num_eigenvalues], dtype=np.float64)

    def time_evolve(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike = "all_up",
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
    ) -> TimeEvolutionResult:
        hamiltonian = model.hamiltonian()
        dimension = hamiltonian.shape[0]
        self._validate_dimension(dimension)

        times = np.asarray(list(times), dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        psi0 = make_initial_state(model.num_sites, initial_state)
        coefficients = eigenvectors.conj().T @ psi0
        subsystem_sites = _normalize_subsystem(model.num_sites, subsystem)
        if not subsystem_sites:
            subsystem_sites = None

        traces = {label: np.zeros_like(times, dtype=np.float64) for label in observables}
        observable_operators = {
            label: _resolve_observable(model, label)
            for label in observables
        }
        entanglement_trace = None
        if subsystem_sites is not None:
            entanglement_trace = np.zeros_like(times, dtype=np.float64)
        final_state = psi0

        for index, time in enumerate(times):
            phases = np.exp(-1.0j * eigenvalues * time)
            state = eigenvectors @ (phases * coefficients)
            final_state = state
            for label, operator in observable_operators.items():
                traces[label][index] = _expectation_value(state, operator)
            if entanglement_trace is not None:
                entanglement_trace[index] = entanglement_entropy(
                    state,
                    model.num_sites,
                    list(subsystem_sites),
                )

        return TimeEvolutionResult(
            model_name=getattr(model, "model_name", type(model).__name__),
            solver="exact_diagonalization",
            times=times,
            observables=traces,
            final_state=final_state,
            entanglement_entropy=entanglement_trace,
            entanglement_subsystem=subsystem_sites,
            model_metadata=_model_metadata(model),
        )

    def _validate_dimension(self, dimension: int) -> None:
        if dimension > self.max_dimension:
            raise ValueError(
                f"exact diagonalization limited to dimension {self.max_dimension}, got {dimension}"
            )


class RustTensorNetworkSolver:
    """Rust-backed DMRG/TDVP bridge for supported 1D spin models."""

    def __init__(
        self,
        *,
        max_bond_dim: int = 64,
        max_sweeps: int = 20,
        energy_tolerance: float = 1e-8,
        lanczos_iterations: int = 20,
        tdvp_method: str = "two_site",
        state_compression_cutoff: float = 1e-10,
        bindings: object | None = None,
    ):
        self.max_bond_dim = max_bond_dim
        self.max_sweeps = max_sweeps
        self.energy_tolerance = energy_tolerance
        self.lanczos_iterations = lanczos_iterations
        self.tdvp_method = tdvp_method
        self.state_compression_cutoff = state_compression_cutoff
        self._bindings_override = bindings

    @property
    def bindings(self) -> object | None:
        if self._bindings_override is not None:
            return self._bindings_override
        return get_rust_bindings()

    def is_available(self) -> bool:
        bindings = self.bindings
        return bindings is not None and hasattr(bindings, "dmrg_ground_state_1d")

    def has_tdvp_bindings(self) -> bool:
        bindings = self.bindings
        return bindings is not None and hasattr(bindings, "tdvp_time_evolution_1d")

    def has_statevector_conversion_bindings(self) -> bool:
        bindings = self.bindings
        return bindings is not None and hasattr(bindings, "statevector_to_mps_1d")

    def has_transition_bindings(self) -> bool:
        bindings = self.bindings
        return (
            bindings is not None
            and hasattr(bindings, "apply_local_pauli_1d")
            and hasattr(bindings, "tdvp_transition_observables_1d")
        )

    def has_entanglement_spectrum_bindings(self) -> bool:
        bindings = self.bindings
        return bindings is not None and hasattr(bindings, "entanglement_spectrum_1d")

    def has_loschmidt_bindings(self) -> bool:
        bindings = self.bindings
        return bindings is not None and hasattr(bindings, "tdvp_loschmidt_echo_1d")

    def supports_ground_state(
        self,
        model: AnyModel,
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
        initial_state: InitialStateLike | None = None,
    ) -> bool:
        if not self.is_available():
            return False
        if not _supports_dmrg_model(model):
            return False
        subsystem_sites = _ordered_subsystem(model.num_sites, subsystem)
        if subsystem_sites and _entropy_bond_for_subsystem(model.num_sites, subsystem) is None:
            return False
        if not _supports_dmrg_initial_state(model.num_sites, initial_state):
            return False
        return all(_supports_dmrg_observable(label) for label in observables)

    def solve_ground_state(
        self,
        model: AnyModel,
        observables: Iterable[str] = (),
        *,
        initial_state: InitialStateLike | None = None,
        subsystem: Iterable[int] | None = None,
        num_eigenvalues: int = 8,
    ) -> GroundStateResult:
        del num_eigenvalues
        if not self.supports_ground_state(
            model,
            observables=observables,
            subsystem=subsystem,
            initial_state=initial_state,
        ):
            raise ValueError("model/observable combination is not supported by the Rust DMRG path")

        bindings = self.bindings
        assert bindings is not None

        entropy_bond = _entropy_bond_for_subsystem(model.num_sites, subsystem)
        payload = {
            "num_sites": model.num_sites,
            "max_bond_dim": self.max_bond_dim,
            "max_sweeps": self.max_sweeps,
            "energy_tolerance": self.energy_tolerance,
            "lanczos_iterations": self.lanczos_iterations,
            "observables": list(observables),
            "entropy_bond": entropy_bond,
        }
        backend_state = _extract_backend_state(initial_state)
        if backend_state is not None:
            payload["state_handle"] = backend_state

        if isinstance(model, TransverseFieldIsing1D):
            payload.update(
                {
                    "model": "transverse_field_ising_1d",
                    "coupling": model.coupling,
                    "transverse_field": model.transverse_field,
                    "longitudinal_field": model.longitudinal_field,
                }
            )
        elif isinstance(model, HeisenbergXXZ1D):
            payload.update(
                {
                    "model": "heisenberg_xxz_1d",
                    "coupling": model.coupling_xy,
                    "anisotropy": model.anisotropy,
                    "field_z": model.field_z,
                }
            )
        elif isinstance(model, HeisenbergXYZ1D):
            payload.update(
                {
                    "model": "heisenberg_xyz_1d",
                    "coupling_x": model.coupling_x,
                    "coupling_y": model.coupling_y,
                    "coupling_z": model.coupling_z,
                    "field_z": model.field_z,
                }
            )
        else:
            raise TypeError(f"unsupported DMRG model type: {type(model).__name__}")

        raw = bindings.dmrg_ground_state_1d(**payload)
        raw_observables = raw.get("observables", {})
        entanglement = raw.get("entanglement_entropy")
        spectral_gap = raw.get("spectral_gap")
        energy = float(raw["ground_state_energy"])
        eigenvalues = np.asarray([energy], dtype=np.float64)

        return GroundStateResult(
            model_name=str(raw.get("model_name", getattr(model, "model_name", type(model).__name__))),
            solver=str(raw.get("solver", "dmrg")),
            dimension=_model_dimension(model),
            ground_state_energy=energy,
            spectral_gap=None if spectral_gap is None else float(spectral_gap),
            eigenvalues=eigenvalues,
            ground_state=None,
            observables={label: float(value) for label, value in dict(raw_observables).items()},
            entanglement_entropy=None if entanglement is None else float(entanglement),
            backend_state=raw.get("state_handle"),
            model_metadata=_model_metadata(model),
        )

    def supports_time_evolution(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike = "all_up",
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
    ) -> bool:
        if not self.has_tdvp_bindings():
            return False
        if not _supports_dmrg_model(model):
            return False
        subsystem_sites = _ordered_subsystem(model.num_sites, subsystem)
        if subsystem_sites and _entropy_bond_for_subsystem(model.num_sites, subsystem) is None:
            return False
        if not self._supports_tdvp_state_input(model.num_sites, initial_state):
            return False
        if not _supports_tdvp_times(times):
            return False
        return all(_supports_dmrg_observable(label) for label in observables)

    def time_evolve(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike = "all_up",
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
    ) -> TimeEvolutionResult:
        times_array = np.asarray(list(times), dtype=np.float64)
        if not self.supports_time_evolution(
            model,
            times_array,
            initial_state=initial_state,
            observables=observables,
            subsystem=subsystem,
        ):
            raise ValueError("model/time grid/state combination is not supported by the Rust TDVP path")

        bindings = self.bindings
        assert bindings is not None
        subsystem_sites = _normalize_subsystem(model.num_sites, subsystem)
        if not subsystem_sites:
            subsystem_sites = None
        entropy_bond = _entropy_bond_for_subsystem(model.num_sites, subsystem)

        payload = {
            "num_sites": model.num_sites,
            "times": times_array.tolist(),
            "method": self.tdvp_method,
            "max_bond_dim": self.max_bond_dim,
            "lanczos_iterations": self.lanczos_iterations,
            "observables": list(observables),
        }
        if entropy_bond is not None:
            payload["entropy_bond"] = entropy_bond
        payload.update(
            self._tdvp_state_payload(
                model.num_sites,
                initial_state,
                model_name=getattr(model, "model_name", type(model).__name__),
                payload_key="initial_state",
            )
        )

        if isinstance(model, TransverseFieldIsing1D):
            payload.update(
                {
                    "model": "transverse_field_ising_1d",
                    "coupling": model.coupling,
                    "transverse_field": model.transverse_field,
                    "longitudinal_field": model.longitudinal_field,
                }
            )
        elif isinstance(model, HeisenbergXXZ1D):
            payload.update(
                {
                    "model": "heisenberg_xxz_1d",
                    "coupling": model.coupling_xy,
                    "anisotropy": model.anisotropy,
                    "field_z": model.field_z,
                }
            )
        elif isinstance(model, HeisenbergXYZ1D):
            payload.update(
                {
                    "model": "heisenberg_xyz_1d",
                    "coupling_x": model.coupling_x,
                    "coupling_y": model.coupling_y,
                    "coupling_z": model.coupling_z,
                    "field_z": model.field_z,
                }
            )
        else:
            raise TypeError(f"unsupported TDVP model type: {type(model).__name__}")

        raw = bindings.tdvp_time_evolution_1d(**payload)
        raw_observables = dict(raw.get("observables", {}))
        traces = {
            label: np.asarray(values, dtype=np.float64)
            for label, values in raw_observables.items()
        }
        raw_entropy = raw.get("entanglement_entropy")
        entanglement_trace = None if raw_entropy is None else np.asarray(raw_entropy, dtype=np.float64)

        return TimeEvolutionResult(
            model_name=str(raw.get("model_name", getattr(model, "model_name", type(model).__name__))),
            solver=str(raw.get("solver", "tdvp")),
            times=np.asarray(raw.get("times", times_array), dtype=np.float64),
            observables=traces,
            backend_state=raw.get("state_handle"),
            entanglement_entropy=entanglement_trace,
            entanglement_subsystem=subsystem_sites,
            model_metadata=_model_metadata(model),
        )

    def supports_two_time_ground_state_correlator(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        pauli: str,
        reference_state: InitialStateLike | None,
    ) -> bool:
        if not self.has_transition_bindings():
            return False
        if pauli not in {"X", "Y", "Z"}:
            return False
        backend_state = _extract_backend_state(reference_state)
        if backend_state is None:
            return False
        if not _supports_dmrg_model(model):
            return False
        if not _supports_tdvp_times(times):
            return False
        return _supports_tdvp_initial_state(model.num_sites, backend_state)

    def supports_loschmidt_echo(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike,
        reference_state: InitialStateLike | None = None,
    ) -> bool:
        if not self.has_loschmidt_bindings():
            return False
        if not _supports_dmrg_model(model):
            return False
        if not _supports_tdvp_times(times):
            return False
        if not self._supports_tdvp_state_input(model.num_sites, initial_state):
            return False
        if reference_state is not None and not self._supports_tdvp_state_input(
            model.num_sites,
            reference_state,
        ):
            return False
        return True

    def loschmidt_echo(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike,
        reference_state: InitialStateLike | None = None,
    ) -> dict[str, object]:
        times_array = np.asarray(list(times), dtype=np.float64)
        if not self.supports_loschmidt_echo(
            model,
            times_array,
            initial_state=initial_state,
            reference_state=reference_state,
        ):
            raise ValueError("model/time grid/state combination is not supported by the Rust Loschmidt path")

        bindings = self.bindings
        assert bindings is not None
        payload = {
            "num_sites": model.num_sites,
            "times": times_array.tolist(),
            "method": self.tdvp_method,
            "max_bond_dim": self.max_bond_dim,
            "lanczos_iterations": self.lanczos_iterations,
        }

        payload.update(
            self._tdvp_state_payload(
                model.num_sites,
                initial_state,
                model_name=getattr(model, "model_name", type(model).__name__),
                payload_key="initial_state",
            )
        )
        if reference_state is not None:
            payload.update(
                self._tdvp_state_payload(
                    model.num_sites,
                    reference_state,
                    model_name=getattr(model, "model_name", type(model).__name__),
                    payload_key="reference_initial_state",
                    handle_key="reference_state_handle",
                )
            )

        if isinstance(model, TransverseFieldIsing1D):
            payload.update(
                {
                    "model": "transverse_field_ising_1d",
                    "coupling": model.coupling,
                    "transverse_field": model.transverse_field,
                    "longitudinal_field": model.longitudinal_field,
                }
            )
        elif isinstance(model, HeisenbergXXZ1D):
            payload.update(
                {
                    "model": "heisenberg_xxz_1d",
                    "coupling": model.coupling_xy,
                    "anisotropy": model.anisotropy,
                    "field_z": model.field_z,
                }
            )
        elif isinstance(model, HeisenbergXYZ1D):
            payload.update(
                {
                    "model": "heisenberg_xyz_1d",
                    "coupling_x": model.coupling_x,
                    "coupling_y": model.coupling_y,
                    "coupling_z": model.coupling_z,
                    "field_z": model.field_z,
                }
            )
        else:
            raise TypeError(f"unsupported TDVP Loschmidt model type: {type(model).__name__}")

        raw = bindings.tdvp_loschmidt_echo_1d(**payload)
        raw_amplitudes = raw.get("amplitudes", [])
        amplitudes = np.asarray(
            [complex(real, imag) for real, imag in raw_amplitudes],
            dtype=np.complex128,
        )
        return {
            "model_name": str(raw.get("model_name", getattr(model, "model_name", type(model).__name__))),
            "solver": str(raw.get("solver", "tdvp_overlap")),
            "times": np.asarray(raw.get("times", times_array), dtype=np.float64),
            "amplitudes": amplitudes,
            "state_handle": raw.get("state_handle"),
        }

    def _supports_tdvp_state_input(
        self,
        num_sites: int,
        state: InitialStateLike,
    ) -> bool:
        backend_state = _extract_backend_state(state)
        if backend_state is not None:
            state_sites = getattr(backend_state, "num_sites", None)
            return state_sites is None or int(state_sites) == num_sites
        if isinstance(state, str):
            return _canonical_product_state_label(num_sites, state) is not None
        dense_state = _coerce_dense_statevector(num_sites, state)
        return dense_state is not None and self.has_statevector_conversion_bindings()

    def _tdvp_state_payload(
        self,
        num_sites: int,
        state: InitialStateLike,
        *,
        model_name: str,
        payload_key: str,
        handle_key: str = "state_handle",
    ) -> dict[str, object]:
        backend_state = _extract_backend_state(state)
        if backend_state is not None:
            return {handle_key: backend_state}
        if isinstance(state, str):
            return {payload_key: _resolve_tdvp_initial_state_label(num_sites, state)}

        dense_state = _coerce_dense_statevector(num_sites, state)
        if dense_state is None:
            raise ValueError(
                "Rust TDVP currently supports string-labeled product states, "
                "dense statevectors/results via state compression, or backend state handles"
            )
        return {
            handle_key: self.statevector_to_backend_state(
                dense_state,
                num_sites=num_sites,
                model_name=model_name,
            )
        }

    def statevector_to_backend_state(
        self,
        state: np.ndarray,
        *,
        num_sites: int,
        model_name: str = "compressed_state",
    ) -> object:
        if not self.has_statevector_conversion_bindings():
            raise ValueError("Rust dense-state conversion bindings are not available")
        dense_state = _coerce_dense_statevector(num_sites, np.asarray(state, dtype=np.complex128))
        if dense_state is None:
            raise ValueError("dense statevector does not match the requested num_sites")
        bindings = self.bindings
        assert bindings is not None
        values = np.asarray(dense_state, dtype=np.complex128)
        norm = np.linalg.norm(values)
        if norm <= 1e-30:
            raise ValueError("dense statevector must have non-zero norm")
        normalized = values / norm
        return bindings.statevector_to_mps_1d(
            num_sites=num_sites,
            statevector=[(float(value.real), float(value.imag)) for value in normalized],
            max_bond_dim=self.max_bond_dim,
            cutoff=self.state_compression_cutoff,
            model_name=model_name,
        )

    def supports_entanglement_spectrum(
        self,
        state: InitialStateLike | object,
        *,
        subsystem: Iterable[int],
    ) -> bool:
        if not self.has_entanglement_spectrum_bindings():
            return False
        backend_state = _extract_backend_state(state)
        if backend_state is None:
            if not _is_backend_state_handle(state):
                return False
            backend_state = state
        state_sites = getattr(backend_state, "num_sites", None)
        if state_sites is None:
            return False
        return _entropy_bond_for_subsystem(int(state_sites), subsystem) is not None

    def entanglement_spectrum(
        self,
        state: InitialStateLike | object,
        *,
        subsystem: Iterable[int],
    ) -> np.ndarray:
        if not self.has_entanglement_spectrum_bindings():
            raise ValueError("Rust entanglement-spectrum bindings are not available")
        backend_state = _extract_backend_state(state)
        if backend_state is None:
            if not _is_backend_state_handle(state):
                raise ValueError("entanglement-spectrum analysis requires a tensor-network state handle")
            backend_state = state

        num_sites = getattr(backend_state, "num_sites", None)
        if num_sites is None:
            raise ValueError("tensor-network state handle does not expose num_sites")
        bond = _entropy_bond_for_subsystem(int(num_sites), subsystem)
        if bond is None:
            raise ValueError(
                "Rust entanglement spectra currently support only prefix subsystems "
                "that map to a single MPS bond"
            )

        bindings = self.bindings
        assert bindings is not None
        raw = bindings.entanglement_spectrum_1d(
            state_handle=backend_state,
            bond=bond,
        )
        if isinstance(raw, dict):
            values = raw.get("spectrum", [])
        else:
            values = raw
        return np.asarray(values, dtype=np.float64)

    def apply_local_pauli(
        self,
        state_handle: object,
        *,
        pauli: str,
        site: int,
    ) -> object:
        if not self.has_transition_bindings():
            raise ValueError("Rust transition-observable bindings are not available")
        bindings = self.bindings
        assert bindings is not None
        return bindings.apply_local_pauli_1d(
            state_handle=state_handle,
            pauli=pauli,
            site=site,
        )

    def transition_observable_evolution(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        reference_state: object,
        source_state: object,
        observables: Iterable[str] = (),
    ) -> dict[str, object]:
        if not self.has_transition_bindings():
            raise ValueError("Rust transition-observable bindings are not available")
        if not _supports_dmrg_model(model):
            raise ValueError("model is not supported by the Rust transition-observable path")
        if not _supports_tdvp_times(times):
            raise ValueError("time grid is not supported by the Rust transition-observable path")

        bindings = self.bindings
        assert bindings is not None
        times_array = np.asarray(list(times), dtype=np.float64)
        payload = {
            "num_sites": model.num_sites,
            "times": times_array.tolist(),
            "method": self.tdvp_method,
            "max_bond_dim": self.max_bond_dim,
            "lanczos_iterations": self.lanczos_iterations,
            "reference_state_handle": reference_state,
            "source_state_handle": source_state,
            "observables": list(observables),
        }

        if isinstance(model, TransverseFieldIsing1D):
            payload.update(
                {
                    "model": "transverse_field_ising_1d",
                    "coupling": model.coupling,
                    "transverse_field": model.transverse_field,
                    "longitudinal_field": model.longitudinal_field,
                }
            )
        elif isinstance(model, HeisenbergXXZ1D):
            payload.update(
                {
                    "model": "heisenberg_xxz_1d",
                    "coupling": model.coupling_xy,
                    "anisotropy": model.anisotropy,
                    "field_z": model.field_z,
                }
            )
        elif isinstance(model, HeisenbergXYZ1D):
            payload.update(
                {
                    "model": "heisenberg_xyz_1d",
                    "coupling_x": model.coupling_x,
                    "coupling_y": model.coupling_y,
                    "coupling_z": model.coupling_z,
                    "field_z": model.field_z,
                }
            )
        else:
            raise TypeError(f"unsupported transition-observable model type: {type(model).__name__}")

        raw = bindings.tdvp_transition_observables_1d(**payload)
        raw_observables = dict(raw.get("observables", {}))
        traces = {
            label: np.asarray(
                [complex(real, imag) for real, imag in values],
                dtype=np.complex128,
            )
            for label, values in raw_observables.items()
        }
        return {
            "model_name": str(raw.get("model_name", getattr(model, "model_name", type(model).__name__))),
            "solver": str(raw.get("solver", "tdvp_transition")),
            "times": np.asarray(raw.get("times", times_array), dtype=np.float64),
            "observables": traces,
            "state_handle": raw.get("state_handle"),
        }


class AutoSolver:
    """Prefer exact diagonalization when feasible, otherwise use supported Rust tensor-network solvers."""

    def __init__(
        self,
        *,
        exact_solver: ExactDiagonalizationSolver | None = None,
        tensor_network_solver: RustTensorNetworkSolver | None = None,
    ):
        self.exact_solver = exact_solver or ExactDiagonalizationSolver()
        self.tensor_network_solver = tensor_network_solver or RustTensorNetworkSolver()
        self.max_dimension = self.exact_solver.max_dimension

    def solve_ground_state(
        self,
        model: AnyModel,
        observables: Iterable[str] = (),
        *,
        initial_state: InitialStateLike | None = None,
        subsystem: Iterable[int] | None = None,
        num_eigenvalues: int = 8,
    ) -> GroundStateResult:
        dimension = _model_dimension(model)
        backend_state = _extract_backend_state(initial_state)
        tensor_supports = self.tensor_network_solver.supports_ground_state(
            model,
            observables=observables,
            subsystem=subsystem,
            initial_state=initial_state,
        )
        if backend_state is not None and dimension > self.exact_solver.max_dimension and not tensor_supports:
            raise ValueError(
                "initial state is stored only as a Rust tensor-network handle, "
                "but the requested ground-state optimization is not supported by the Rust DMRG path"
            )
        if (
            dimension > self.exact_solver.max_dimension
            and tensor_supports
        ):
            return self.tensor_network_solver.solve_ground_state(
                model,
                observables=observables,
                initial_state=initial_state,
                subsystem=subsystem,
                num_eigenvalues=num_eigenvalues,
            )
        return self.exact_solver.solve_ground_state(
            model,
            observables=observables,
            initial_state=initial_state,
            subsystem=subsystem,
            num_eigenvalues=num_eigenvalues,
        )

    def spectrum(self, model: AnyModel, num_eigenvalues: int | None = None) -> np.ndarray:
        return self.exact_solver.spectrum(model, num_eigenvalues=num_eigenvalues)

    def time_evolve(
        self,
        model: AnyModel,
        times: Iterable[float],
        *,
        initial_state: InitialStateLike = "all_up",
        observables: Iterable[str] = (),
        subsystem: Iterable[int] | None = None,
    ) -> TimeEvolutionResult:
        times_array = np.asarray(list(times), dtype=np.float64)
        dimension = _model_dimension(model)
        backend_state = _extract_backend_state(initial_state)
        if backend_state is not None:
            if self.tensor_network_solver.supports_time_evolution(
                model,
                times_array,
                initial_state=initial_state,
                observables=observables,
                subsystem=subsystem,
            ):
                return self.tensor_network_solver.time_evolve(
                    model,
                    times_array,
                    initial_state=initial_state,
                    observables=observables,
                    subsystem=subsystem,
                )
            if _extract_dense_state(initial_state) is None:
                raise ValueError(
                    "initial state is stored only as a Rust tensor-network handle, "
                    "but the requested quench or entropy cut is not supported by the Rust TDVP path"
                )
        if (
            dimension > self.exact_solver.max_dimension
            and self.tensor_network_solver.supports_time_evolution(
                model,
                times_array,
                initial_state=initial_state,
                observables=observables,
                subsystem=subsystem,
            )
        ):
            return self.tensor_network_solver.time_evolve(
                model,
                times_array,
                initial_state=initial_state,
                observables=observables,
                subsystem=subsystem,
            )
        return self.exact_solver.time_evolve(
            model,
            times_array,
            initial_state=initial_state,
            observables=observables,
            subsystem=subsystem,
        )


AnyModel = CustomHamiltonian | object


def _model_dimension(model: AnyModel) -> int:
    dimension = getattr(model, "dimension", None)
    if dimension is not None:
        return int(dimension)
    return 1 << int(model.num_sites)


def _model_metadata(model: AnyModel) -> dict[str, object]:
    if isinstance(model, CustomHamiltonian):
        return {
            "model_name": model.model_name,
            "kind": "custom_hamiltonian",
            "dimension": model.dimension,
            "num_sites": model.num_sites,
        }
    if is_dataclass(model):
        metadata = asdict(model)
        metadata["model_name"] = getattr(model, "model_name", type(model).__name__)
        return metadata
    return {"model_name": getattr(model, "model_name", type(model).__name__)}


def _supports_dmrg_model(model: AnyModel) -> bool:
    if isinstance(model, TransverseFieldIsing1D):
        return model.num_sites >= 2 and model.boundary == "open"
    if isinstance(model, HeisenbergXXZ1D):
        return model.num_sites >= 2 and model.boundary == "open"
    if isinstance(model, HeisenbergXYZ1D):
        return model.num_sites >= 2 and model.boundary == "open"
    return False


def _supports_dmrg_observable(label: str) -> bool:
    if label in {"energy", "magnetization_z", "staggered_magnetization_z"}:
        return True
    tokens = PAULI_TOKEN.findall(label)
    if not tokens or "".join(f"{pauli}{site}" for pauli, site in tokens) != label:
        return False
    return len(tokens) <= 2


def _supports_dmrg_initial_state(
    num_sites: int,
    initial_state: InitialStateLike | None,
) -> bool:
    if initial_state is None:
        return True
    backend_state = _extract_backend_state(initial_state)
    if backend_state is None:
        return False
    state_sites = getattr(backend_state, "num_sites", None)
    return state_sites is None or int(state_sites) == num_sites


def _entropy_bond_for_subsystem(num_sites: int, subsystem: Iterable[int] | None) -> int | None:
    ordered = _ordered_subsystem(num_sites, subsystem)
    if not ordered:
        return None
    if ordered != list(range(ordered[-1] + 1)):
        return None
    if ordered[-1] >= num_sites - 1:
        return None
    return ordered[-1]


def _ordered_subsystem(num_sites: int, subsystem: Iterable[int] | None) -> list[int] | None:
    if subsystem is None:
        return None
    ordered = sorted(set(int(site) for site in subsystem))
    if any(site < 0 or site >= num_sites for site in ordered):
        return None
    return ordered


def _normalize_subsystem(num_sites: int, subsystem: Iterable[int] | None) -> tuple[int, ...] | None:
    ordered = _ordered_subsystem(num_sites, subsystem)
    if subsystem is not None and ordered is None:
        raise ValueError("subsystem indices out of range")
    if ordered is None:
        return None
    return tuple(ordered)


def _supports_tdvp_initial_state(num_sites: int, initial_state: InitialStateLike) -> bool:
    backend_state = _extract_backend_state(initial_state)
    if backend_state is not None:
        state_sites = getattr(backend_state, "num_sites", None)
        return state_sites is None or int(state_sites) == num_sites
    if not isinstance(initial_state, str):
        return False
    return _canonical_product_state_label(num_sites, initial_state) is not None


def _supports_tdvp_times(times: Iterable[float]) -> bool:
    times_array = np.asarray(list(times), dtype=np.float64)
    if times_array.ndim != 1:
        return False
    if not np.all(np.isfinite(times_array)):
        return False
    if np.any(times_array < -1e-12):
        return False
    if times_array.size > 1 and np.any(np.diff(times_array) < -1e-12):
        return False
    return True


def make_initial_state(num_sites: int, state: InitialStateLike) -> np.ndarray:
    dense_state = _extract_dense_state(state)
    if dense_state is not None:
        state = dense_state
    elif _is_backend_state_handle(state):
        raise ValueError(
            "initial state is stored only as a Rust tensor-network handle; "
            "use a Rust-backed tensor-network quench path for this state"
        )
    elif isinstance(state, (GroundStateResult, TimeEvolutionResult)):
        raise ValueError(
            "initial state result does not carry a dense wavefunction; "
            "use a Rust-backed tensor-network quench path for this state"
        )

    if isinstance(state, np.ndarray):
        normalized = np.asarray(state, dtype=np.complex128)
        norm = np.linalg.norm(normalized)
        if norm == 0:
            raise ValueError("initial state must have non-zero norm")
        return normalized / norm

    canonical = _canonical_product_state_label(num_sites, state)
    if canonical is not None:
        return product_state(num_sites, canonical)

    raise ValueError(f"unsupported initial state specification: {state}")


def _extract_backend_state(state: InitialStateLike) -> object | None:
    if isinstance(state, (GroundStateResult, TimeEvolutionResult)):
        return state.backend_state
    if _is_backend_state_handle(state):
        return state
    return None


def _extract_dense_state(state: InitialStateLike) -> np.ndarray | None:
    if isinstance(state, GroundStateResult):
        return state.ground_state
    if isinstance(state, TimeEvolutionResult):
        return state.final_state
    return None


def _coerce_dense_statevector(num_sites: int, state: InitialStateLike | np.ndarray) -> np.ndarray | None:
    if isinstance(state, np.ndarray):
        dense_state = np.asarray(state, dtype=np.complex128)
    else:
        dense_state = _extract_dense_state(state)
        if dense_state is not None:
            dense_state = np.asarray(dense_state, dtype=np.complex128)
    if dense_state is None:
        return None
    if dense_state.ndim != 1:
        return None
    expected_dim = 1 << num_sites
    if dense_state.size != expected_dim:
        return None
    return dense_state


def _is_backend_state_handle(state: object) -> bool:
    return (
        not isinstance(state, (GroundStateResult, TimeEvolutionResult, str, np.ndarray))
        and hasattr(state, "num_sites")
        and hasattr(state, "solver")
        and (
            hasattr(state, "save_json")
            or hasattr(state, "to_json")
            or type(state).__name__ == "TensorNetworkState1D"
        )
    )


def _resolve_tdvp_initial_state_label(num_sites: int, initial_state: InitialStateLike) -> str:
    if not isinstance(initial_state, str):
        raise ValueError(
            "Rust TDVP currently supports string-labeled product states or backend state handles"
        )
    canonical = _canonical_product_state_label(num_sites, initial_state)
    if canonical is not None:
        return canonical
    raise ValueError(f"unsupported TDVP initial state specification: {initial_state}")


def basis_state(num_sites: int, bitstring: str) -> np.ndarray:
    if len(bitstring) != num_sites:
        raise ValueError("bitstring length must match num_sites")
    vector = np.zeros(1 << num_sites, dtype=np.complex128)
    vector[int(bitstring, 2)] = 1.0
    return vector


def product_state(num_sites: int, specification: str) -> np.ndarray:
    canonical = _canonical_product_state_label(num_sites, specification)
    if canonical is None:
        raise ValueError(f"unsupported product-state specification: {specification}")

    state = np.asarray([1.0 + 0.0j], dtype=np.complex128)
    for symbol in canonical:
        state = np.kron(state, _single_site_product_state(symbol))
    return np.asarray(state, dtype=np.complex128)


def _canonical_product_state_label(num_sites: int, label: str) -> str | None:
    if label == "all_up":
        return "0" * num_sites
    if label == "all_down":
        return "1" * num_sites
    if label == "neel":
        return "".join("10"[(site % 2)] for site in range(num_sites))
    if label == "plus_x":
        return "+" * num_sites
    if label == "minus_x":
        return "-" * num_sites
    if label == "plus_y":
        return "R" * num_sites
    if label == "minus_y":
        return "L" * num_sites
    if label == "domain_wall":
        split = num_sites // 2
        return "".join("0" if site < split else "1" for site in range(num_sites))
    if label == "anti_domain_wall":
        split = num_sites // 2
        return "".join("1" if site < split else "0" for site in range(num_sites))

    if len(label) != num_sites:
        return None
    normalized = label.replace("r", "R").replace("l", "L")
    if set(normalized).issubset(PRODUCT_STATE_CHARS):
        return normalized
    return None


def _single_site_product_state(symbol: str) -> np.ndarray:
    if symbol == "0":
        return np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    if symbol == "1":
        return np.asarray([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    scale = 1.0 / np.sqrt(2.0)
    if symbol == "+":
        return np.asarray([scale, scale], dtype=np.complex128)
    if symbol == "-":
        return np.asarray([scale, -scale], dtype=np.complex128)
    if symbol == "R":
        return np.asarray([scale, 1.0j * scale], dtype=np.complex128)
    if symbol == "L":
        return np.asarray([scale, -1.0j * scale], dtype=np.complex128)
    raise ValueError(f"unsupported single-site product-state symbol: {symbol}")


def entanglement_entropy(state: np.ndarray, num_sites: int, subsystem: list[int]) -> float:
    probabilities = entanglement_spectrum(state, num_sites, subsystem)
    probabilities = probabilities[probabilities > 1e-12]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def entanglement_spectrum(state: np.ndarray, num_sites: int, subsystem: list[int]) -> np.ndarray:
    if not subsystem:
        return np.asarray([1.0], dtype=np.float64)
    if any(site < 0 or site >= num_sites for site in subsystem):
        raise ValueError("subsystem indices out of range")

    complement = [site for site in range(num_sites) if site not in subsystem]
    tensor = np.reshape(state, (2,) * num_sites)
    permutation = subsystem + complement
    reordered = np.transpose(tensor, axes=permutation)
    reshaped = np.reshape(reordered, (1 << len(subsystem), 1 << len(complement)))
    singular_values = np.linalg.svd(reshaped, compute_uv=False)
    probabilities = singular_values**2
    total = float(np.sum(probabilities))
    if total <= 1e-30:
        return np.asarray([1.0], dtype=np.float64)
    probabilities = probabilities / total
    probabilities = probabilities[probabilities > 1e-12]
    return np.asarray(probabilities, dtype=np.float64)


def _resolve_observable(model: AnyModel, label: str) -> np.ndarray:
    if label == "energy":
        return model.hamiltonian()
    if label == "magnetization_z":
        operator = sum(
            pauli_string_operator(model.num_sites, {site: "Z"})
            for site in range(model.num_sites)
        )
        return operator / model.num_sites
    if label == "staggered_magnetization_z":
        operator = sum(
            ((-1) ** site) * pauli_string_operator(model.num_sites, {site: "Z"})
            for site in range(model.num_sites)
        )
        return operator / model.num_sites

    tokens = PAULI_TOKEN.findall(label)
    if not tokens or "".join(f"{pauli}{site}" for pauli, site in tokens) != label:
        raise ValueError(
            "observable labels must be 'energy', 'magnetization_z', "
            "'staggered_magnetization_z', or Pauli strings like 'Z0Z1'"
        )
    terms = {int(site): pauli for pauli, site in tokens}
    return pauli_string_operator(model.num_sites, terms)


def _expectation_value(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(state, operator @ state)))
