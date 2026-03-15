# Physics Research With nQPU

The new Python research surface is centered on `nqpu.physics` and the
`ModelQPU` facade. The goal is to make model-Hamiltonian work feel like a
stable workflow instead of a grab-bag of unrelated modules.

## Core API

```python
from nqpu import ModelQPU
from nqpu.physics import TransverseFieldIsing1D, HeisenbergXXZ1D, HeisenbergXYZ1D
```

Available today:

- `TransverseFieldIsing1D`
- `HeisenbergXXZ1D`
- `HeisenbergXYZ1D`
- `CustomHamiltonian`
- `AutoSolver`
- `ExactDiagonalizationSolver`
- `RustTensorNetworkSolver`
- `ModelQPU.ground_state(...)`
- `ModelQPU.spectrum(...)`
- `ModelQPU.quench(...)`
- `ModelQPU.loschmidt_echo(...)`
- `ModelQPU.dqpt_diagnostics(...)`
- `ModelQPU.sweep_parameter(...)`

`ModelQPU()` now defaults to `AutoSolver`, which keeps exact diagonalization for
small systems and can switch to Rust-backed tensor-network solvers for larger
1D open-boundary spin chains when the optional `nqpu_metal` bindings are
installed.

## Ground States

```python
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=6, coupling=1.0, transverse_field=0.7)

result = qpu.ground_state(
    model,
    observables=["magnetization_z", "Z0Z1"],
    subsystem=[0, 1, 2],
)

print(result.ground_state_energy)
print(result.spectral_gap)
print(result.observables)
print(result.entanglement_entropy)
```

Rust-backed ground states can now also warm-start from a prior tensor-network
state handle:

```python
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
initial = qpu.ground_state(TransverseFieldIsing1D(num_sites=16, transverse_field=0.5))
retuned = qpu.ground_state(
    TransverseFieldIsing1D(num_sites=16, transverse_field=0.6),
    initial_state=initial,
    observables=["magnetization_z"],
)
```

Supported observables:

- `energy`
- `magnetization_z`
- `staggered_magnetization_z`
- Pauli strings like `Z0`, `X0X1`, `Z0Z1`

## Parameter Sweeps

```python
import numpy as np
from nqpu import ModelQPU
from nqpu.physics import HeisenbergXXZ1D

qpu = ModelQPU()
model = HeisenbergXXZ1D(num_sites=6, anisotropy=0.5)

sweep = qpu.sweep_parameter(
    model,
    "anisotropy",
    np.linspace(0.25, 1.5, 11),
    observables=["magnetization_z"],
)

print(sweep.values)
print(sweep.energies)
print(sweep.spectral_gaps)
print(sweep.entanglement_entropy)
print(sweep.observables["magnetization_z"])
```

Sweeps now preserve the solver used at each point plus per-point model metadata,
spectral gaps when available, and entanglement entropy when `subsystem=[...]`
is requested.

You can also adaptively refine a coarse sweep to zoom into likely transition
regions:

```python
import numpy as np
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=8, transverse_field=0.7)

adaptive = qpu.adaptive_sweep_parameter(
    model,
    "transverse_field",
    [0.2, 0.6, 1.0, 1.4, 1.8],
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3],
    metric="spectral_gap",
    max_refinement_rounds=2,
    refinements_per_round=1,
    checkpoint_path="checkpoints/tfim_adaptive.json",
)

print(adaptive.values)
print(adaptive.refinement_history)
```

The default `metric="spectral_gap"` inserts midpoints in the interval with the
smallest average coarse-grid gap. Other supported refinement metrics are
`energy`, `entanglement_entropy`, or any requested observable label such as
`magnetization_z`. If you use an observable as the refinement metric and did
not request it explicitly, the adaptive sweep will add it automatically so the
refinement trace is available in the saved result.

Adaptive sweeps also support multiple interval-selection strategies:

- `strategy="minimum_average"`: choose the interval with the lowest average
  metric value. This is the default for `metric="spectral_gap"`.
- `strategy="gradient_magnitude"`: choose the interval with the largest
  coarse-grid slope. This is the default for non-gap metrics.
- `strategy="curvature"`: choose the interval adjacent to the largest discrete
  second-derivative signal.
- `strategy="target_crossing"` with `target_value=...`: choose the interval
  whose metric straddles or comes closest to a requested target value.

Point placement inside the selected interval is controlled separately:

- `insertion_policy="equal_spacing"`: add evenly spaced interior points. This
  is the default for non-crossing refinement and supports
  `points_per_interval > 1`.
- `insertion_policy="target_linear"`: add one asymmetric point at the linear
  interpolation estimate of the requested target crossing. This is the default
  for `strategy="target_crossing"`.

Example target-crossing zoom on an order parameter:

```python
adaptive = qpu.adaptive_sweep_parameter(
    model,
    "transverse_field",
    [0.2, 0.6, 1.0, 1.4, 1.8],
    metric="magnetization_z",
    strategy="target_crossing",
    target_value=0.0,
    insertion_policy="target_linear",
    max_refinement_rounds=2,
)
```

Example multi-point equal-spacing refinement inside a selected interval:

```python
adaptive = qpu.adaptive_sweep_parameter(
    model,
    "transverse_field",
    [0.2, 0.6, 1.0, 1.4, 1.8],
    metric="magnetization_z",
    strategy="target_crossing",
    target_value=0.0,
    insertion_policy="equal_spacing",
    points_per_interval=2,
    max_refinement_rounds=1,
)
```

Long sweeps can also checkpoint progress and resume without recomputing earlier
grid points:

```python
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=12, transverse_field=0.7)
values = [0.4, 0.6, 0.8, 1.0]

partial_or_complete = qpu.sweep_parameter(
    model,
    "transverse_field",
    values,
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3, 4, 5],
    checkpoint_path="checkpoints/tfim_resume.json",
)

resumed = qpu.sweep_parameter(
    model,
    "transverse_field",
    values,
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3, 4, 5],
    checkpoint_path="checkpoints/tfim_resume.json",
    resume=True,
)

print(resumed.completed_points, resumed.is_complete)
```

For Rust-backed sweeps, the checkpoint now includes a tensor-network state
sidecar for each completed sweep point. On resume, the next unfinished point is
warm-started from the restored backend state of the nearest completed point in
parameter space, not just the previous grid index.
Adaptive sweeps use the same checkpoint format and also persist the evolving
refinement grid plus `refinement_history`, so a resumed run continues from the
already-expanded scan instead of restarting from the original seed values.

## Real-Time Dynamics

```python
import numpy as np
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=5, transverse_field=1.0)

evolution = qpu.quench(
    model,
    times=np.linspace(0.0, 5.0, 51),
    initial_state="neel",
    observables=["magnetization_z", "Z0Z1"],
    subsystem=[0, 1],
)

print(evolution.times)
print(evolution.observables["magnetization_z"])
print(evolution.entanglement_entropy)
```

Prepared states can now flow directly into later quenches:

```python
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
initial_model = TransverseFieldIsing1D(num_sites=10, transverse_field=0.6)
final_model = TransverseFieldIsing1D(num_sites=10, transverse_field=1.0)

ground = qpu.ground_state(initial_model)
evolution = qpu.quench(
    final_model,
    times=[0.0, 0.05, 0.10],
    initial_state=ground,
    observables=["magnetization_z", "Z0Z1"],
    subsystem=[0, 1, 2, 3, 4],
)
```

## Correlations and Structure Factors

You can derive equal-time correlation matrices and both static and dynamical
structure factors from the same `ModelQPU` interface:

```python
import numpy as np
from nqpu import ModelQPU, TransverseFieldIsing1D

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=6, transverse_field=0.8)

correlations = qpu.correlation_matrix(model, pauli="Z", connected=True)
static_sf = qpu.structure_factor(
    model,
    momenta=[0.0, np.pi / 2.0, np.pi],
    pauli="Z",
    connected=True,
)
dynamic_sf = qpu.dynamic_structure_factor(
    model,
    times=[0.0, 0.05, 0.10],
    momenta=[0.0, np.pi],
    pauli="Z",
    connected=True,
    initial_state="neel",
)
frequency_sf = qpu.frequency_structure_factor(
    dynamic_sf,
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)
two_time = qpu.two_time_correlator(
    model,
    times=[0.0, 0.05, 0.10],
    pauli="Z",
    connected=True,
)
local_response = qpu.linear_response_spectrum(
    model,
    times=[0.0, 0.05, 0.10],
    pauli="Z",
    source_sites=[0, 2, 4],
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)
spectrum = qpu.entanglement_spectrum(
    model,
    subsystem=[0, 1, 2, 3],
    num_levels=4,
)
response = qpu.linear_response_spectrum(
    two_time,
    momenta=[0.0, np.pi],
    frequencies=[-20.0, 0.0, 20.0],
    window="hann",
)

print(correlations.matrix.shape)
print(static_sf.values)
print(dynamic_sf.values.shape)
print(frequency_sf.intensity)
print(two_time.values.shape)
print(local_response.measure_sites)
print(spectrum.eigenvalues, spectrum.entanglement_energies)
print(response.intensity)
```

Notes:

- `pauli=` can be `"X"`, `"Y"`, or `"Z"`
- `connected=True` subtracts disconnected products such as
  `<Z_i><Z_j>` before forming the matrix or structure factor
- if `momenta` is omitted, the static and dynamic structure-factor helpers use
  the default finite-size grid `2*pi*n/N`
- the dynamic helper is built from the same one- and two-site Pauli traces as
  `quench(...)`, so it scales like `O(N^2)` requested observables
- `frequency_structure_factor(...)` applies a direct Fourier transform over the
  time-domain structure factor and returns a complex spectrum plus a convenience
  `intensity` view equal to `abs(values)`
- you can call `ModelQPU.frequency_structure_factor(...)` directly on a model
  and time grid, or reuse an existing `DynamicStructureFactorResult`
- this is currently a Fourier transform of the quench-time equal-time
  structure factor, not yet a full two-time correlation-function computation of
  canonical `S(q, \omega)`
- `two_time_correlator(...)` returns the full time-dependent site-by-site
  correlator tensor `<O_i(t) O_j(0)>`, and `linear_response_spectrum(...)`
  builds a commutator-based response spectrum from that tensor
- `measure_sites=` and `source_sites=` let you restrict those calculations to
  a selected set of sites; for response calculations, passing only one of them
  reuses that subset for both source and measurement sites
- `entanglement_spectrum(...)` returns reduced-density-matrix eigenvalues,
  Schmidt values, and entanglement energies for a chosen subsystem
- `loschmidt_echo(...)` returns the return amplitude, echo probability, and
  per-site return rate for a chosen initial/reference state pair
- when a Rust-backed DMRG ground state is available, those helpers can also use
  an approximate TDVP-transition path on supported open-boundary 1D spin
  chains, which evolves the locally perturbed source state and contracts it
  against the fixed ground-state reference
- the Rust entanglement-spectrum fast path currently supports only prefix
  subsystems that map to a single MPS bond
- `loschmidt_echo(...)` can also use a Rust TDVP-overlap path on the same
  supported 1D models when the initial/reference states are product states or
  Rust tensor-network backend states
- Rust-backed Loschmidt echo manifests retain the evolved tensor-network state
  as a sidecar checkpoint when saved with `save_loschmidt_echo_result(...)`
- for Rust-backed runs, the same model and observable restrictions as the
  DMRG/TDVP bridge still apply

Rust-backed tensor-network states can also be checkpointed and restored:

```python
from nqpu import (
    ModelQPU,
    TransverseFieldIsing1D,
    load_entanglement_spectrum_result,
    load_loschmidt_echo_result,
    load_response_spectrum_result,
    load_ground_state_result,
    load_sweep_result,
    load_tensor_network_state,
    load_two_time_correlator_result,
    save_entanglement_spectrum_result,
    save_loschmidt_echo_result,
    save_response_spectrum_result,
    save_ground_state_result,
    save_sweep_result,
    save_tensor_network_state,
    save_two_time_correlator_result,
)

qpu = ModelQPU()
model = TransverseFieldIsing1D(num_sites=16, transverse_field=0.7)
ground = qpu.ground_state(model)

save_tensor_network_state(ground, "checkpoints/tfim_gs.json")
restored_state = load_tensor_network_state("checkpoints/tfim_gs.json")
save_ground_state_result(ground, "checkpoints/tfim_gs_result.json")
restored_ground = load_ground_state_result("checkpoints/tfim_gs_result.json")
entanglement = qpu.entanglement_spectrum(model, subsystem=[0, 1, 2, 3], num_levels=4)
save_entanglement_spectrum_result(entanglement, "checkpoints/tfim_entanglement.json")
restored_entanglement = load_entanglement_spectrum_result(
    "checkpoints/tfim_entanglement.json"
)
save_two_time_correlator_result(two_time, "checkpoints/tfim_two_time.json")
restored_two_time = load_two_time_correlator_result("checkpoints/tfim_two_time.json")
save_response_spectrum_result(response, "checkpoints/tfim_response.json")
restored_response = load_response_spectrum_result("checkpoints/tfim_response.json")
echo = qpu.loschmidt_echo(model, times=[0.0, 0.05, 0.10], initial_state="neel")
save_loschmidt_echo_result(echo, "checkpoints/tfim_echo.json")
restored_echo = load_loschmidt_echo_result("checkpoints/tfim_echo.json")
sweep = qpu.sweep_parameter(
    model,
    "transverse_field",
    [0.5, 0.7, 0.9],
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3, 4, 5, 6, 7],
)
save_sweep_result(sweep, "checkpoints/tfim_sweep.json")
restored_sweep = load_sweep_result("checkpoints/tfim_sweep.json")

evolution = qpu.quench(
    TransverseFieldIsing1D(num_sites=16, transverse_field=1.0),
    times=[0.0, 0.05, 0.10],
    initial_state=restored_ground,
    observables=["magnetization_z"],
    subsystem=[0, 1, 2, 3, 4, 5, 6, 7],
)
```

Supported initial states:

- `"all_up"`
- `"all_down"`
- `"neel"`
- `"plus_x"`, `"minus_x"`, `"plus_y"`, `"minus_y"`
- `"domain_wall"`, `"anti_domain_wall"`
- a computational-basis bitstring like `"010101"`
- a per-site product-state string like `"+-RL01"`
- a normalized statevector
- a prior `GroundStateResult` or `TimeEvolutionResult`
- a loaded Rust `TensorNetworkState1D` checkpoint handle

For larger open-boundary 1D chains, `ModelQPU.quench(...)` can now switch to
the Rust TDVP path automatically when:

- the model is `TransverseFieldIsing1D`, `HeisenbergXXZ1D`, or `HeisenbergXYZ1D`
- the time points are non-negative and nondecreasing
- the initial state is a supported product-state label such as `"all_up"`,
  `"neel"`, `"plus_x"`, or `"domain_wall"`, a per-site product string like
  `"+-RL01"`, a dense statevector, or a prior `GroundStateResult` /
  `TimeEvolutionResult`; dense exact states are compressed into an MPS handle
  before the Rust TDVP path runs
- observables are `energy`, `magnetization_z`, `staggered_magnetization_z`, or
  one- or two-site Pauli strings like `Z0Z1`
- optional entropy tracing uses `subsystem=[...]`; the Rust fast path currently
  requires that subsystem to be a contiguous prefix such as `[0, 1, 2]`

For return-rate analysis, `ModelQPU.dqpt_diagnostics(...)` builds on
`ModelQPU.loschmidt_echo(...)` and reports candidate cusp times from the
discrete return-rate trace. The helper exposes `min_prominence=...` and
`min_cusp=...` filters so you can suppress shallow oscillations before
refining a candidate DQPT window.

Cross-model reuse of Rust tensor-network states is now explicit. If a backend
state was prepared under one Hamiltonian and then reused under another,
`ModelQPU.quench(...)`, `ModelQPU.loschmidt_echo(...)`, and
`ModelQPU.dqpt_diagnostics(...)` require `initial_state_model=` or
`reference_state_model=` so mismatched state handles are rejected unless the
source model is declared deliberately. The same explicit provenance rule also
applies to Rust-backed `ModelQPU.two_time_correlator(...)` and
`ModelQPU.linear_response_spectrum(...)` runs that start from a prepared
tensor-network state.

DQPT diagnostics can now also be checkpointed like the other research result
types with `save_dqpt_diagnostics_result(...)` and
`load_dqpt_diagnostics_result(...)`, including the optional backend-state
sidecar when the diagnostics were produced by the Rust Loschmidt path.
`ModelQPU.scan_dqpt_parameter(...)` extends that into a checkpointable
parameter-scan workflow with per-point return-rate traces and detected cusp
candidates across a sweep grid. `ModelQPU.adaptive_scan_dqpt_parameter(...)`
adds adaptive refinement on top of that fixed-grid scan, using metrics like
`strongest_cusp_strength`, `strongest_prominence`, `candidate_count`, or
`strongest_candidate_time` together with the existing refinement strategies and
insertion policies.
`ModelQPU.adaptive_dqpt_diagnostics(...)` adds the complementary time-domain
workflow, refining the DQPT time grid itself against `return_rate`, `echo`, or
`amplitude_magnitude` before or alongside a parameter scan.

## Current Limits

- Exact diagonalization still handles spectra and remains the fallback for
  unsupported time grids or arbitrary statevectors.
- Rust-backed DMRG currently covers only ground states for open-boundary
  `TransverseFieldIsing1D`, `HeisenbergXXZ1D`, and `HeisenbergXYZ1D` chains.
- Rust-backed TDVP currently covers real-time quenches for the same
  open-boundary 1D models from product-state initial conditions or previously
  prepared Rust tensor-network states.
- Quench entanglement traces are currently one cut at a time. Exact
  diagonalization supports any subsystem; the Rust TDVP path supports prefix
  cuts that map to a single MPS bond.
- Tensor-network checkpoints currently use a JSON format intended for
  same-project research workflows, not a long-term cross-version interchange
  guarantee.
- Saved result manifests preserve observables, entropy traces, model metadata,
  and optionally a sidecar backend-state checkpoint. Saved sweep manifests
  preserve phase-diagram data, solver provenance, per-point metadata, and
  `completed_points` for resumable runs. Adaptive sweeps also preserve the
  evolving sweep grid, refinement metric, refinement strategy, target value
  when used, insertion policy, points-per-interval, and `refinement_history`.
  DQPT scan manifests now preserve the same adaptive metadata and can also
  persist a backend-state sidecar for each completed point so resumed Rust
  scans can restart from the last completed tensor-network state.
- The Rust tensor-network bridge currently supports `energy`,
  `magnetization_z`, `staggered_magnetization_z`, and one- or two-site Pauli
  strings such as `Z0` or `X0X1`.
- Correlation matrices and structure factors are derived from one- and two-site
  Pauli traces, so they currently scale like `O(N^2)` observables and are most
  practical for moderate system sizes.
- Two-time correlators and linear-response spectra currently use dense exact
  diagonalization by default. A limited approximate Rust path is available only
  for DMRG-ground-state references on supported open-boundary 1D spin chains,
  and it scales as one TDVP evolution per source site.
- Rust-backed sweep checkpoints still resume at point boundaries. If a single
  DMRG solve is interrupted mid-point, that point restarts from the last
  completed point's state rather than an intra-point partial solve state.
- General TDVP workflows beyond product-state quenches, plus the broader Rust
  many-body stack, are still not wired into the Python research surface.
