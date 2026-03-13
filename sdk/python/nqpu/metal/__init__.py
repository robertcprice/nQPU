"""
nQPU Metal Python utilities.

Provides Python-level circuit optimization, QASM parsing,
Qiskit compatibility, and error mitigation wrapping the Rust core.
"""

from __future__ import annotations

from importlib import import_module

from .._compat import has_rust_bindings

RUST_BINDINGS_AVAILABLE = has_rust_bindings()
QISKIT_AVAILABLE = False

_EXPORTS = {
    "QASMGate": ("nqpu.metal.qasm", "QASMGate"),
    "QASMImporter": ("nqpu.metal.qasm", "QASMImporter"),
    "QASMExporter": ("nqpu.metal.qasm", "QASMExporter"),
    "MitigationMethod": ("nqpu.metal.error_mitigation", "MitigationMethod"),
    "MitigationResult": ("nqpu.metal.error_mitigation", "MitigationResult"),
    "ReadoutErrorMitigator": ("nqpu.metal.error_mitigation", "ReadoutErrorMitigator"),
    "ZeroNoiseExtrapolator": ("nqpu.metal.error_mitigation", "ZeroNoiseExtrapolator"),
    "CircuitOptimizer": ("nqpu.metal.optimizer", "CircuitOptimizer"),
    "OptimizationLevel": ("nqpu.metal.optimizer", "OptimizationLevel"),
    "Gate": ("nqpu.metal.optimizer", "Gate"),
    "NQPUBackend": ("nqpu.metal.qiskit_compat", "NQPUBackend"),
    "NQPUJob": ("nqpu.metal.qiskit_compat", "NQPUJob"),
    "get_backend": ("nqpu.metal.qiskit_compat", "get_backend"),
}


def __getattr__(name: str):
    global QISKIT_AVAILABLE
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc

    if not RUST_BINDINGS_AVAILABLE and module_name != "nqpu.metal.qiskit_compat":
        raise AttributeError(
            f"{name} requires the optional Rust bindings; build or install `nqpu_metal` first"
        )

    module = import_module(module_name)
    value = getattr(module, attr_name)
    if module_name == "nqpu.metal.qiskit_compat":
        QISKIT_AVAILABLE = True
    globals()[name] = value
    return value


__all__ = ["QISKIT_AVAILABLE", "RUST_BINDINGS_AVAILABLE", *_EXPORTS.keys()]
