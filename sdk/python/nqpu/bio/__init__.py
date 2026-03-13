"""Lazy biology exports for optional bio modules."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "QuantumDNA": ("bio.quantum_organism", "QuantumDNA"),
    "QuantumCreature": ("bio.quantum_life", "QuantumCreature"),
    "QuantumEcosystem": ("bio.quantum_life", "QuantumEcosystem"),
    "CreatureTraits": ("bio.quantum_life", "CreatureTraits"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
