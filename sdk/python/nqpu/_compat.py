"""Compatibility helpers for optional Rust bindings."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from types import ModuleType
import sys


def _candidate_binding_paths() -> list[Path]:
    sdk_dir = Path(__file__).resolve().parents[2]
    return [
        sdk_dir / "rust" / "target" / "debug",
        sdk_dir / "rust" / "target" / "release",
    ]


@lru_cache(maxsize=1)
def get_rust_bindings() -> ModuleType | None:
    """Return the compiled `nqpu_metal` module when available."""
    try:
        return import_module("nqpu_metal")
    except ImportError:
        for path in _candidate_binding_paths():
            if not path.is_dir():
                continue
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            try:
                return import_module("nqpu_metal")
            except ImportError:
                continue
    return None


def has_rust_bindings() -> bool:
    return get_rust_bindings() is not None
