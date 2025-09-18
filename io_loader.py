"""Utility helpers for loading the ``io.frames_api`` module safely."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys


def _load_frames_module():
    module_name = 'io.frames_api'
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = Path(__file__).resolve().parent / 'io' / 'frames_api.py'
    spec = util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f'unable to locate frames_api module at {module_path}')

    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_module = _load_frames_module()

Frames = getattr(_module, 'Frames')
PolicySpec = getattr(_module, 'PolicySpec')

__all__ = ['Frames', 'PolicySpec']
