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


try:
    _module = _load_frames_module()
except ModuleNotFoundError as exc:
    if exc.name != 'pandas':  # pragma: no cover - propagate unexpected errors
        raise

    original_exc = exc

    def _raise_pandas_error() -> None:
        raise ImportError(
            "pandas is required for io.frames_api; install it with `pip install pandas`."
        ) from original_exc

    class Frames:  # type: ignore[no-redef]
        """Placeholder that raises when pandas-dependent frames are unavailable."""

        def __init__(self, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        @classmethod
        def coerce(cls, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        def __getattr__(self, _name):  # pragma: no cover - simple guard
            _raise_pandas_error()

    class PolicySpec:  # type: ignore[no-redef]
        """Placeholder that raises when pandas-dependent policy specs are unavailable."""

        def __init__(self, *_args, **_kwargs):  # pragma: no cover - simple guard
            _raise_pandas_error()

        def to_policy(self):  # pragma: no cover - simple guard
            _raise_pandas_error()
else:
    Frames = getattr(_module, 'Frames')
    PolicySpec = getattr(_module, 'PolicySpec')

__all__ = ['Frames', 'PolicySpec']
