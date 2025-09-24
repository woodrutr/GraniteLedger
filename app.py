"""Entrypoint helpers for the GraniteLedger user interfaces."""

import importlib

from gui import app as streamlit_app

__all__ = ['main']


def main() -> None:
    """Execute the Streamlit application defined in :mod:`gui.app`."""

    streamlit_app.main()


if __name__ == "__main__":
    main()


def __getattr__(name: str):
    """Lazily expose Dash helpers without requiring Dash dependencies at import."""

    if name in {'run_mode', 'app_main'}:
        dash_app = importlib.import_module('gui.dash_app')
        return getattr(dash_app, name)
    raise AttributeError(f"module 'app' has no attribute {name!r}")
