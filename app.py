"""Streamlit entrypoint for the BlueSky policy simulator.

This wrapper module delegates to :mod:`gui.app` so commands like
``streamlit run app.py`` render the Streamlit interface.
"""

from gui import app as streamlit_app


def main() -> None:
    """Execute the Streamlit application defined in :mod:`gui.app`."""

    streamlit_app.main()


if __name__ == "__main__":
    main()
