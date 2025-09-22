"""Streamlit entrypoint for the BlueSky policy simulator.

This wrapper module delegates to :mod:`graniteledger.gui.app` so commands like
``streamlit run app.py`` render the Streamlit interface.
"""

from graniteledger.gui import app as streamlit_app


def main() -> None:
    """Execute the Streamlit application defined in :mod:`graniteledger.gui.app`."""

    streamlit_app.main()


if __name__ == "__main__":
    main()
