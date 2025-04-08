"""main.py for Bluesky Prototype"""

# Import packages
from pathlib import Path
import tomllib
import os
import argparse
import types
import shutil

# Import python modules
from definitions import PROJECT_ROOT


# Configuration Class for Data Settings
class Config_settings:
    """Generates the model settings that are used to generate the input data"""

    def __init__(self, args: argparse.Namespace | None = None):
        """Creates configuration object upon instantiation"""

        # __INIT__: Grab arguments
        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.output_dir = None
            self.args.debug = False

        # __INIT__: Setting output paths
        self.OUTPUT_ROOT = self._establish_output_dir(self.args)

        # __INIT__: residential settings
        # TODO: establish configuration file
        self.build_stock_db = True
        self.test_build_data = False

    def _establish_output_dir(self, args=None):
        """Setup the output directory and write out its name for other scripts to grab

        Parameters
        ----------
        args : argparse.Namespace, optional
            arguments, by default None

        Returns
        -------
        os.Path
            path for output directory


        Raises
        ------
        FileNotFoundError
            Error if output path does not exist"
        """

        if args.output_dir is not None:
            output_dir = Path(self.args.output_dir)
        else:
            output_dir = Path(PROJECT_ROOT / 'output')
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"The output path '{output_dir}' does not exist.")

        return output_dir
