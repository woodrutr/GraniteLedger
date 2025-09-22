"""main.py for Bluesky Prototype"""

# Import packages
from pathlib import Path
import tomllib
import os
import argparse
import types
import shutil

# Import python modules
from graniteledger.definitions import PROJECT_ROOT


# Configuration Class for Data Settings
class Config_settings:
    """Generates the model settings that are used to generate the input data"""

    def __init__(self, config_path: Path, args: argparse.Namespace | None = None):
        """Creates configuration object upon instantiation"""

        # __INIT__: Grab arguments
        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.output_dir = None
            self.args.data = None
            self.args.debug = False

        # __INIT__: Setting output paths
        self.OUTPUT_ROOT = self._establish_output_dir(self.args)

        # Grab configs
        with open(config_path, 'rb') as src:
            config = tomllib.load(src)

        # __INIT__: electricity option settings
        self.data_choices = list(config['executions'].keys())

        if self.args.data is not None:
            for data_run in self.data_choices:
                setattr(self, data_run, False)
            setattr(self, self.args.data, True)

        else:
            for data_run in self.data_choices:
                setattr(self, data_run, config['executions'][data_run])

        self.executions = []
        for execution in config['executions']:
            if getattr(self, execution) is True:
                self.executions.append(execution)

        # __INIT__: electricity value settings
        self.first_year = config['first_year']
        self.last_year = config['last_year']
        self.year_range = range(self.first_year, self.last_year + 1)
        self.population_year = config['population_year']
        self.EIA860m_excel_name = config['EIA860m_excel_name']

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
            Error if output path does not exist
        """
        if args.output_dir is not None:
            output_dir = Path(self.args.output_dir)
        else:
            output_dir = Path(PROJECT_ROOT / 'output')
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"The output path '{output_dir}' does not exist.")

        return output_dir
