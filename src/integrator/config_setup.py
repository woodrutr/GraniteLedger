"""This file contains Config_settings class. It establishes the main settings used when running
the model. It takes these settings from the run_config.toml file. It contains universal configurations
(e.g., configs that cut across modules and/or solve options) and module specific configs."""

###################################################################################################
# Setup

# Import packages
from logging import getLogger
import pandas as pd
import numpy as np
import tomllib
from pathlib import Path
from datetime import datetime
import types
import argparse

# Import scripts
from definitions import PROJECT_ROOT
from src.integrator.utilities import create_temporal_mapping
from src.integrator.utilities import make_dir

# Establish logger
logger = getLogger(__name__)


###################################################################################################
# Configuration Class


class Config_settings:
    """Generates the model settings that are used to solve. Settings include:
    - Iterative Solve Config Settings
    - Spatial Config Settings
    - Temporal Config Settings
    - Electricity Config Settings
    - Other
    """

    def __init__(
        self,
        config_path: Path,
        args: argparse.Namespace | None = None,
        test=False,
        years_ow=[],
        regions_ow=[],
    ):
        """Creates configuration object upon instantiation

        Parameters
        ----------
        config_path : Path
            Path to run_config.toml
        args : Namespace
            Parsed arguments fed to main.py or other parsed object
        test : bool, optional
            _description_, by default False
        years_ow : list, optional
            _description_, by default []
        regions_ow : list, optional
            _description_, by default []

        Raises
        ------
        ValueError
            No modules turned on; check run_config.toml
        ValueError
            sw_expansion: Must turn RM switch off if no expansion
        ValueError
            sw_expansion: Must turn learning switch off if no expansion

        """
        # __INIT__: Grab arguments namespace and set paths

        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.op_mode = None
            self.args.debug = False
        self.PROJECT_ROOT = PROJECT_ROOT

        # __INIT__: Dump toml, sse args to set mode
        with open(config_path, 'rb') as src:
            config = tomllib.load(src)

        ############################################################################################
        # Universal Configs

        # __INIT__: Default Solve Mode
        self.default_mode = config['default_mode']

        # __INIT__: If no mode is specified read default mode from TOML
        if not self.args.op_mode:
            print('No mode arg passed, therefore...')
            self.selected_mode = self.default_mode
            print(f'using mode {self.default_mode} specified in run_config file')
        else:
            self.selected_mode = self.args.op_mode

        # __INIT__: Setting output paths
        # Setup the output directory and write out its name for other scripts to grab
        OUTPUT_ROOT = Path(
            PROJECT_ROOT
            / 'output'
            / f"{self.selected_mode}_{datetime.now().strftime('%Y_%m_%d_%H%Mh')}"
        )

        make_dir(OUTPUT_ROOT)
        with open('output_root.txt', 'w') as file:
            file.write(str(OUTPUT_ROOT))
        self.OUTPUT_ROOT = OUTPUT_ROOT

        #####
        ### __INIT__: Methods and Modules Configuration
        #####

        # __INIT__: Set modules from config
        self.electricity = config['electricity']
        self.hydrogen = config['hydrogen']
        self.residential = config['residential']

        # __INIT__: Redirects and raises based on conditions
        if (not any((self.electricity, self.hydrogen, self.residential))) and (
            self.selected_mode in ('unified-combo', 'gs-combo', 'standalone')
        ):
            raise ValueError('No modules turned on; check run_config.toml')

        # __INIT__: Single module case
        if [self.electricity, self.hydrogen, self.residential].count(True) == 1:
            print('you selected a combo mode, but only one module is turned on')
            self.run_method = 'run_standalone'

        # __INIT__: Combinations of Modules and Mode --> run guidance
        match self.selected_mode:
            case 'unified-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print(
                        'running unified-combo with electricity, hydrogen, and residential modules'
                    )
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_unified'
                self.method_options = {}
            case 'gs-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print('running gs-combo with electricity, hydrogen, and residential modules')
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_gs'
                self.method_options = {
                    'update_h2_price': self.hydrogen,
                    'update_elec_price': True,
                    'update_h2_demand': self.hydrogen,
                    'update_load': self.residential,
                    'update_elec_demand': False,
                }
            case 'standalone':
                self.run_method = 'run_standalone'
            case 'elec':
                self.run_method = 'run_elec_solo'
            case 'h2':
                self.run_method = 'run_h2_solo'
            case 'residential':
                self.run_method = 'run_residential_solo'
            case _:
                logger.error('Unkown op mode specified... exiting')

        # __INIT__: Iterative Solve Configs
        self.tol = config['tol']
        self.force_10 = config['force_10']
        self.max_iter = config['max_iter']

        # __INIT__: Spatial Configs
        if not test or len(regions_ow) == 0:
            self.regions = list(pd.read_csv(config['regions']).dropna()['region'])
        else:
            self.regions = regions_ow

        # __INIT__:  Temporal Configs
        self.sw_temporal = config['sw_temporal']
        self.cw_temporal = create_temporal_mapping(self.sw_temporal)

        # __INIT__:  Temporal Configs - Years
        self.sw_agg_years = config['sw_agg_years']
        year_frame = pd.read_csv(config['years'])
        if not test or len(years_ow) == 0:
            self.years = list(year_frame.dropna()['year'])
        else:
            self.years = years_ow

        if self.sw_agg_years and len(self.years) > 1:
            self.start_year = year_frame['year'][0]
            all_years_list = list(range(self.start_year, self.years[-1] + 1))
        else:
            self.start_year = self.years[0]
            all_years_list = self.years

        solve_array = np.array(self.years)
        mapped_list = [solve_array[solve_array >= year].min() for year in all_years_list]
        self.year_map = pd.DataFrame({'y': all_years_list, 'Map_y': mapped_list})
        self.year_weights = (
            self.year_map.groupby(['Map_y'])
            .agg('count')
            .reset_index()
            .rename(columns={'Map_y': 'y', 'y': 'year_weights'})
        )

        ############################################################################################
        # Electricity Configs
        self.sw_trade = config['sw_trade']
        self.sw_expansion = config['sw_expansion']
        self.sw_rm = config['sw_rm']
        self.sw_ramp = config['sw_ramp']
        self.sw_reserves = config['sw_reserves']
        self.sw_learning = config['sw_learning']

        # __INIT__:  throwing errors if certain combinations of switches
        if self.sw_expansion == 0:  # expansion off
            if self.sw_rm == 1:
                raise ValueError('sw_expansion: Must turn RM switch off if no expansion')
            if self.sw_learning == 1:
                raise ValueError('sw_expansion: Must turn learning switch off if no expansion')

        ############################################################################################
        # Residential Configs
        if not test or len(regions_ow) == 0:
            self.view_regions = config['view_regions']
            self.view_years = config['view_years']
            self.sensitivity = config['sensitivity']
            self.change_var = config['change_var']
            self.percent_change = config['percent_change']
            self.complex = config['complex']

        ############################################################################################
        # Hydrogen Configs
        self.h2_data_folder = self.PROJECT_ROOT / config['h2_data_folder']
