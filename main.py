"""main.py for Bluesky Prototype"""

# Import packages
import os
import logging
from pathlib import Path
import tomllib
import types

# Import python modules
from definitions import PROJECT_ROOT
from src.common.config_setup import Config_settings
from src.common.utilities import setup_logger, get_args
from src.integrator.gaussseidel import run_gs
from src.integrator.unified import run_unified
from src.integrator.runner import run_standalone, run_elec_solo, run_h2_solo, run_residential_solo

# Specify config path
default_config_path = Path(PROJECT_ROOT, 'src/common', 'run_config.toml')


def app_main(selected_mode, capacity_build_limits=None):
    """main run through the bsky gui app

    Parameters
    ----------
    selected_mode : str
        selected mode to run model
    capacity_build_limits : Mapping | None
        Optional nested mapping of capacity limits provided by the GUI
    """
    app_args = types.SimpleNamespace()
    app_args.op_mode = selected_mode
    app_args.debug = False
    app_args.output_name = None
    app_args.capacity_build_limits = capacity_build_limits

    app_settings = Config_settings(config_path=default_config_path, args=app_args)
    main(app_settings)


def main(settings: Config_settings | None = None):
    """
    Runs model as defined in settings

    Parameters
    -------
    args: settings
        Contains configuration settings for which models and solvers to run
    """
    # MAIN - Parse the args to get selected mode if one is provided
    args = get_args()

    # MAIN - Instantiate config object if none passed
    if not settings:
        settings = Config_settings(config_path=default_config_path, args=args)

    # MAIN - Establish the logger
    logger = setup_logger(settings)
    logger = logging.getLogger(__name__)

    # MAIN - Log settings
    logger.info('Starting Logging')
    logger.debug('Logging level set to DEBUG')
    logger.info(f'Model running in: {settings.selected_mode} mode')
    logger.info('Config settings:')
    logger.info(f'Regions: {settings.regions}')
    logger.info(f'Years: {settings.years}')
    with open(default_config_path, 'rb') as f:
        data = tomllib.load(f)
    config_list = []
    for key, value in data.items():
        config_list.append(f'{key}: {value}')
    logger.info(config_list)
    logger.debug(f'Config settings: these settings dont have checks: {settings.missing_checks}')
    settings.cw_temporal.to_csv(Path(settings.OUTPUT_ROOT / 'cw_temporal.csv'), index=False)

    # MAIN - Run the cases you want to run based on the mode and settings you pass
    runner = globals()[settings.run_method]
    runner(settings)

    # MAIN - print the output directory once run is finished
    path_parts = os.path.normpath(settings.OUTPUT_ROOT).split(os.sep)
    output_name = os.path.join(path_parts[-2], path_parts[-1])
    print(f'Results located in: {output_name}')


if __name__ == '__main__':
    main()
