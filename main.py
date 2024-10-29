"""main.py for Bluesky Prototype"""

import os
import argparse
import logging
from pathlib import Path

from definitions import PROJECT_ROOT
from src.integrator.config_setup import Config_settings
from src.integrator.utilities import setup_logger

from src.integrator.gaussseidel import run_gs
from src.integrator.unified import run_unified
from src.integrator.runner import run_standalone, run_elec_solo, run_h2_solo, run_residential_solo

# Specify config path
default_config_path = Path(PROJECT_ROOT, 'src/integrator', 'run_config.toml')


def get_args():
    """_summary_

    Returns
    -------
    args: Namespace
        Contains arguments pass to main.py executable
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='description:\n'
        '\tBuilds and runs models based on user inputs set in src/integrator/run_config.toml\n'
        '\tMode argument determines which models are run and how they are integrated and solved\n'
        '\tUniversal and module-specific options contained within run_config.toml\n'
        '\tUser can specify regions, time periods, solver options, and mode in run_config\n'
        '\tUsers can also specify the mode via command line argument or run_config.toml',
    )
    parser.add_argument(
        '--mode',
        choices=['unified-combo', 'gs-combo', 'standalone', 'elec', 'h2', 'residential'],
        dest='op_mode',
        help='The mode to run:\n\n'
        'unified-combo:  run unified optimization method, iteratively solves modules turned on in the run_congif file\n'
        'gs-combo:  run gauss-seidel method, iteratively solves modules turned on in the run_congif file\n'
        'standalone: runs in standalone the modules that are turned on in the run_config file\n'
        'elec:  run the electricity module standalone\n'
        'h2:  run the hydrogen module standalone\n'
        'residential: run the residential modulel standalone, solves updated load based on new given prices\n\n'
        'Mode can be set either via --mode command or in run_config.toml.\n'
        'If no --mode option is provided, default_mode in run_config.toml is used.',
    )
    parser.add_argument('--debug', action='store_true', help='set logging level to DEBUG')

    # parsing arguments
    args = parser.parse_args()

    return args


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
    logger = setup_logger(settings.OUTPUT_ROOT)
    logger = logging.getLogger(__name__)
    logger.info('Starting Logging')
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('Logging level set to DEBUG')
    logger.info(f'Model running in: {settings.selected_mode} mode')

    # MAIN - Log settings
    logger.info(f'Regions: {settings.regions}')
    logger.info(f'Years: {settings.years}')
    settings.cw_temporal.to_csv(Path(settings.OUTPUT_ROOT / 'cw_temporal.csv'), index=False)

    # MAIN - Run the cases you want to run based on the mode and settings you pass
    runner = globals()[settings.run_method]
    runner(settings)

    # MAIN - print the output directory once run is finished
    path_parts = os.path.normpath(settings.OUTPUT_ROOT).split(os.sep)
    output_name = os.path.join(path_parts[-2], path_parts[-1])
    print(f'Results located in: {output_name}')
    os.remove('output_root.txt')


if __name__ == '__main__':
    main()
