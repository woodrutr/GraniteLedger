"""main.py for Bluesky Prototype electricity_data_pipeline"""

# Import packages
import logging
from pathlib import Path
import tomllib
import os
import argparse
import types
import glob
import shutil

# Import python modules
from graniteledger.definitions import PROJECT_ROOT
from settings.config_setup import Config_settings
import src.runner as elec_methods


def get_args(execution_options: list | None = None):
    """parses arguments

    Parameters
    ----------
    execution_options : list | None, optional
        name of data execution function to run, by default None (will run all)

    Returns
    -------
    args: Namespace
        Contains arguments pass to main.py executable
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='description:\n'
        '\tBuilds and generates model inputs based on user inputs set in data_config.toml\n',
    )
    parser.add_argument(
        '--output',
        dest='output_dir',
        help='The location where the outputs are written.\n'
        'The output directory can be set either via --output command or in data_config.toml.\n'
        'If no --output option is provided, output_dir in data_config.toml is used.\n',
    )
    parser.add_argument(
        '--data',
        choices=execution_options,
        dest='data',
        help='Option to run just one of the executions instead of editing the data_config.toml.\n'
        'If no --data option is provided, default settings in data_config.toml are used.\n'
        'As more options are added to data_config.toml, arg options will automatically update.\n',
        # f'The data options include: {execution_options}\n',
    )
    parser.add_argument('--debug', action='store_true', help='Sets logging level to DEBUG.')

    # parsing arguments
    args = parser.parse_args()

    return args


# Logger Setup
def setup_logger(settings):
    """initiates logging

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings
    """
    # set up root logger
    log_path = Path(PROJECT_ROOT, 'log')
    if not Path.is_dir(log_path):
        Path.mkdir(log_path)

    # set logger level
    if settings.args.debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    # set logger configs
    logging.basicConfig(
        filename=Path(PROJECT_ROOT, 'log', 'run.log'),
        encoding='utf-8',
        filemode='w',
        format='%(asctime)s | %(name)s | %(levelname)s :: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=loglevel,
    )
    logging.getLogger('pyomo').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


# Main Execution
def main(config_file: str | None = None):
    """Executes main project

    Parameters
    ----------
    config_file : str | None, optional
        file which contains configuration settings, by default None
    """

    # MAIN - Get configuation settings
    if not config_file:
        config_file = 'data_config.toml'
    config_path = Path(PROJECT_ROOT, 'settings', config_file)
    with open(config_path, 'rb') as src:
        config = tomllib.load(src)

    # MAIN - Parse the args to get selected mode if one is provided
    execution_options = list(config['executions'].keys())
    args = get_args(execution_options)

    # MAIN - Establish the settings
    settings = Config_settings(config_path, args=args)
    print(f'writing outputs to {settings.OUTPUT_ROOT}')

    # MAIN - Establish the logger
    logger = setup_logger(settings)
    logger = logging.getLogger(__name__)
    logger.info('Starting Logging')
    logger.debug('Logging level set to DEBUG')

    # MAIN - Run functions in electricity runner
    for execution in settings.executions:
        method = getattr(elec_methods, execution)
        logger.info(f'running {execution}')
        method(settings)


if __name__ == '__main__':
    main()
