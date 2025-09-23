"""main.py for Bluesky Prototype"""

# Import packages
import logging
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
        raise ModuleNotFoundError(
            'Python 3.11+ or the tomli package is required to read TOML configuration files.'
        ) from exc
import os
import argparse
import types
import glob
import shutil

# Import python modules
from main.definitions import PROJECT_ROOT
from config_setup import Config_settings
from src.runner import residential_runner


def get_args():
    """gets arguments

    Returns
    -------
    args: Namespace
        Contains arguments pass to main.py executable
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='description:\n'
        '\tBuilds and generates model inputs based on user inputs set in data_config.toml\n'
        '\tOutput argument determines where input files are written out to\n'
        '\tUniversal and module-specific options contained within data_config.toml\n'
        '\tUsers can specify the output directory via command line argument or in data_config.toml',
    )
    parser.add_argument(
        '--output',
        dest='output_dir',
        help='The location where the outputs are written.\n'
        'The output directory can be set either via --output command or in data_config.toml.\n'
        'If no --output option is provided, output_dir in data_config.toml is used.',
    )
    parser.add_argument('--debug', action='store_true', help='set logging level to DEBUG')

    # parsing arguments
    args = parser.parse_args()

    return args


# Logger Setup
def setup_logger():
    """initiates logging"""
    # set up root logger
    log_path = Path(PROJECT_ROOT, 'log')
    if not Path.is_dir(log_path):
        Path.mkdir(log_path)
    logging.basicConfig(
        filename=Path(PROJECT_ROOT, 'log', 'run.log'),
        encoding='utf-8',
        filemode='w',
        format='%(asctime)s | %(name)s | %(levelname)s :: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.INFO,
    )
    logging.getLogger('pyomo').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


# Main Execution
def main(settings: Config_settings | None = None):
    """
    Executes main project
    """
    # MAIN - Parse the args to get selected mode if one is provided
    args = get_args()

    # MAIN - Establish the settings
    if not settings:
        settings = Config_settings(args=args)
    print(f'writing outputs to {settings.OUTPUT_ROOT}')

    # MAIN - Establish the logger
    logger = setup_logger()
    logger = logging.getLogger(__name__)
    logger.info('Starting Logging')
    if args.debug:
        logger.setLevel(logging.DEBUG)
    logger.debug('Logging level set to DEBUG')

    # Run Residential Runner
    residential_runner(settings)

    # Close the logger
    logger.info('Ending Logging')
    logging.shutdown()


if __name__ == '__main__':
    main()
