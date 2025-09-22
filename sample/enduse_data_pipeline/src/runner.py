"""This file is runs scripts in this directory"""

###################################################################################################
# Setup

# Import pacakges
from pathlib import Path
import sys as sys
from logging import getLogger
import shutil
import os

# Import python modules
from graniteledger.definitions import PROJECT_ROOT
from config_setup import Config_settings
from graniteledger.src.enduse_db import main as enduse_db
from graniteledger.src.enduse_demand import main as enduse_demand

# Establish logger
logger = getLogger(__name__)


# Main Execution
def residential_runner(settings: Config_settings | None = None):
    """sets up running the enduse_db function and the enduse_demand based on settings

    Parameters
    ----------
    settings : Config_settings
        project settings

    Raises
    ------
    ValueError
        if stock_database is missing, change settings
    """
    # if none, create a settings object
    if not settings:
        settings = Config_settings()

    # Establish output directory
    output_dir = settings.OUTPUT_ROOT

    # if you need to build the stock_db
    if settings.build_stock_db is True:
        if settings.test_build_data is False:
            print('Warning: You set test_build_data to False, which means full enduse profiles')
            print('   Just a heads up, this may take awhile... +1 hr... SORRY!')
            print('   Also requires 10 GB of storage. Check the run log for progress.')
        else:
            print('Warning: You set test_build_data to True, which means enduse profiles are just')
            print('   a sample of the whole data, set to False to reproduce current model inputs.')
            print('   Note, full run requires 10+ GB of local storage and takes 1+ hours to run.')
        enduse_db(settings)

    # Check that stock_db exists
    input_path = Path(PROJECT_ROOT / 'input')
    if settings.test_build_data is False:
        db_path = Path(input_path, 'stock_database.db')
        logger.info('using stock_database')
        if not os.path.exists(db_path):
            raise ValueError('stock_database missing, set build_stock_db to true')
        print('Expect to wait 30 more min to create enduse shapes... SORRY!')
    else:
        db_path = Path(input_path, 'stock_database_test.db')
        logger.info('using stock_database_test')

    # create enduse shapes
    enduse_shapes = enduse_demand(settings)
    test = settings.test_build_data
    if test is True:
        enduse_shapes.to_csv(Path(output_dir / 'EnduseShapes_test.csv'), index=False)
    else:
        enduse_shapes.to_csv(Path(output_dir / 'EnduseShapes.csv'), index=False)
