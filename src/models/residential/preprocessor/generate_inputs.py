"""
This file contains the options to re-create the input files. It creates:
 - Load.csv: electricity demand for all model years (used in residential and electricity)
 - BaseElecPrice.csv: electricity prices for initial model year (used in residential only)
Uncomment out the functions at the end of this file in the "if __name__ == '__main__'" statement
in order to generate new load or base electricity prices.

"""

from pathlib import Path
import pandas as pd
import os

# Set directories
# TODO: import structure is to support running locally, will consider changing
PROJECT_ROOT = Path(__file__).parents[4]
os.chdir(PROJECT_ROOT)
data_root = Path(PROJECT_ROOT, 'src/models/residential/input')

# Import scripts
from src.integrator import config_setup
from src.integrator.utilities import make_dir, setup_logger, get_output_root
from src.integrator.runner import run_elec_solo


def scale_load():
    """Reads in BaseLoad.csv (load for all regions/hours for first year)
    and LoadScalar.csv (a multiplier for all model years). Merges the
    data and multiplies the load by the scalar to generate new load
    estimates for all model years.

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe that contains load for all regions/years/hours
    """
    # combine first year baseload data with scalar data for all years
    baseload = pd.read_csv(data_root / 'BaseLoad.csv')
    scalar = pd.read_csv(data_root / 'preprocessor_inputs' / 'LoadScalar.csv')
    df = pd.merge(scalar, baseload, how='cross')

    # scale load in each year by scalar
    df['Load'] = round(df['Load'] * df['scalar'], 3)
    df = df.drop(columns=['scalar'])

    # reorder columns
    df = df[['r', 'y', 'hr', 'Load']]

    return df


def scale_load_with_enduses():
    """Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv
    (the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier
    for all model years by enduse category). Merges the data and multiplies the load by the
    adjusted enduse scalar and then sums up to new load estimates for all model years.

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe that contains load for all regions/years/hours
    """
    # share of total base load that is assigned to each enduse cat
    eu = pd.read_csv(Path(data_root / 'preprocessor_inputs' / 'EnduseBaseShares.csv'))

    # annual incremental growth (percent of eu baseload)
    eus = pd.read_csv(Path(data_root / 'preprocessor_inputs' / 'EnduseScalar.csv'))

    # converts the annual increment to percent of total baseload
    eu = pd.merge(eu, eus, how='left', on='enduse_cat')
    eu['increment_annual'] = eu['increment_annual'] * eu['base_year_share']
    eu = eu.drop(columns=['base_year_share'])

    # baseload total
    load = pd.read_csv(Path(data_root / 'BaseLoad.csv'))
    bla = load.groupby(by=['r'], as_index=False).sum().drop(columns=['hr'])

    # converts the annual increment to mwh
    eu = pd.merge(eu, bla, how='cross')
    eu['increment_annual'] = eu['increment_annual'] * eu['Load']
    eu = eu.drop(columns=['Load'])

    # percent of enduse load for each hour
    euh = pd.read_csv(Path(data_root / 'preprocessor_inputs' / 'enduse_shapes.csv'))

    # converts the annual increment to an hourly increment
    eu = pd.merge(eu, euh, how='left', on=['enduse_cat'])
    eu['increment'] = eu['increment_annual'] * eu['share']
    eu = eu.drop(columns=['increment_annual', 'share'])
    eu = eu.groupby(by=['y', 'r', 'hr'], as_index=False).sum().drop(columns=['enduse_cat'])

    # creates future load
    load = pd.merge(load, eu, how='left', on=['r', 'hr'])
    load['Load'] = load['Load'] + load['increment']
    load = load[['r', 'y', 'hr', 'Load']]

    return load


def compare_load_method_results():
    """runs the two methods for developing future load estimates and then creates to review files.
    review1 sums the hourly data up by region and year. review2 writes out the hourly data for the
    final model year for all regions. The data is written out to csvs for user inspection.
    """
    df1 = scale_load()
    df2 = scale_load_with_enduses()
    df2 = df2.rename(columns={'Load': 'Load2'})
    df = pd.merge(df1, df2, on=['r', 'y', 'hr'])
    df3 = df.groupby(by=['r', 'y'], as_index=False).sum().drop(columns=['hr'])
    df3.to_csv(Path(data_root / 'preprocessor_inputs' / 'review1.csv'), index=False)
    df4 = df[df['y'] == df['y'].max()]
    df4.to_csv(Path(data_root / 'preprocessor_inputs' / 'review2.csv'), index=False)


def base_price():
    """Runs the electricity model with base price configuration settings and then
    merges the electricity prices and temporal crosswalk data produced from the run
    to generate base year electricity prices.

    Returns
    -------
    pandas.core.frame.DataFrame
        dataframe that contains base year electricity prices for all regions/hours
    """
    OUTPUT_ROOT = get_output_root()
    make_dir(OUTPUT_ROOT)
    logger = setup_logger(OUTPUT_ROOT)

    # run electricity model with base price config settings
    baseprice_config_path = Path(data_root / 'preprocessor_inputs' / 'baseprice_config.toml')
    price_settings = config_setup.Config_settings(baseprice_config_path, test=True)
    run_elec_solo(price_settings)

    # grab electricity model output results
    cw_temporal = pd.read_csv(Path(OUTPUT_ROOT / 'cw_temporal.csv'))
    elec_price = pd.read_csv(Path(OUTPUT_ROOT / 'electricity' / 'prices' / 'elec_price.csv'))

    # keep only the electricity price data needed
    base_year = elec_price['y'].min()
    elec_price = elec_price[elec_price['y'] == base_year]
    elec_price = elec_price[['r', 'y', 'hr', 'price_wt']].rename(columns={'hr': 'Map_hr'})

    # crosswalk the electricity prices to all hours in the base year
    cw_temporal = cw_temporal[['hr', 'Map_hr']]
    elec_price = pd.merge(elec_price, cw_temporal, how='right', on=['Map_hr'])
    elec_price = elec_price.drop(columns=['Map_hr'])
    elec_price = elec_price[['r', 'y', 'hr', 'price_wt']]
    elec_price.sort_values(['r', 'y', 'hr'], inplace=True)

    return elec_price


if __name__ == '__main__':
    # Comment on/off each function as needed
    # scale_load().to_csv(data_root / 'Load.csv', index=False)
    # scale_load_with_enduses().to_csv(data_root / 'Load.csv', index=False)
    # compare_load_method_results()
    base_price().to_csv(data_root / 'BaseElecPrice.csv', index=False)
    pass
