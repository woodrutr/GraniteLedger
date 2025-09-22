# Import python packages
import os as os
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from logging import getLogger
import logging
import requests
import zipfile
import h5py

# Import python modules
from graniteledger.definitions import PROJECT_ROOT

# Establish logger
logger = getLogger(__name__)


#################################################################################################
# Functions


def calc_pop_cw(cw: pd.DataFrame, settings):
    """maps user-defined region to county and calculates fraction of user-defined region population in each county

    Parameters
    ----------
    cw : pd.DataFrame
        crosswalk of county and user-defined region
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame which contains user-defined region/county mapping and the fraction of the user-defined region population in each county
    """

    # calc_pop_cw -- read in county population for population_year and combine w/ regional crosswalk
    population_year = settings.population_year
    pop = pd.read_csv(PROJECT_ROOT / 'input/County_Population_2010-2022.csv')
    pop = pop[pop['year'] == population_year]
    pop = pop.drop(columns=['year'])
    pop = pd.merge(cw, pop, how='right', on=['FIPS_cnty']).dropna().reset_index(drop=True)

    # calc_pop_cw -- account for multiple regions being assigned to one county, split up pop equally
    cnty_cnt = pop[['FIPS_cnty', 'population']].copy().rename(columns={'population': 'count'})
    cnty_cnt = cnty_cnt.groupby(by=['FIPS_cnty'], as_index=False).count()
    pop = pd.merge(pop, cnty_cnt, how='left', on=['FIPS_cnty'])
    pop['population'] = pop['population'] / pop['count']
    pop = pop.drop(columns=['count'])
    logger.debug(f'calc_pop_cw: {pop.columns}')

    # calc_pop_cw -- get cw id column name for groupby
    cw_id = list(cw.columns)[-1]
    logger.debug(f'calc_pop_cw: {cw_id}')

    # calc_pop_cw -- calculate the regional population
    reg_pop = pop[[cw_id, 'population']].groupby(by=[cw_id], as_index=False).sum()
    reg_pop = reg_pop.rename(columns={'population': 'reg_pop'})
    logger.debug(f'calc_pop_cw: {reg_pop.columns}')

    # calc_pop_cw -- calculate regional share
    pop = pd.merge(pop, reg_pop, how='left', on=[cw_id])
    pop['pop_share'] = pop['population'] / pop['reg_pop']
    pop = pop.drop(columns=['population', 'reg_pop'])

    return pop


def get_names(df: pd.DataFrame, pop: pd.DataFrame):
    """pulls column names: region name, data value name, and groupby names

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing data by user-defined region
    pop : pd.DataFrame
        data frame containing population by county

    Returns
    -------
    tuple
       where [0] is type str: user-defined region column name
       where [1] is type str: data column name
       where [2] is type list (or str): column names to group by
    """
    # get col names
    cw_id = [item for item in list(pop.columns) if item not in ['FIPS_cnty', 'pop_share']][0]
    data_id = list(df.columns)[-1]
    groupby_cols = [item for item in list(df.columns) if item not in [cw_id, data_id]]
    logger.debug(f'get_names: {cw_id}')
    logger.debug(f'get_names: {data_id}')

    return cw_id, data_id, groupby_cols


def sum_data_cnty(df: pd.DataFrame, pop: pd.DataFrame):
    """calculate population-weighted data by county

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing data by user-defined region
    pop : pd.DataFrame
        data frame containing population by county

    Returns
    -------
    pd.DataFrame
        data frame containing population-weighted data by county
    """

    # sum_data_cnty -- get col names
    cw_id, data_id, groupby_cols = get_names(df, pop)

    # sum_data_cnty -- merge data with county pop and regional pop data, determine county share
    df = pd.merge(pop, df, how='right', on=[cw_id])
    df[data_id] = df[data_id] * df['pop_share']

    # sum_data_cnty -- calculate total data for each county
    df = df.drop(columns=[cw_id, 'pop_share'])
    df = df.groupby(by=['FIPS_cnty'] + groupby_cols, as_index=False).sum(data_id)
    df['FIPS_cnty'] = df['FIPS_cnty'].astype(int)

    return df


def avg_data(df: pd.DataFrame, pop: pd.DataFrame):
    """calculate population-averaged value over user-specified region

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing data by EMM region
    pop : pd.DataFrame
        data frame containing population by EMM region

    Returns
    -------
    pd.DataFrame
        data frame containing an average value over user-specified region
    """
    # avg_data -- get col names
    cw_id, data_id, groupby_cols = get_names(df, pop)

    # avg_data -- merge data with county pop and regional pop data
    df = pd.merge(pop, df, how='right', on=[cw_id])
    cw = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    df = pd.merge(df, cw, how='left', on=['FIPS_cnty'])

    # avg_data -- determine share of data that goes to county
    df = df.drop(columns=['FIPS_cnty', cw_id])
    df = df.groupby(by=['region'] + groupby_cols, as_index=False).sum()
    df[data_id] = df[data_id] / df['pop_share']
    df = df.drop(columns=['pop_share'])

    return df


#################################################################################################
# Spatial Setup for SupplyCurve


def read_860m(settings):
    """reads EIA860-m data from downloaded xlsx and merges operating with planned tabs

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame containing EIA860-m data with operating with planned information
    """
    # read_860m -- establish 860m path
    excel_name = settings.EIA860m_excel_name
    excel_path = Path(PROJECT_ROOT / 'input' / f'{excel_name}.xlsx')

    # read_860m -- read in the operating tab from the excel file
    op = pd.read_excel(excel_path, sheet_name='Operating', skiprows=2)
    keep_cols = [
        'Plant ID',
        'Generator ID',
        'Plant State',
        'County',
        'Net Summer Capacity (MW)',
        'Technology',
        'Status',
        'Operating Year',
    ]
    op = op[keep_cols + ['Planned Retirement Year']]

    # read_860m -- read in the planned tab from the excel file
    pl = pd.read_excel(excel_path, sheet_name='Planned', skiprows=2)
    pl = pl.rename(columns={'Planned Operation Year': 'Operating Year'})
    pl = pl[keep_cols]

    # read_860m -- join the operating and planned datasets
    raw = pd.concat([op, pl])

    return raw


def create_supplycurve_cnty(df: pd.DataFrame, settings):
    """creates data frame containing supply curves by county using real existing and planned EIA860 capacity data.

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing EIA860m capacity data for existing and planned builds
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame containing supply curves by county
    """
    # create_supplycurve_cnty -- create an ID column
    df = df.dropna(subset=['Plant ID']).copy()
    df['Plant ID'] = df['Plant ID'].astype(int)
    df['ID'] = df['Plant ID'].astype(str) + '_' + df['Generator ID'].astype(str)

    # create_supplycurve_cnty -- add model technology codes
    cwt = pd.read_csv(PROJECT_ROOT / 'input/cw_tech.csv')
    df = pd.merge(df, cwt, how='left', on=['Technology'])

    # create_supplycurve_cnty -- clean up county IDs, remove counties we don't want to keep
    cwc = pd.read_csv(PROJECT_ROOT / 'input/cw_county.csv')
    cwc = cwc[['State', 'County', 'FIPS_cnty']].rename(columns={'State': 'Plant State'})
    df = pd.merge(df, cwc, how='left', on=['Plant State', 'County'])
    df = df.dropna(subset=['FIPS_cnty'])
    df['FIPS_cnty'] = df['FIPS_cnty'].astype(int)

    # create_supplycurve_cnty -- remove rows with statuses we don't want to keep
    cws = pd.read_csv(PROJECT_ROOT / 'input/cw_status.csv')
    df = pd.merge(df, cws, how='left', on=['Status'])
    df = df[df['Keep'] == 1]

    # create_supplycurve_cnty -- clean up columns
    drop = ['Plant ID', 'Generator ID', 'Technology', 'Plant State', 'County', 'Status', 'Keep']
    rename = {
        'Net Summer Capacity (MW)': 'Capacity',
        'Operating Year': 'year',
        'Planned Retirement Year': 'Ret_Year',
    }
    df = df.drop(columns=drop).rename(columns=rename)

    # create_supplycurve_cnty -- keep only online years relavent to the model
    df.loc[df['year'] < settings.first_year, 'year'] = settings.first_year
    df['year'] = df['year'].astype(int)
    df.loc[df['year'] > settings.last_year, 'Drop'] = 1
    df = df[df['Drop'] != 1].drop(columns=['Drop'])

    # create_supplycurve_cnty -- keep only retirement years relavent to the model
    df.loc[df['Ret_Year'] == ' ', 'Ret_Year'] = 9999
    df.loc[df['Ret_Year'].isna(), 'Ret_Year'] = 9999
    df.loc[df['Ret_Year'] > settings.last_year, 'Ret_Year'] = 9999
    df['Ret_Year'] = df['Ret_Year'].astype(int)

    # create_supplycurve_cnty -- remove rows with missing capacity data
    df.loc[df['Capacity'] == ' ', 'Capacity'] = 0
    df.loc[df['Capacity'].isna(), 'Capacity'] = 0
    df['Capacity'] = df['Capacity'].astype(float) / 1000

    # create_supplycurve_cnty -- group data by technology/county/year/retirement year
    df = df.drop(columns=['ID'])
    df = df.groupby(by=['tech', 'FIPS_cnty', 'year', 'Ret_Year'], as_index=False).sum()

    # create_supplycurve_cnty -- create full index to merge to
    indx = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv').drop(columns=['region'])
    cwt = pd.read_csv(PROJECT_ROOT / 'input/cw_tech.csv')
    indx = pd.merge(indx, pd.DataFrame(cwt['tech'].unique(), columns=['tech']), how='cross')
    indx = pd.merge(indx, pd.DataFrame(settings.year_range, columns=['year']), how='cross')

    # create_supplycurve_cnty -- add online capacity for each county/technology/year
    online = df.drop(columns=['Ret_Year'])
    online = online.groupby(by=['tech', 'FIPS_cnty', 'year'], as_index=False).sum()
    frame = pd.merge(indx, online, how='left', on=['FIPS_cnty', 'tech', 'year'])

    # create_supplycurve_cnty -- add planned retirement capacity for each county/technology/year
    offline = df.drop(columns=['year'])
    offline = offline.groupby(by=['tech', 'FIPS_cnty', 'Ret_Year'], as_index=False).sum()
    offline = offline.rename(columns={'Capacity': 'Ret_Capacity', 'Ret_Year': 'year'})
    frame = pd.merge(frame, offline, how='left', on=['FIPS_cnty', 'tech', 'year'])
    frame = frame.fillna(0)

    # create_supplycurve_cnty -- calculate the cumulative sum of the capacities minus retirements
    frame = frame.set_index(['tech', 'FIPS_cnty', 'year'])
    frame['Cap_Cum'] = frame.groupby(by=['tech', 'FIPS_cnty'])['Capacity'].cumsum()
    frame['Ret_Cap_Cum'] = frame.groupby(by=['tech', 'FIPS_cnty'])['Ret_Capacity'].cumsum()
    frame = frame.reset_index()
    frame = frame[frame['Cap_Cum'] != 0]
    frame['SupplyCurve'] = frame['Cap_Cum'] - frame['Ret_Cap_Cum']
    frame = frame.drop(columns=['Capacity', 'Ret_Capacity', 'Cap_Cum', 'Ret_Cap_Cum'])

    # create_supplycurve_cnty -- combine data with steps
    cwst = pd.read_csv(PROJECT_ROOT / 'input/cw_steps.csv')
    frame = pd.merge(frame, cwst, how='right', on=['tech']).fillna(0)
    frame['SupplyCurve'] = (frame['SupplyCurve'] / frame['count']).apply(lambda x: round(x, 2))
    frame = frame.drop(columns=['count'])
    frame = frame[['FIPS_cnty', 'tech', 'step', 'year', 'SupplyCurve']]
    frame = frame[frame['year'] > 0]

    # create_supplycurve_cnty -- add DGPV capacity
    cw = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    dg = pd.read_csv(PROJECT_ROOT / 'input/dgpv_cap.csv')
    pop = calc_pop_cw(cw, settings)
    dg = sum_data_cnty(dg, pop)
    frame = pd.concat([frame, dg])

    return frame


def create_supplycurve_r(frame: pd.DataFrame, settings):
    """aggregates supply curves from county to user-specified regional level

    Parameters
    ----------
    frame : pd.DataFrame
        data frame containing supply curves at the county-level
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame containing supply curves at user-specified regional level
    """
    # agg the data up to the model region level
    cwr = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    frame = pd.merge(frame, cwr, how='right', on=['FIPS_cnty'])
    frame = frame.drop(columns=['FIPS_cnty'])
    frame = frame.groupby(by=['tech', 'region', 'year', 'step'], as_index=False).sum()
    frame = frame[['region', 'tech', 'step', 'year', 'SupplyCurve']]

    # create full index to merge to
    indx = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv').drop(columns=['FIPS_cnty'])
    indx = indx.drop_duplicates()
    cwst = pd.read_csv(PROJECT_ROOT / 'input/cw_steps.csv').drop(columns=['count'])
    new_row = pd.DataFrame({'tech': [15], 'step': [2]})
    cwst = pd.concat([cwst, new_row], ignore_index=True)
    indx = pd.merge(indx, cwst, how='cross')
    indx = pd.merge(indx, pd.DataFrame(settings.year_range, columns=['year']), how='cross')

    frame = pd.merge(indx, frame, on=['region', 'tech', 'step', 'year'], how='left').fillna(0)
    return frame


def supplycurve(settings):
    """create supply curves at user-specified regional level and write out to csv

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings
    """
    df = read_860m(settings)
    df = create_supplycurve_cnty(df, settings)
    # supplycurve -- option to view the supply curve data at the county level if uncomment this line
    # df.to_csv(PROJECT_ROOT / 'output/SupplyCurve_cnty.csv', index=False)
    create_supplycurve_r(df, settings).to_csv(PROJECT_ROOT / 'output/SupplyCurve.csv', index=False)


#################################################################################################
# Spatial Setup for VRE CF Data


# TODO: Future update to grab wind and solar data from weather project rather than NREL files
def readin_sql(settings):
    """Load table from SQLite DB

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        weather/solar data read in from sql db
    """
    db_dir = 'PATH-TO-DB/noaa_db.db'
    engine = create_engine('sqlite:///' + db_dir)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base = automap_base()
    Base.prepare(autoload_with=engine)
    metadata = MetaData()
    metadata.reflect(engine)
    counties_noaa = Base.classes.counties_noaa

    df = session.query(counties_noaa).filter(counties_noaa.year == settings.population_year).all()
    df = [[t for t in s.__dict__.items()] for s in df]
    df = [{k: v for k, v in t} for t in df]
    df = pd.DataFrame(df)
    df = df.drop(columns=['sa_instance_state'])

    session.close()

    return df


def download_file(url: os.PathLike | str, filename: str):
    """download file from url to local inputs

    Parameters
    ----------
    url : os.PathLike | str
        url path to file
    filename : str
        name of file to download to locally
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_loc = Path(PROJECT_ROOT, 'input', filename)
        with open(file_loc, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        logger.info(f'Successfully downloaded {filename}')
    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')


def wind_cnty_cf():
    """get onshore wind capacity factor data by county

    Returns
    -------
    pd.DataFrame
        data frame containing onshore wind capacity factor by county
    """

    # Download NREL data files
    file_list = [
        'reference_access_2030_moderate_115hh_170rd_rep-profiles_2012%20(1).h5',
        'reference_access_2030_moderate_115hh_170rd_supply-curve%20(1).csv',
    ]
    for filename in file_list:
        if os.path.exists(PROJECT_ROOT / 'input' / filename):
            pass
        else:
            url = 'https://data.openei.org/files/6119/' + filename
            download_file(url, filename)

    # Process NREL resource profiles
    filename = file_list[0]
    file = Path(PROJECT_ROOT / 'input', filename)
    with h5py.File(file, 'r') as fh:
        for dset in fh.keys():
            logger.debug(f'{dset}: {fh[dset].shape}')

    # Creating a dataframe from the h5 data ID = rows, hours = columns
    # time = pd.DataFrame(np.array(h5py.File(file)['time_index']))
    id = pd.DataFrame(np.array(h5py.File(file)['meta']))
    data = pd.DataFrame(np.array(h5py.File(file)['rep_profiles_0'])).transpose()

    # dropping columns not needed
    id = id[['sc_point_gid']]
    data = pd.concat([id, data], axis=1)

    # getting median county sc_point_gid id
    nrel_wsc = file_list[1]
    wsc_file = Path(PROJECT_ROOT / 'input', nrel_wsc)
    wsc = pd.read_csv(wsc_file)
    wsc = wsc[['sc_point_gid', 'cnty_fips', 'county', 'state', 'mean_cf']]

    def median_with_id(group):
        """Function to get the row corresponding to the median data value

        Parameters
        ----------
        group : pd.DataFrame
            data frame containing capacity factor data by location id

        Returns
        -------
        pd.Series
            series with location id and median cf closest to the median capacity factor in a county
        """
        median_value = group['mean_cf'].median()
        closest_idx = (
            (group['mean_cf'] - median_value).abs().idxmin()
        )  # Find index of closest median
        return pd.Series(
            {'sc_point_gid': group.loc[closest_idx, 'sc_point_gid'], 'mean_cf': median_value}
        )

    # Group by county and apply function
    df_median = (
        wsc.groupby('cnty_fips').apply(median_with_id).reset_index().drop(columns=['mean_cf'])
    )

    # Merge profile dataset to county list and melt
    df = pd.merge(df_median, data, how='left', on=['sc_point_gid'])
    df = df.rename(columns={'cnty_fips': 'FIPS_cnty'}).drop(columns=['sc_point_gid'])
    df = pd.melt(
        df, id_vars=['FIPS_cnty'], var_name='hour', value_name='CapFactorVRE', ignore_index=True
    )

    # Clean up data
    df['hour'] = df['hour'] + 1
    df['CapFactorVRE'] = df['CapFactorVRE'] / 1000
    df['CapFactorVRE'] = df['CapFactorVRE'].round(4)
    df['tech'] = 14
    df['step'] = 1
    logger.debug(f'wind_cnty_cf: {df.columns}')

    return df


def windoff_cnty_cf():
    """get offshore wind capacity factor data by county

    Returns
    -------
    pd.DataFrame
        data frame containing offshore wind capacity factor data by county
    """
    df = pd.read_csv(PROJECT_ROOT / 'input/CapFactorVRE_wnoff.csv')
    df = df[df['tech'] == 13]
    wf_cnty = pd.read_csv(PROJECT_ROOT / 'input/cw_wnoff.csv')
    logger.debug(f'windoff_cnty_cf: FIPS_cnty: {wf_cnty.dtypes}')
    cwc = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    logger.debug(f'windoff_cnty_cf: FIPS_cnty: {cwc.dtypes}')
    wf_cnty = pd.merge(wf_cnty, cwc, how='left', on=['FIPS_cnty']).dropna()
    df = pd.merge(wf_cnty, df, how='right', on=['region']).drop(columns=['region'])

    return df


def solar_cnty_cf():
    """get solar capacity factor data by county

    Returns
    -------
    pd.DataFrame
        data frame which contains solar capacity factor data by county
    """
    cf = pd.read_csv(PROJECT_ROOT / 'input/pvcf.csv')
    cwpv = pd.read_csv(PROJECT_ROOT / 'input/cw_pv.csv')
    logger.debug(f'solar_cnty_cf: {cwpv.shape}')
    df = pd.merge(cwpv, cf, on=['Solar_RE_GID'], how='left').drop(columns=['Solar_RE_GID'])
    logger.debug(f'solar_cnty_cf: {df.shape}')
    df = pd.melt(df, id_vars=['FIPS_cnty']).rename(
        columns={'variable': 'hour', 'value': 'CapFactorVRE'}
    )
    df['tech'] = 15
    df['step'] = 1
    df2 = df.copy()
    df2['step'] = 2
    df = pd.concat([df, df2])
    logger.debug(f'solar_cnty_cf: {df.columns}')

    return df


def vre_spatial(df: pd.DataFrame):
    """aggregates VRE capacity factor by user-specified region (takes mean of counties)

    Parameters
    ----------
    df : pd.DataFrame
        data frame with county-level VRE capacity factor

    Returns
    -------
    pd.DataFrame
        data frame containing VRE capacity factor by user-specified region
    """
    cw = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    df = pd.merge(df, cw, on=['FIPS_cnty'], how='right').drop(columns=['FIPS_cnty'])
    df = df.groupby(by=['tech', 'region', 'step', 'hour'], as_index=False).mean('CapFactorVRE')
    logger.debug(f'vre_spatial: {df.columns}')

    return df


def vre_capfactor(settings):
    """get VRE capacity factor for onshore wind, offshore wind, and solar, and write out to csv

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings
    """
    logger.info('vre_spatial(windoff_cnty_cf())')
    df = vre_spatial(windoff_cnty_cf())
    logger.info('vre_spatial(wind_cnty_cf())')
    wf = vre_spatial(wind_cnty_cf())
    logger.info('vre_spatial(solar_cnty_cf())')
    pv = vre_spatial(solar_cnty_cf())
    df = pd.concat([df, wf, pv])
    df['CapFactorVRE'] = df['CapFactorVRE'].round(6)
    df.to_csv(PROJECT_ROOT / 'output/CapFactorVRE.csv', index=False)


#################################################################################################
# Spatial Setup for Tranmission data


def TranLimitCapInt(settings):
    """maps international transmission capacity from county to user-specified region

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame containing international transmission capacity by user-specified region
    """
    cw = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    df = pd.read_csv(PROJECT_ROOT / 'input/TranLineLimitCan_fips.csv')
    df = pd.merge(cw, df, how='right', on=['FIPS_cnty']).drop(columns=['FIPS_cnty'])
    df = df.groupby(by=['region', 'region1'], as_index=False).sum()

    # TODO: Replace these with set data
    df = pd.merge(df, pd.DataFrame(list(settings.year_range), columns=['year']), how='cross')
    df = pd.merge(df, pd.DataFrame(list(range(1, 5)), columns=['season']), how='cross')

    last_column = df.pop('TranLimitCapInt')
    df.insert(len(df.columns), 'TranLimitCapInt', last_column)
    logger.debug(f'TranLimitCapInt: {df.columns}')

    return df


def TranCostInt(settings):
    """maps international transmission cost from county to user-specified region

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings

    Returns
    -------
    pd.DataFrame
        data frame containing international transmission cost by user-specified region
    """

    df = pd.read_csv(PROJECT_ROOT / 'input/TranCostCan_fips.csv')
    cw = pd.read_csv(PROJECT_ROOT / 'input/cw_r.csv')
    df = pd.merge(cw, df, how='right', on=['FIPS_cnty']).drop(columns=['FIPS_cnty'])
    df = df.groupby(by=['region', 'region1', 'step'], as_index=False).mean()

    # TODO: Replace these with set data
    df = pd.merge(df, pd.DataFrame(list(settings.year_range), columns=['year']), how='cross')

    last_column = df.pop('TranCostInt')
    df.insert(len(df.columns), 'TranCostInt', last_column)
    logger.debug(f'TranCostInt: {df.columns}')

    return df


def transmission(settings):
    """collects and writes out csv's for international transmission limit and cost by user-specified region

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings
    """
    TranLimitCapInt(settings).to_csv(PROJECT_ROOT / 'output/TranLimitCapInt.csv', index=False)
    TranCostInt(settings).to_csv(PROJECT_ROOT / 'output/TranCostInt.csv', index=False)


#################################################################################################
# Spatial Setup for other data


# TODO: switch this to sql db
def readin_csvs(all_frames: dict):
    """reads in csvs for all other data, creating dict of all data frames

    Parameters
    ----------
    all_frames : dict
        empty initialized dict

    Returns
    -------
    dict of ps.DataFrames
        dictionary with data frames containing all non-regional data
    """
    csv_dir = Path(PROJECT_ROOT, 'input', 'other_data')
    for filename in os.listdir(csv_dir):
        logger.debug(f'readin_csvs: {filename[:-4]}')
        f = os.path.join(csv_dir, filename)
        if os.path.isfile(f):
            all_frames[filename[:-4]] = pd.read_csv(f)
            # print(filename[:-4],all_frames[filename[:-4]].columns)

    return all_frames


def other_data(settings):
    """read in, find average over user-specified regions, and write out to csvs: all other data

    Parameters
    ----------
    settings : settings.config_setup.Config_settings
        input settings
    """
    all_frames = {}
    all_frames = readin_csvs(all_frames)
    other_cw = pd.read_csv(Path(PROJECT_ROOT, 'input', 'emm_county.csv'))
    pop = calc_pop_cw(other_cw, settings)

    for key in all_frames.keys():
        logger.info(f'other_data: {key}')
        logger.debug(f'other_data: {key}: {all_frames[key].columns}')
        avg_data(all_frames[key], pop).to_csv(Path(PROJECT_ROOT, f'output/{key}.csv'), index=False)

        pass
