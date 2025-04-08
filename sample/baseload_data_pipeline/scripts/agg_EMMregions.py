# Import python packages
import os as os
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, MetaData, Table, select
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
import logging
import requests

# Read in snakemake inputs
# population_year = snakemake.params['pop_year']
population_year = 2022

# population_year = 2022

demand_data_dir = 'outputs/load-predict'
demand_data_filename = 'LoadShapesData_predict_monthly_ol.csv'

#################################################################################################
# Spatial Setup for Load Data


def read_in_load_data():
    """Read in csv input BA load data and reformat region names

    Returns
    -------
    pd.DataFrame
        df with input BA load data
    """
    # read_in_load_data -- add csv input files to all frames dictionary
    data_dir = demand_data_dir
    filename = demand_data_filename
    df = pd.read_csv(Path(data_dir, filename))
    df['Value'] = df['Value'] / 1000.0
    df = df.rename(columns={'PCA_SHORT_NAME': 'BASR_Code'})

    # read_in_load_data -- create region id for input data
    df[['BA_Code', 'BA_BASR_Code']] = df['BASR_Code'].str.split('_', expand=True)
    df['BA_BASR_Code'] = df['BA_BASR_Code'].fillna('NA')
    df['BA_BASR_Code'] = df['BA_Code'] + '-' + df['BA_BASR_Code'].astype(str)
    df = df.drop(columns=['BA_Code', 'BASR_Code'])
    first_column = df.pop('BA_BASR_Code')
    df.insert(0, 'BA_BASR_Code', first_column)

    return df


def calc_pop_cw(cw: pd.DataFrame):
    """maps BA region to county and calculates fraction of BA region population in each county

    Parameters
    ----------
    cw : pd.DataFrame
        crosswalk of county and BA region

    Returns
    -------
    pd.DataFrame
        data frame which contains BA region/county mapping and the fraction of the BA region population in each county
    """
    # calc_pop_cw -- read in county population for population_year and combine w/ regional crosswalk
    pop = pd.read_csv('outputs/census/County_Population.csv')
    pop = pop[pop['year'] == population_year]
    pop = pop.drop(columns=['year'])
    pop = pd.merge(cw, pop, how='right', on=['FIPS_cnty']).dropna().reset_index(drop=True)

    # calc_pop_cw -- account for multiple regions being assigned to one county, split up pop equally
    cnty_cnt = pop[['FIPS_cnty', 'population']].copy().rename(columns={'population': 'count'})
    cnty_cnt = cnty_cnt.groupby(by=['FIPS_cnty'], as_index=False).count()
    pop = pd.merge(pop, cnty_cnt, how='left', on=['FIPS_cnty'])
    pop['population'] = pop['population'] / pop['count']
    pop = pop.drop(columns=['count'])

    # calc_pop_cw -- get cw id column name for groupby
    cw_id = list(cw.columns)[-1]

    # calc_pop_cw -- calculate the regional population
    reg_pop = pop[[cw_id, 'population']].groupby(by=[cw_id], as_index=False).sum()
    reg_pop = reg_pop.rename(columns={'population': 'reg_pop'})

    # calc_pop_cw -- calculate regional share (pop_share = fraction of the BA region population in a given county)
    pop = pd.merge(pop, reg_pop, how='left', on=[cw_id])
    pop['pop_share'] = pop['population'] / pop['reg_pop']
    pop = pop.drop(columns=['population', 'reg_pop'])

    return pop


def get_names(df: pd.DataFrame, pop: pd.DataFrame):
    """pulls column names: region name, data value name, and groupby names

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing load by BA region
    pop : pd.DataFrame
        data frame containing population by county

    Returns
    -------
    tuple
       where [0] is type str: BA region column name (expect: BA_BASR_Code)
       where [1] is type str: data column name (expect: Value)
       where [2] is type list (or str): column names to group by (expect: ['Month','Day','Hour'])
    """
    # get col names
    cw_id = [item for item in list(pop.columns) if item not in ['FIPS_cnty', 'pop_share']][0]
    data_id = list(df.columns)[-1]
    groupby_cols = [item for item in list(df.columns) if item not in [cw_id, data_id]]

    return cw_id, data_id, groupby_cols


def sum_data_cnty(df: pd.DataFrame, pop: pd.DataFrame):
    """calculate population-weighted load by county

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing load by BA region
    pop : pd.DataFrame
        data frame containing population by county

    Returns
    -------
    pd.DataFrame
        data frame containing population-weighted load by county
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


def sum_data_r(df: pd.DataFrame, pop: pd.DataFrame):
    """sum load data from county-level to EMM-region-level

    Parameters
    ----------
    df : pd.DataFrame
        data frame containing load by county
    pop : pd.DataFrame
        data frame containing population by county (only used to pull columns)

    Returns
    -------
    pd.DataFrame
        data frame containing load by EMM region
    """
    # sum_data_r -- get col names
    cw_id, data_id, groupby_cols = get_names(df, pop)
    groupby_cols.remove('FIPS_cnty')

    # sum_data_r -- calculate total data for each region
    cw = pd.read_csv('inputs/maps/cw_r.csv')  # BA region/county crosswalk
    df = pd.merge(df, cw, how='left', on=['FIPS_cnty'])
    df = df.drop(columns=['FIPS_cnty'])
    df = df.groupby(by=['region'] + groupby_cols, as_index=False).sum(data_id)

    return df


def format_demand(demand: pd.DataFrame):
    """pivot demand data to 8760 hours

    Parameters
    ----------
    demand : pd.DataFrame
        demand data in Month/Day/Hour format

    Returns
    -------
    pd.DataFrame
        pivoted demand data in 6750 hour format
    """
    demand['region'] = demand['region'].astype(int)
    demand.loc[demand['Value'] == 0, 'Value'] = np.nan
    demand = demand.pivot_table(
        index=['Month', 'Day', 'Hour'], columns='region', values='Value'
    ).reset_index()
    demand = demand.ffill(axis=0)
    demand['hr'] = demand.index + 1
    demand = demand[['hr'] + [col for col in demand.columns if col != 'hr']]
    demand = pd.melt(
        demand, id_vars=['hr'], value_vars=list(range(1, 26)), var_name='region', value_name='Load'
    ).reset_index(drop=True)

    demand = demand.rename(columns={'hr': 'hour'})
    demand = demand[['region', 'hour', 'Load']]

    return demand


def baseload():
    """apply population-weighted BA-to-EMM region mapping and write out to BaseLoad.csv
    1. loads in data at BA granularity
    2. calculates fraction of BA population in each county
    3. maps load data to county (population-weighted)
    4. maps and aggregates load data to EMM region
    5. formats load in 8760 hour format
    """
    # baseload -- read in state-county-BA-SR crosswalk and fips county-state codes
    BASR_cw = pd.read_csv('outputs/crosswalks/EIA_BA-BASR-County.csv')
    BASR_cw = BASR_cw.drop(columns=['Region_Name', 'FIPS_st', 'BA_Code', 'BASR_Code'])
    # filter out by population year
    BASR_cw = BASR_cw[BASR_cw['Year'] == population_year].drop(columns=['Year'])

    # baseload -- apply baseload functions
    df_all = read_in_load_data()
    pop = calc_pop_cw(BASR_cw)
    demand = sum_data_cnty(df_all, pop)
    demand = sum_data_r(demand, pop)
    ff = format_demand(demand)
    ff.to_csv('outputs/BaseLoad.csv', index=False)

    ##################################################################################################################


# Main Project Execution


def main():
    """writes out csv of baseload aggregated to EMM regions"""
    print(os.getcwd())
    baseload()


if __name__ == '__main__':
    main()
