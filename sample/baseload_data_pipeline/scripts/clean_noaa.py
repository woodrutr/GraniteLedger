"""creation of BA-level NOAA-ISD weather data series

Summary
-------
This script constructs a dataset of weather-related predictors at hourly timesteps by BA regions

The script relies on previously constructed crosswalks that assign counties to BAs; the script loads these crosswalks to identify which county weather data to load.
Once this data is loaded, the data is collapsed to a BA-level weather series using weighted averages of county variables, with the weights determined by county population.
The resulting dataset is saved as a csv.

Snakemake
---------

l_dbloc
    Snakemake-originated string pointing to database file locations

l_years
    Snakemake-originated list of years to use to construct final dataset
"""

import pandas as pd
import numpy as np
import os
import re
from functools import partial
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

#####
### Read Snakemake inputs
#####
l_dbloc = snakemake.params['db_loc']
l_years = snakemake.params['years_sm']

# l_dbloc = "N:/NextGen Developers/Projects/demand_profiles"
# l_years = [x for x in range(1990, 2023)]
l_var = ['tmp_air_celsius_10', 'tmp_dew_celsius_10', 'wind_dir_degrees', 'wind_spd_m_per_sec_10']


def ba_basr_list():
    """ba_basr_list
    Load crosswalk between BAs and Counties to obtain list of all BAs and BASRs

    Returns
    -------
    l_ba_basr : list
        List of BAs and BASRs
    """
    ### Load Crosswalks and Merge Balancing AUthorites
    df_walk = pd.read_csv(os.path.join('outputs', 'crosswalks', 'EIA_BA-BASR-County.csv'))
    l_ba_basr = df_walk['BA_BASR_Code'].drop_duplicates().tolist()
    return l_ba_basr


def main():
    """Prepare connect to raw noaa data"""

    #####
    ### Step 1: Setting up NOAA tables
    #####

    db_noaa_Base = automap_base()
    db_noaa_engine = create_engine(f'sqlite:///{l_dbloc}/noaa_db.db')
    db_noaa_metadata = MetaData()
    db_noaa_metadata.reflect(bind=db_noaa_engine)
    db_noaa_Base.prepare(autoload_with=db_noaa_engine)
    db_noaa_sessionmaker = sessionmaker(bind=db_noaa_engine)

    #####
    ### Step 2: Load lists of unique ba/basr to run through merging functions; define and run
    #####

    l_ba_basr = ba_basr_list()

    def f_loadmerge_ba(
        i_ba_basr,
        i_year,
        l_var=l_var,
        l_dbloc=l_dbloc,
        md_noaa=db_noaa_metadata,
        b_noaa=db_noaa_Base,
        eg_noaa=db_noaa_engine,
        s_noaa=db_noaa_sessionmaker,
    ):
        """f_loadmerge_ba
        Loads NOAA data at county scale and creates weighted average of weather outcomes at BA/BASR scale

        Parameters
        ----------
        i_ba_basr : string
            BA and/or BASR code
        l_var : list, optional
            List of variables from NOAA to load, by default l_var
        l_dbloc : string, optional
            Location of database file, by default l_dbloc
        md_noaa : SQLAlchemy Metadata Object, optional
            NOAA db metadata object, by default db_noaa_metadata
        b_noaa : SQLAlchemy Base Class, optional
            Base class for NOAA db, by default db_noaa_Base
        eg_noaa : SQLAlchemy Engine Object, optional
            Engine object for NOAA db, by default db_noaa_engine
        s_noaa : SQLAlchemy Sessionmaker Object, optional
            Sessionmaker object for NOAA db, by default db_noaa_sessionmaker

        Returns
        -------
        message : string
            String indicating success or failure of merge
        """
        try:
            ### Pull BA and BASR from code
            str_ba = i_ba_basr[0 : i_ba_basr.find('-')]
            str_basr = i_ba_basr[(i_ba_basr.find('-') + 1) : len(i_ba_basr)]

            ### If BASR is NA, indicate that the BA/BASR is a BA, else BASR
            if str_basr == 'NA':
                str_ba_basr_ind = 'BA'
                l_ba_basr_cw = [str_basr]
            else:
                str_ba_basr_ind = 'BASR'
                l_ba_basr_cw = pd.read_csv(
                    os.path.join('outputs', 'crosswalks', 'EIA_BASR-Details.csv')
                )
                l_ba_basr_cw = l_ba_basr_cw[l_ba_basr_cw.BASR_Code == str_basr]
                l_ba_basr_cw = l_ba_basr_cw.BASR_Original.tolist()

            ### Create sessions for querying tables within function scope
            session_noaa = s_noaa()

            ### Capture models from automapped base for noaa db
            ba_county = b_noaa.classes.ba_counties
            county_noaa = b_noaa.classes.counties_noaa
            county = b_noaa.classes.counties

            #####
            ### Filter counties needed for ba by year; load counties from session
            #####
            df_fips_fil = (
                session_noaa.query(ba_county).filter(ba_county.BA_BASR_Code == i_ba_basr).all()
            )
            df_fips_fil = [[t for t in s.__dict__.items()] for s in df_fips_fil]
            df_fips_fil = [{k: v for k, v in t} for t in df_fips_fil]
            df_fips_fil = pd.DataFrame(df_fips_fil).drop(['_sa_instance_state'], axis=1)

            l_fips_fil = df_fips_fil['FIPS_cnty'].drop_duplicates().to_list()

            ### Grab counties and filter for 2019 population
            df_county = session_noaa.query(county).filter(county.FIPS_cnty.in_(l_fips_fil)).all()
            df_county = [[t for t in s.__dict__.items()] for s in df_county]
            df_county = [{k: v for k, v in t} for t in df_county]
            df_county = pd.DataFrame(df_county).drop(['_sa_instance_state'], axis=1)
            df_county = df_county[df_county['year'] == 2019]
            df_county = df_county[['population', 'FIPS_cnty']]

            ### TEMPORARY: Load csvs of each county needed for weather data; sql pulls taking too long
            l_fips_join = '|'.join(l_fips_fil)
            l_files_year = [
                x
                for x in os.listdir(f'{l_dbloc}/FIPS-NOAA')
                if re.search(f'({l_fips_join})', x) is not None
            ]
            l_files_year = [x for x in l_files_year if re.search(f'-{i_year}', x) is not None]
            df_weather = pd.concat([pd.read_csv(f'{l_dbloc}/FIPS-NOAA/{x}') for x in l_files_year])
            df_weather['FIPS_cnty'] = [str(x).rjust(5, '0') for x in df_weather['FIPS_cnty']]
            df_weather_pop = df_weather.merge(
                df_county[['population', 'FIPS_cnty']], how='left', on=['FIPS_cnty']
            ).reset_index(drop=True)

            ### Create weights based on population per year
            df_weather_pop = df_weather_pop.merge(
                right=df_weather_pop.groupby(by=['year', 'month', 'day', 'hour'])['population']
                .sum()
                .reset_index(),
                how='left',
                on=['year', 'month', 'day', 'hour'],
                suffixes=['', '_sum'],
            )
            df_weather_pop['weight'] = np.round(
                df_weather_pop['population'] / df_weather_pop['population_sum'], 3
            )

            ### Generate Weighted Variable for timesteps in data
            for i_var in l_var:
                try:
                    df_weather_pop[f'{i_var}_'] = round(
                        df_weather_pop['weight'] * df_weather_pop[i_var], 3
                    )
                    df_weather_pop = df_weather_pop.merge(
                        df_weather_pop.groupby(['year', 'month', 'day', 'hour'])[f'{i_var}_']
                        .sum()
                        .reset_index(),
                        how='left',
                        on=['year', 'month', 'day', 'hour'],
                        suffixes=['', 'weighted'],
                    )
                except Exception as e:
                    print(
                        f'Error collapsing county observations for {i_ba_basr} in {i_year} for variable {i_var}: {e}'
                    )
                    df_weather_pop[f'{i_var}_weighted'] = None

            l_keepvar = [f'{x}_weighted' for x in l_var] + ['year', 'month', 'day', 'hour']
            df_weather_pop = df_weather_pop[l_keepvar]
            df_weather_pop = df_weather_pop.drop_duplicates()
            df_weather_pop.set_index(['year', 'month', 'day', 'hour'], inplace=True)

            ### Write to csv at db location directory TEMP
            df_weather_pop.to_csv(
                os.path.join(f'{l_dbloc}/BA-NOAA/BA-NOAA_{i_ba_basr}_{i_year}.csv')
            )

            ### Return message; print
            message = f'County collapse for {i_ba_basr} in {i_year} completed. Written temporarily to csv.'
            print(message)
            return message
        except Exception as e:
            message = f'County collapse for {i_ba_basr} in {i_year} failed. Check for errors: {e}'
            print(message)
            return message

    for year in l_years:
        f_loadmerge_ba_p = partial(f_loadmerge_ba, i_year=year)
        results = list(map(f_loadmerge_ba_p, l_ba_basr))

    #####
    ### Create csv file that indicate runs for snakemake purposes
    #####
    df_run_summary = pd.DataFrame({'BA_BASR_Code': l_ba_basr, 'RunMessage': results})
    df_run_summary.to_csv(
        os.path.join(
            f'{l_dbloc}/BA-NOAA/BA-NOAA_SnakemakeTrack_{l_years[0]}-{l_years[len(l_years)-1]}.csv'
        )
    )


if __name__ == '__main__':
    main()
