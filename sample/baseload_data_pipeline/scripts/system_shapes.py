# Import python packages
import pandas as pd
import numpy as np

# Read in snakemake inputs
sm_years = snakemake.params['years_sm']

##################################################################################################################
# Functions used to determine day-type data


def create_930_annual(regionality, year):
    """reads annnual 930 files

    Parameters
    ----------
    regionality : str
        BALANCE or SUBREGION, the level of regionality to pull
    year : int
        year to pull data

    Returns
    -------
    pd.DataFrame
        EIA930 data from a particular year/regionality
    """
    df = pd.read_csv(f'inputs/EIA930/EIA930_{regionality}_{year}.csv')
    return df


def create_combined_annual(year_range):
    """combines the annual data for all years and for ba and sr regions

    Parameters
    ----------
    year_range : list
        years to combine data

    Returns
    -------
    pd.DataFrame
        combined data for balancing authorities/subregions and all years in year_range
    """
    # Balancing authority data
    regionality = 'BALANCE'
    ba = pd.DataFrame()
    for year in year_range:
        ba = pd.concat([ba, create_930_annual(regionality, year)])

    # TEMP FIX FOR BAD DATA
    ba = ba[~ba['Balancing Authority'].isin(['GRIF'])]

    # Balancing authority sub-regional data
    regionality = 'SUBREGION'
    sr = pd.DataFrame()
    for year in year_range:
        sr = pd.concat([sr, create_930_annual(regionality, year)])

    # removes BAs that have sub-regional data from the BA dataset
    basr_names = sr['Balancing Authority'].unique()
    df = ba[~ba['Balancing Authority'].isin(basr_names)].copy()

    # Combineds the BA and subregional data into a single frame
    sr['Balancing Authority'] = sr['Balancing Authority'] + '_' + sr['Sub-Region']
    sr = sr.drop(columns=['Sub-Region'])
    df = pd.concat([df, sr]).rename(columns={'Data Date': 'Date', 'Hour Number': 'Hour'})

    # Reformatting df columns and removing excess rows
    df['Date'] = pd.to_datetime(df['Date']).dt.date.astype(str)
    df.insert(2, 'Month', pd.to_datetime(df['Date']).dt.month.astype(int))

    return df


# cleans demand data
def demand_imputation(df):
    """
    Performs the following imputation logic for regional data:

    If a full month of data is missing, keep as empty, otherwise:
    1.) change poorly reported numbers
        a.) sets to nan hours with very low and very high demand
        b.) sets to nan hours with very large absolute changes in demand
        c.) overwrites demand based on a lower and/or upper demand threshold
    2.) fill in missing values
        a.) interpolates demand for days with only a few missing hours
        b.) uses demand from the day before and/or after
        c.) interpolates again for days with some more missing hours
        d.) uses median values for each region/month/hour

    Parameters
    ----------
    df : pd.DataFrame
        eia930subregion dataframe

    Returns
    -------
    pd.DataFrame
        eia930subregion dataframe with imputed demand
    """
    regionality = 'Balancing Authority'

    # clean up column data
    df['Demand (MW)'] = df['Demand (MW)'].apply(pd.to_numeric, errors='coerce')
    df = df[df['Hour'] <= 24].copy()
    df['Year'] = pd.DatetimeIndex(df['Date']).year

    # remove regions with no data for all years
    empty_reg = df[[regionality, 'Demand (MW)']].groupby(by=[regionality], as_index=False).sum()
    empty_reg = empty_reg[empty_reg['Demand (MW)'] == 0].drop(columns=['Demand (MW)'])
    empty_reg['Missing_Region_Flag'] = 1
    df = pd.merge(df, empty_reg, on=[regionality], how='left')
    df = df[df['Missing_Region_Flag'].isna()].drop(columns=['Missing_Region_Flag'])

    # flag regions/months with an entire month of missing data
    empty_mon = (
        df[[regionality, 'Month', 'Year', 'Demand (MW)']]
        .groupby(by=[regionality, 'Month', 'Year'], as_index=False)
        .sum()
    )
    empty_mon = empty_mon[empty_mon['Demand (MW)'] == 0].drop(columns=['Demand (MW)'])
    empty_mon['Missing_Month_Flag'] = 1
    df = pd.merge(df, empty_mon, on=[regionality, 'Month', 'Year'], how='left')

    # create imputed demand column
    df['imputed_demand'] = df['Demand (MW)']

    # set to nan hours with very low demands
    df.loc[df['imputed_demand'] <= 0, 'Lower_Error_Flag'] = 1
    df.loc[df['imputed_demand'] <= 0, 'imputed_demand'] = np.nan

    # set to nan hours with very high demands
    high_limt = (
        df.groupby(by=[regionality, 'Month', 'Hour'])['imputed_demand']
        .agg(['median'])
        .reset_index()
    )
    high_limt['Upper_Error'] = high_limt['median'] * 3
    high_limt = high_limt[[regionality, 'Month', 'Hour', 'Upper_Error']]
    df = pd.merge(df, high_limt, on=[regionality, 'Month', 'Hour'], how='left')
    df.loc[df['imputed_demand'] > df['Upper_Error'], 'Upper_Error_Flag'] = 1
    df.loc[df['imputed_demand'] > df['Upper_Error'], 'imputed_demand'] = np.nan

    # calculate the change in demand from one hour to the next by region
    df = df.reset_index(drop=True)
    for _, group in df.groupby(regionality):
        group_index = group.index
        df.loc[group_index, 'demand_increment'] = abs(
            group['imputed_demand'] - group['imputed_demand'].shift(-1)
        )

    # calculate threshold for very large changes in incremental demand
    increment = (
        df.groupby(by=[regionality])['demand_increment']
        .agg(['count', 'median', 'mean', 'std'])
        .reset_index()
    )
    increment['increment_threshold'] = increment['median'] + 4 * increment['std']
    df = pd.merge(
        df, increment.drop(columns=['count', 'median', 'mean', 'std']), how='left', on=[regionality]
    )

    # identify and set to nan values with very large changes in incremental demand
    df.loc[df['demand_increment'] > df['increment_threshold'], 'Increment_Flag'] = 1
    df['demand_increment'] = df['demand_increment'].shift(1)
    df.loc[df['demand_increment'] > df['increment_threshold'], 'Increment_Flag'] += 1
    df.loc[df['Increment_Flag'] == 2, 'imputed_demand'] = np.nan
    df.loc[df['Increment_Flag'] == 1, 'Increment_Flag'] = np.nan
    df.loc[df['Increment_Flag'] == 2, 'Increment_Flag'] = 1

    # calculate high and low demand hour thresholds by region, month, and hour
    thresholds = (
        df.groupby(by=[regionality, 'Month', 'Hour'])['imputed_demand']
        .agg(['count', 'median', 'mean', 'std'])
        .reset_index()
    )
    thresholds['Upper_Threshold'] = thresholds['median'] + 4 * thresholds['std']
    thresholds['Lower_Threshold'] = thresholds['median'] - 4 * thresholds['std']
    thresholds.loc[thresholds['Lower_Threshold'] < 0, 'Lower_Threshold'] = 0
    thresholds = thresholds[
        [regionality, 'Month', 'Hour', 'median', 'Upper_Threshold', 'Lower_Threshold']
    ]
    df = pd.merge(df, thresholds, on=[regionality, 'Month', 'Hour'], how='left')

    # replace high and low demand numbers with threshold values
    df.loc[df['imputed_demand'] < df['Lower_Threshold'], 'Lower_Threshold_Flag'] = 1
    df.loc[df['imputed_demand'] > df['Upper_Threshold'], 'Upper_Threshold_Flag'] = 1
    df.loc[df['imputed_demand'] < df['Lower_Threshold'], 'imputed_demand'] = df['Lower_Threshold']
    df.loc[df['imputed_demand'] > df['Upper_Threshold'], 'imputed_demand'] = df['Upper_Threshold']

    # interpolate nan values for days with only a few missing hours
    df.loc[df['imputed_demand'].isna(), 'daily_nan_sum'] = 1
    interpo_set = (
        df[[regionality, 'Date', 'daily_nan_sum']]
        .groupby(by=[regionality, 'Date'], as_index=False)
        .sum()
    )
    df = df.drop(columns=['daily_nan_sum'])
    df = pd.merge(df, interpo_set, how='left', on=[regionality, 'Date'])
    interpo_set = df[df['daily_nan_sum'] <= 4].copy()
    interpo_set.loc[interpo_set['imputed_demand'].isna(), 'Interpolate_Flag'] = 1
    for r in interpo_set[regionality].unique():
        print(r)
        int_tmp = interpo_set[interpo_set[regionality] == r]
        interpo_set.loc[interpo_set[regionality] == r, 'imputed_demand'] = int_tmp[
            'imputed_demand'
        ].interpolate()

    noninterp_set = df[df['daily_nan_sum'] > 4].copy()
    df = pd.concat([interpo_set, noninterp_set])
    df.sort_values(by=[regionality, 'UTC Time at End of Hour'], inplace=True)  # changed from local

    # Group by BA and calculate hours values for the day before and/or after
    for _, group in df.groupby(regionality):
        group_index = group.index
        df.loc[group_index, 'shift_demand_d'] = (
            group['imputed_demand'].shift(24) + group['imputed_demand'].shift(-24)
        ) / 2
        df.loc[group_index, 'shift_demand_db'] = group['imputed_demand'].shift(24)
        df.loc[group_index, 'shift_demand_da'] = group['imputed_demand'].shift(-24)

    # fill in missing values with hours from the day before and/or after
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_d'].isna())), 'Shift_Day_Flag'] = 1
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_d'].isna())), 'imputed_demand'] = df[
        'shift_demand_d'
    ]
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_db'].isna())), 'Shift_Day_Flag'] = 1
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_db'].isna())), 'imputed_demand'] = (
        df['shift_demand_db']
    )
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_da'].isna())), 'Shift_Day_Flag'] = 1
    df.loc[((df['imputed_demand'].isna()) & (~df['shift_demand_da'].isna())), 'imputed_demand'] = (
        df['shift_demand_da']
    )

    # interpolate again!
    df.loc[df['imputed_demand'].isna(), 'daily_nan_sum_2'] = 1
    interpo_set = (
        df[[regionality, 'Date', 'daily_nan_sum_2']]
        .groupby(by=[regionality, 'Date'], as_index=False)
        .sum()
    )
    df = df.drop(columns=['daily_nan_sum_2'])
    df = pd.merge(df, interpo_set, how='left', on=[regionality, 'Date'])
    interpo_set = df[df['daily_nan_sum_2'] <= 8].copy()
    interpo_set.loc[interpo_set['imputed_demand'].isna(), 'Interpolate_Flag'] = 1
    interpo_set['imputed_demand'] = interpo_set['imputed_demand'].interpolate()
    for r in interpo_set[regionality].unique():
        print(r)
        int_tmp = interpo_set[interpo_set[regionality] == r]
        interpo_set.loc[interpo_set[regionality] == r, 'imputed_demand'] = int_tmp[
            'imputed_demand'
        ].interpolate()
    noninterp_set = df[df['daily_nan_sum_2'] > 8].copy()
    df = pd.concat([interpo_set, noninterp_set])

    df.sort_values(by=[regionality, 'UTC Time at End of Hour'], inplace=True)  # changed from local

    # Fill remaining NaNs with month/hour mean
    df.loc[df['imputed_demand'].isna(), 'Median_Flag'] = 1
    df.loc[df['imputed_demand'].isna(), 'imputed_demand'] = df['median']

    # set back to zero region/months where entire month was missing
    df['imputed_demand'] = df['imputed_demand'].astype(int)

    # remove extra columns develop for QA
    df = df.drop(columns=['Demand (MW)'])
    df = df.rename(columns={'imputed_demand': 'Demand (MW)'})
    final_table_columns = [
        'Balancing Authority',
        'UTC Time at End of Hour',
        'Month',
        'Hour',
        'Date',
        'Demand (MW)',
    ]  # removed 'Hour','Date','Month','Year'
    df = df.drop(columns=[col for col in df if col not in final_table_columns])

    return df


##################################################################################################################
# Main Project Execution


def main():
    """writes out csv of cleaned 930 hourly load data for specified years and all regions"""
    df = create_combined_annual(sm_years)
    df = demand_imputation(df)
    df.to_csv('outputs/LoadCleaned.csv')


if __name__ == '__main__':
    main()
