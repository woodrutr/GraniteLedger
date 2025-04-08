import pandas as pd
import datetime

# Read in snakemake inputs
sm_years = snakemake.params['years_sm']


def EIA930_download(year, regionality, desired_columns):
    """downloads EIA930 data from url for specified year and cleans names slightly

    Parameters
    ----------
    year : int
        year to pull
    regionality : str
        BALANCE or SUBREIGON, the regionality of data
    desired_columns : list of str
        list of column names to pull from url

    Returns
    -------
    pd.DataFrame
        all data from specified year
    """
    df_all = pd.DataFrame()

    urls = [
        f'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_{regionality}_{year}_Jan_Jun.csv',
        f'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_{regionality}_{year}_Jul_Dec.csv',
    ]

    for url in urls:
        try:
            df = pd.read_csv(url, usecols=desired_columns, encoding='utf-8', low_memory=False)

            if 'Demand (MW) (Adjusted)' in df.columns:
                df = df.rename(columns={'Demand (MW) (Adjusted)': 'Demand (MW)'})
            df['UTC Time at End of Hour'] = pd.to_datetime(df['UTC Time at End of Hour'])
            df['Local Time at End of Hour'] = pd.to_datetime(df['Local Time at End of Hour'])
            df['Demand (MW)'] = df['Demand (MW)'].replace(',', '', regex=True)
            df['Demand (MW)'] = (
                pd.to_numeric(df['Demand (MW)'], errors='coerce').fillna(0).astype(int)
            )

            df_all = pd.concat([df_all, df])

        except Exception as e:
            print(f'Error downloading or processing data for year {year} from {url}: {e}')
            raise

    return df_all


if __name__ == '__main__':
    sm_years = [2018, 2019, 2020, 2021, 2022, 2023]
    ba_years = [x for x in sm_years if x > 2017]
    sr_years = [x for x in sm_years if x > 2018]

    desired_cols = [
        'Balancing Authority',
        'UTC Time at End of Hour',
        'Demand (MW) (Adjusted)',
        'Hour Number',
        'Data Date',
        'Local Time at End of Hour',
    ]

    print('Start BA')
    for year in ba_years:
        print(' ', year)
        EIA930_download(year, 'Balance', desired_cols).to_csv(
            f'inputs/EIA930/EIA930_Balance_{year}.csv', index=False
        )

    desired_cols = [
        'Balancing Authority',
        'Sub-Region',
        'UTC Time at End of Hour',
        'Demand (MW)',
        'Hour Number',
        'Data Date',
        'Local Time at End of Hour',
    ]

    print('Start BA-SR')
    for year in sr_years:
        print(' ', year)
        EIA930_download(year, 'Subregion', desired_cols).to_csv(
            f'inputs/EIA930/EIA930_Subregion_{year}.csv', index=False
        )
