"""This file creates the stock database
The stock database contains a table for each building stock type from the comstock and resstock models
The building stock tables contain state-level hourly 2018 equipment kwh demand

Source data websites:
https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock
%2F2024%2Fresstock_amy2018_release_2%2Ftimeseries_aggregates%2Fby_state%2Fupgrade%3D0%2F
https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock
%2F2023%2Fcomstock_amy2018_release_2%2Ftimeseries_aggregates%2Fby_state%2Fupgrade%3D0%2F"""

# import packages
from collections import defaultdict
import io
import pandas as pd
import sqlite3
import grequests
from pathlib import Path
import os
# import requests  # IF needed, this import should be done after grequests, per grequests documentation due to monkeypatching.

# Set directories
# TODO: import structure is to support running locally, will consider changing
PROJECT_ROOT = Path(__file__).parents[4]
os.chdir(PROJECT_ROOT)
input_path = Path(PROJECT_ROOT / 'src/models/residential/input/preprocessor_inputs')

# Set test = True if you want to run a test version to see if the code is working
# Note: requires 10 GB of storage
test = False

# Set db_dir to directory where stock database will live
if test:
    db_dir = Path(input_path / 'stock_database_test.db')
else:
    db_dir = Path(input_path / 'stock_database.db')


# Generate stock list
def stock_list(sector_abbr, building_col):
    """creates a building stock list based on input files

    Parameters
    ----------
    sector_abbr : str
        "com" or "res" for commercial or residential
    building_col : str
        name of the column from the resstock/comstock data
        'in.comstock_building_type' or 'in.geometry_building_type_recs'

    Returns
    -------
    list
        list of building stock types
    """
    stock_list = pd.read_csv(Path(input_path / f'{sector_abbr}stock_assignments.csv'))[
        building_col
    ].unique()
    stock_list = [item.replace(' ', '_') for item in stock_list]

    if test:
        stock_list = stock_list[:2]  # test set

    return stock_list


# Downloads state-level stock data from NREL's OEDI Data Lake
def stock_download(vyear, stock_dir, stock_list, state_list, db_dir):
    """downloads from the web each stock + state combination available and saves datatables
    locally into a SQL database

    Parameters
    ----------
    vyear : str
        version year for which series of data to download
    stock_dir : str
        e.g., 'comstock_amy2018_release_1' or 'resstock_amy2018_release_1' for commercial or residential
    stock_list : list
        list of building stock types
    state_list : list
        list of 48 contiguous states abbreviated + DC
    db_dir : path
        location to stash 'stock_database.db'
    """
    # formulate ALL of the requests, indexed by stock
    request_urls: dict[str, list[str]] = defaultdict(list)
    i = 1
    for stock in stock_list:
        print(f'...processing stock: {stock} ({i}/{len(stock_list)})')
        i += 1
        data: list[pd.DataFrame] = []
        for state in state_list:
            # Define the URLs for the CSV files
            urlp1 = 'https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/'
            urlp2 = 'end-use-load-profiles-for-us-building-stock/'
            urlp3 = '/timeseries_aggregates/by_state/upgrade=0/state='
            url = f'{urlp1}{urlp2}{vyear}{stock_dir}{urlp3}{state}/up00-{state.lower()}-{stock}.csv'
            request_urls[stock].append(url)

        # generate unsent requests
        data_requests = (grequests.get(u) for u in request_urls[stock])

        # send all-at-once (parallel)
        responses = grequests.imap(data_requests, size=20)
        data_requests.close()

        # iterate over the responses
        for response in responses:
            if response.status_code == 200:
                # Download CSV data from the URLs and append to the list
                url_data = response.content
                df = pd.read_csv(io.StringIO(url_data.decode('utf-8')))
                data.append(df)
            else:
                print(
                    f'NOTE: problem with request to: {response.request.url} \n'
                    f'generated response code: {response.status_code}'
                )
                print('  No data loaded for that stock/location...')

        # Concatenate all DataFrames in the list  --  after they are all individually gathered
        df = pd.concat(data, ignore_index=True)
        # Write the concatenated data to an SQLite database
        with sqlite3.connect(db_dir) as conn:
            df.to_sql(stock, conn, index=False, if_exists='replace')

    print(f'Data has been downloaded and written to {db_dir}.')


def main(db_dir):
    """main execution for enduse_db. downloads from the web the residential and commercial
    datatables for each stock+state and saves them locally into a SQL database

    Parameters
    ----------
    db_dir : path
        location to stash 'stock_database.db'
    """
    # States: List of lower 48 states, does not include AK or HI, does include DC
    # breaking out into separate lists to make it easier to read using ruff formatting
    sl1 = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT']
    sl2 = ['DC', 'DE', 'FL', 'GA', 'IA', 'ID']
    sl3 = ['IL', 'IN', 'KS', 'KY', 'LA', 'MA']
    sl4 = ['MD', 'ME', 'MI', 'MN', 'MO', 'MS']
    sl5 = ['MT', 'NC', 'ND', 'NE', 'NH', 'NJ']
    sl6 = ['NM', 'NV', 'NY', 'OH', 'OK', 'OR']
    sl7 = ['PA', 'RI', 'SC', 'SD', 'TN', 'TX']
    sl8 = ['UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    state_list = sl1 + sl2 + sl3 + sl4 + sl5 + sl6 + sl7 + sl8

    if test:
        state_list = state_list[:2]  # test set

    # Comstock building stock list and website directory
    comstock_list = stock_list('com', 'in.comstock_building_type')
    comstock_dir = '/comstock_amy2018_release_1'

    # Write Comstock data to sqlite database
    print('COMSTOCK Database')
    stock_download('2024', comstock_dir, comstock_list, state_list, db_dir)
    print()

    # Resstock building stock list and website directory
    resstock_list = stock_list('res', 'in.geometry_building_type_recs')
    resstock_dir = '/resstock_amy2018_release_1.1'

    # Write Resstock data to sqlite database
    print('RESSTOCK Database')
    stock_download('2022', resstock_dir, resstock_list, state_list, db_dir)
    print()


if __name__ == '__main__':
    print(
        'NOTE:  This code may FAIL if run over VPN due to difficulty \n'
        'in the requests module to verify web certificates through VPN.'
    )
    main(db_dir)
