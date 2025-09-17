"""This file is the main preprocessor for the electricity model.

It established the parameters and sets that will be used in the model. It contains:
 - A class that contains all sets used in the model
 - A collection of support functions to read in and setup parameter data
 - The preprocessor function, which produces an instance of the Set class and a dict of params
 - A collection of support functions to write out the inputs to the output directory

"""

###################################################################################################
# Setup

# Import pacakges
from pathlib import Path
import pandas as pd
import os

# Import python modules
from definitions import PROJECT_ROOT
from src.common.utilities import scale_load, scale_load_with_enduses

# switch to load data from csvs(0) or from db(1)
# note: this is a future feature, currently not available
db_switch = 0

if db_switch == 1:
    from sqlalchemy import create_engine, MetaData, Table, select
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import sessionmaker

# Establish paths
data_root = Path(PROJECT_ROOT, 'input', 'electricity')


DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH = {
    1: 1000.0,
    2: 800.0,
    3: 620.0,
    4: 370.0,
    5: 0.0,
    6: 0.0,
    7: 0.0,
    8: 0.0,
    9: 0.0,
    10: 0.0,
    11: 0.0,
    12: 0.0,
    13: 0.0,
    14: 0.0,
    15: 0.0,
}


###################################################################################################
class Sets:
    """Generates an initial batch of sets that are used to solve electricity model. Sets include: \n
    - Scenario descriptor and model switches \n
    - Regional sets \n
    - Temporal sets \n
    - Technology type sets \n
    - Supply curve step sets \n
    - Other

    """

    def __init__(self, settings):
        # Output root
        self.OUTPUT_ROOT = settings.OUTPUT_ROOT

        # Switches
        self.sw_trade = settings.sw_trade
        self.sw_expansion = settings.sw_expansion
        self.sw_agg_years = settings.sw_agg_years
        self.sw_rm = settings.sw_rm
        self.sw_ramp = settings.sw_ramp
        self.sw_learning = settings.sw_learning
        self.sw_reserves = settings.sw_reserves
        self.carbon_cap = settings.carbon_cap
        self.carbon_allowance_procurement_overrides = (
            getattr(settings, 'carbon_allowance_procurement', {})
        )
        self.carbon_allowance_start_bank = getattr(
            settings, 'carbon_allowance_start_bank', 0.0
        )
        self.carbon_allowance_bank_enabled = getattr(
            settings, 'carbon_allowance_bank_enabled', True
        )
        self.carbon_allowance_allow_borrowing = getattr(
            settings, 'carbon_allowance_allow_borrowing', False
        )
        self.carbon_cap_groups = getattr(settings, 'carbon_cap_groups', {}) or {}
        self.active_carbon_cap_groups = {}
        self.cap_groups = []
        self.cap_group_membership = pd.DataFrame(columns=['cap_group', 'region'])
        self.carbon_allowance_by_cap_group = pd.DataFrame()
        self.carbon_price_by_cap_group = pd.DataFrame()

        self.restypes = [
            'spinning',
            'regulation',
            'flex',
        ]  # old vals: 1=spinning, 2=regulation, 3=flex
        self.sw_builds = pd.read_csv(data_root / 'sw_builds.csv')
        self.sw_retires = pd.read_csv(data_root / 'sw_retires.csv')

        # Load Setting
        self.load_scalar = settings.scale_load

        # Regional Sets
        self.region = settings.regions

        # Temporal Sets
        self.sw_temporal = settings.sw_temporal
        self.cw_temporal = settings.cw_temporal

        # Temporal Sets - Years
        self.years = settings.years
        self.y = settings.years
        self.start_year = settings.start_year
        self.year_map = settings.year_map
        self.WeightYear = settings.WeightYear

        # Temporal Sets - Seasons and Days
        self.season = range(1, self.cw_temporal['Map_s'].max() + 1)
        self.num_days = self.cw_temporal['Map_day'].max()
        self.day = range(1, self.num_days + 1)

        # Temporal Sets - Hours
        # number of time periods in a day
        self.num_hr_day = int(
            self.cw_temporal['Map_hour'].max() / self.cw_temporal['Map_day'].max()
        )
        self.h = range(1, self.num_hr_day + 1)
        # Number of time periods the model solves for: days x number of periods per day
        self.num_hr = self.num_hr_day * self.num_days
        self.hour = range(1, self.num_days * len(self.h) + 1)
        # First time period of the day and all time periods that are not the first hour
        self.hour1 = range(1, self.num_days * len(self.h) + 1, len(self.h))
        self.hour23 = list(set(self.hour) - set(self.hour1))

        # Technology Sets
        def load_and_assign_subsets(df, col):
            """create list based on tech subset assignment

            Parameters
            ----------
            df : pd.DataFrame
                data frame containing tech subsets
            col : str
                name of tech subset

            Returns
            -------
            list
                list of techs in subset
            """
            # set attributes for the main list
            main = list(df.columns)[0]
            df = df.set_index(df[main])

            # return subset of list based on col assignments
            subset_list = list(df[df[col].notna()].index)
            # print(col,subset_list)

            return subset_list

        # read in subset dataframe from inputs
        tech_subsets = pd.read_csv(data_root / 'tech_subsets.csv')
        self.tech_subset_names = tech_subsets.columns

        for tss in self.tech_subset_names:
            # create the technology subsets based on the tech_subsets input
            setattr(self, tss, load_and_assign_subsets(tech_subsets, tss))

        # Misc Inputs
        self.step = range(1, 5)
        self.TransLoss = 0.02  # Transmission losses %
        self.H2Heatrate = (
            13.84 / 1000000
        )  # 13.84 kwh/kg, for kwh/kg H2 -> 54.3, #conversion kwh/kg to GWh/kg


###################################################################################################
# functions to read in and setup parameter data


### Load csvs
def readin_csvs(all_frames):
    """Reads in all of the CSV files from the input dir and returns a dictionary of dataframes,
    where the key is the file name and the value is the table data.

    Parameters
    ----------
    all_frames : dictionary
        empty dictionary to be filled with dataframes

    Returns
    -------
    dictionary
        completed dictionary filled with dataframes from the input directory
    """
    csv_dir = Path(data_root, 'cem_inputs')
    for filename in os.listdir(csv_dir):
        f = Path(csv_dir, filename)
        if os.path.isfile(f):
            all_frames[filename[:-4]] = pd.read_csv(f)
    return all_frames


### Load table from SQLite DB
def readin_sql(all_frames):
    """Reads in all of the tables from a SQL databased and returns a dictionary of dataframes,
    where the key is the table name and the value is the table data.

    Parameters
    ----------
    all_frames : dictionary
        empty dictionary to be filled with dataframes

    Returns
    -------
    dictionary
        completed dictionary filled with dataframes from the input directory
    """
    db_dir = data_root / 'cem_inputs_database.db'
    engine = create_engine('sqlite:///' + db_dir)
    Session = sessionmaker(bind=engine)
    session = Session()

    Base = automap_base()
    Base.prepare(autoload_with=engine)
    metadata = MetaData()
    metadata.reflect(engine)

    for table in metadata.tables.keys():
        all_frames[table] = load_data(table, metadata, engine)
        all_frames[table] = all_frames[table].drop(columns=['id'])

    session.close()

    return all_frames


def load_data(tablename, metadata, engine):
    """loads the data from the SQL database; used in readin_sql function.

    Parameters
    ----------
    tablename : string
        table name
    metadata : SQL metadata
        SQL metadata
    engine : SQL engine
        SQL engine

    Returns
    -------
    dataframe
        table from SQL db as a dataframe
    """
    table = Table(tablename, metadata, autoload_with=engine)
    query = select(table.c).where()

    with engine.connect() as connection:
        result = connection.execute(query)
        df = pd.read_sql(query, connection)

    return df


def subset_dfs(all_frames, setin, i):
    """filters dataframes based on the values within the set

    Parameters
    ----------
    all_frames : dictionary
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        contains an initial batch of sets that are used to solve electricity model
    i : string
        name of the set contained within the sets class that the df will be filtered based on.

    Returns
    -------
    dictionary
        completed dictionary filled with dataframes filtered based on set inputs specified
    """
    for key in all_frames:
        if i in all_frames[key].columns:
            all_frames[key] = all_frames[key].loc[all_frames[key][i].isin(getattr(setin, i))]

    return all_frames


def _expand_frame_to_cap_groups(df, groups, group_column='cap_group'):
    """Ensure a dataframe contains entries for each requested cap group."""

    if df is None or df.empty or not groups:
        columns = [group_column] + [col for col in (df.columns if df is not None else []) if col != group_column]
        return pd.DataFrame(columns=columns)

    df = df.copy()
    groups = {str(group) for group in groups}

    if group_column in df.columns:
        df[group_column] = df[group_column].astype(str)
        df = df[df[group_column].isin(groups)]
    else:
        df['_tmp'] = 1
        groups_df = pd.DataFrame({group_column: sorted(groups)})
        groups_df['_tmp'] = 1
        df = pd.merge(groups_df, df, on='_tmp', how='outer').drop(columns=['_tmp'])

    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)

    column_order = [group_column] + [col for col in df.columns if col != group_column]
    df = df[column_order]
    return df


def _apply_cap_group_value_overrides(df, group_definitions, value_col, override_keys):
    """Apply overrides defined in the configuration to cap group values."""

    if df is None or df.empty or not group_definitions:
        return df

    df = df.copy()
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

    for group_name, config in group_definitions.items():
        if not isinstance(config, dict):
            continue
        overrides = None
        for key in override_keys:
            value = config.get(key)
            if isinstance(value, dict) and value:
                overrides = value
                break
        if not overrides:
            continue
        for year_key, override_value in overrides.items():
            try:
                year = int(year_key)
                override_float = float(override_value)
            except (TypeError, ValueError):
                continue
            mask = (df['cap_group'] == str(group_name)) & (df['year'] == year)
            if mask.any():
                df.loc[mask, value_col] = override_float
            else:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                'cap_group': [str(group_name)],
                                'year': [year],
                                value_col: [override_float],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    df = df.sort_values(['cap_group', 'year']).reset_index(drop=True)
    return df


def build_cap_group_inputs(all_frames, setin):
    """Construct cap group membership and allowance/price tables."""

    cap_group_map = all_frames.pop('CarbonCapGroupMap', None)
    setin.cap_groups = []
    setin.cap_group_membership = pd.DataFrame(columns=['cap_group', 'region'])
    setin.active_carbon_cap_groups = {}

    if cap_group_map is None or cap_group_map.empty:
        return all_frames, setin

    required_cols = {'cap_group', 'region'}
    if not required_cols.issubset(cap_group_map.columns):
        raise ValueError(
            'CarbonCapGroupMap input must include cap_group and region columns'
        )

    membership = cap_group_map[list(required_cols)].copy()
    membership['cap_group'] = membership['cap_group'].astype(str)
    membership['region'] = pd.to_numeric(membership['region'], errors='coerce')
    membership = membership.dropna(subset=['region'])
    membership['region'] = membership['region'].astype(int)
    membership = membership[membership['region'].isin(setin.region)]
    membership = membership.drop_duplicates()

    if membership.empty:
        return all_frames, setin

    configured_groups = getattr(setin, 'carbon_cap_groups', {}) or {}
    configured_groups = {
        str(group_name): (config if isinstance(config, dict) else {})
        for group_name, config in configured_groups.items()
    }
    if configured_groups:
        membership = membership[membership['cap_group'].isin(configured_groups.keys())]

    active_groups = sorted(membership['cap_group'].unique())
    if not active_groups:
        return all_frames, setin

    membership = membership.sort_values(['cap_group', 'region']).reset_index(drop=True)

    membership_df = membership.copy()
    membership_df['membership'] = 1.0
    all_frames['CarbonCapGroupMembership'] = membership_df[
        ['cap_group', 'region', 'membership']
    ]

    setin.cap_groups = active_groups
    setin.cap_group_membership = membership[['cap_group', 'region']]
    setin.active_carbon_cap_groups = {
        group: configured_groups.get(group, {}) for group in active_groups
    }

    allowance_df = all_frames.get('CarbonAllowanceProcurement')
    if allowance_df is not None and not allowance_df.empty:
        if 'year' not in allowance_df.columns and allowance_df.index.name == 'year':
            allowance_df = allowance_df.reset_index()
        allowance_groups = _expand_frame_to_cap_groups(allowance_df, set(active_groups))
        allowance_groups = _apply_cap_group_value_overrides(
            allowance_groups,
            setin.active_carbon_cap_groups,
            'CarbonAllowanceProcurement',
            ('allowances', 'allowance_procurement', 'carbon_allowance_procurement'),
        )
        allowance_groups = allowance_groups[
            ['cap_group', 'year', 'CarbonAllowanceProcurement']
        ]
        all_frames['CarbonAllowanceProcurementByCapGroup'] = allowance_groups

    price_df = all_frames.get('CarbonAllowancePrice')
    if price_df is not None and not price_df.empty:
        if 'year' not in price_df.columns and price_df.index.name == 'year':
            price_df = price_df.reset_index()
        price_groups = _expand_frame_to_cap_groups(price_df, set(active_groups))
        price_groups = _apply_cap_group_value_overrides(
            price_groups,
            setin.active_carbon_cap_groups,
            'CarbonPrice',
            ('prices', 'price', 'carbon_price'),
        )
        price_groups = price_groups[['cap_group', 'year', 'CarbonPrice']]
        all_frames['CarbonAllowancePriceByCapGroup'] = price_groups

    return all_frames, setin


def fill_values(row, subset_list):
    """Function to fill in the subset values, is used to assign all years within the year
    solve range to each year the model will solve for.

    Parameters
    ----------
    row : int
        row number in df
    subset_list : list
        list of values to map

    Returns
    -------
    int
        value from subset_list
    """
    if row in subset_list:
        return row
    for i in range(len(subset_list) - 1):
        if subset_list[i] < row < subset_list[i + 1]:
            return subset_list[i + 1]
    return subset_list[-1]


def avg_by_group(df, set_name, map_frame):
    """takes in a dataframe and groups it by the set specified and then averages the data.

    Parameters
    ----------
    df : dataframe
        parameter data to be modified
    set_name : str
        name of the column/set to average the data by
    map_frame : dataframe
        data that maps the set name to the new grouping for that set

    Returns
    -------
    dataframe
        parameter data that is averaged by specified set mapping
    """
    map_df = map_frame.copy()
    df = df.sort_values(by=list(df[:-1]))
    # print(df.tail())

    # location of y column and list of cols needed for the groupby
    pos = df.columns.get_loc(set_name)
    map_name = 'Map_' + set_name
    groupby_cols = list(df.columns[:-1]) + [map_name]
    groupby_cols.remove(set_name)

    # group df by year map data and update y col
    df[set_name] = df[set_name].astype(int)
    df = pd.merge(df, map_df, how='left', on=[set_name])
    df = df.groupby(by=groupby_cols, as_index=False).mean()
    df[set_name] = df[map_name].astype(int)
    df = df.drop(columns=[map_name]).reset_index(drop=True)

    # move back to original position
    y_col = df.pop(set_name)
    df.insert(pos, set_name, y_col)

    # used to qa
    df = df.sort_values(by=list(df[:-1]))
    # print(df.tail())

    return df


# add seasons to data without seasons
def add_season_index(cw_temporal, df, pos):
    """adds a season index to the input dataframe

    Parameters
    ----------
    cw_temporal : dataframe
        dataframe that includes the season index
    df : dataframe
        parameter data to be modified
    pos : int
        column position for the seasonal set

    Returns
    -------
    dataframe
        modified parameter data now indexed by season
    """
    df_s = cw_temporal[['Map_s']].copy().rename(columns={'Map_s': 'season'}).drop_duplicates()
    df = pd.merge(df, df_s, how='cross')
    s_col = df.pop('season')
    df.insert(pos, 'season', s_col)

    return df


def time_map(cw_temporal, rename_cols):
    """create temporal mapping parameters

    Parameters
    ----------
    cw_temporal : pd.DataFrame
        temporal crosswalks
    rename_cols : dict
        columns to rename from/to

    Returns
    -------
    pd.DataFrame
        data frame with temporal mapping parameters
    """
    df = cw_temporal[list(rename_cols.keys())].rename(columns=rename_cols).drop_duplicates()
    return df


def capacitycredit_df(all_frames, setin):
    """builds the capacity credit dataframe

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    pd.DataFrame
        formatted capacity credit data frame
    """
    df = pd.merge(
        all_frames['SupplyCurve'], all_frames['MapHourSeason'], on=['season'], how='left'
    ).drop(columns=['season'])

    # capacity credit is hourly capacity factor for vre technologies
    df = pd.merge(
        df, all_frames['CapFactorVRE'], how='left', on=['tech', 'year', 'region', 'step', 'hour']
    ).rename(columns={'CapFactorVRE': 'CapacityCredit'})

    # capacity credit = 1 for dispatchable technologies
    df['CapacityCredit'] = df['CapacityCredit'].fillna(1)

    # capacity credit is seasonal limit for hydro
    df2 = pd.merge(
        all_frames['HydroCapFactor'],
        all_frames['MapHourSeason'],
        on=['season'],
        how='left',
    ).drop(columns=['season'])
    df2['tech'] = setin.T_hydro[0]
    df = pd.merge(df, df2, how='left', on=['tech', 'region', 'hour'])
    df.loc[df['tech'].isin(setin.T_hydro), 'CapacityCredit'] = df['HydroCapFactor']
    df = df.drop(columns=['SupplyCurve', 'HydroCapFactor'])
    df = df[['tech', 'year', 'region', 'step', 'hour', 'CapacityCredit']]
    return df


def create_hourly_params(all_frames, key, cols):
    """Expands params that are indexed by season to be indexed by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    key : str
        name of data frame to access
    cols : list[str]
        column names to keep in data frame

    Returns
    -------
    pd.DataFrame
        data frame with name key with new hourly index
    """
    df = pd.merge(all_frames[key], all_frames['MapHourSeason'], on=['season'], how='left').drop(
        columns=['season']
    )
    df = df[cols]
    return df


def create_subsets(df, col, subset):
    """Create subsets off of full sets

    Parameters
    ----------
    df : pd.DataFrame
        data frame of full data
    col : str
        column name
    subset : list[str]
        names of values to subset

    Returns
    -------
    pd.DataFrame
        data frame containing subset of full data
    """
    df = df[df[col].isin(subset)].dropna()
    return df


def create_hourly_sets(all_frames, df):
    """expands sets that are indexed by season to be indexed by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    df : pd.DataFrame
        data frame containing seasonal data

    Returns
    -------
    pd.DataFrame
        data frame containing updated hourly set
    """
    df = pd.merge(df, all_frames['MapHourSeason'].reset_index(), on=['season'], how='left').drop(
        columns=['season']
    )
    return df


def hourly_sc_subset(all_frames, subset):
    """Creates sets/subsets that are related to the supply curve

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    subset : list
        list of technologies to subset

    Returns
    -------
    pd.DataFrame
        data frame containing sets/subsets related to supply curve
    """
    df = create_hourly_sets(
        all_frames, create_subsets(all_frames['SupplyCurve'].reset_index(), 'tech', subset)
    )
    return df


def hr_sub_sc_subset(all_frames, T_subset, hr_subset):
    """creates supply curve subsets by hour

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    T_subset : list
        list of technologies to subset
    hr_subset : list
        list of hours to subset

    Returns
    -------
    pd.DataFrame
        data frame containing supply curve related hourly subset
    """
    il = ['tech', 'year', 'region', 'step', 'hour']
    df_index = create_subsets(hourly_sc_subset(all_frames, T_subset), 'hour', hr_subset).set_index(
        il
    )
    return df_index


def step_sub_sc_subset(all_frames, T_subset, step_subset):
    """creates supply curve subsets by step

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    T_subset : list
       technologies to subset
    step_subset : list
        step numbers to subset

    Returns
    -------
    pd.DataFrame
        data frame containing supply curve subsets by step
    """
    df = create_subsets(
        create_subsets(all_frames['SupplyCurve'].reset_index(), 'tech', T_subset),
        'step',
        step_subset,
    )
    return df


def create_sc_sets(all_frames, setin):
    """creates supply curve sets

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    Sets
       updated Set containing all sets related to supply curve
    """
    # sets that are related to the supply curve
    index_list = ['tech', 'year', 'region', 'step', 'hour']

    setin.generation_total_index = hourly_sc_subset(all_frames, setin.T_gen).set_index(index_list)
    setin.Storage_index = hourly_sc_subset(all_frames, setin.T_stor).set_index(index_list)
    setin.H2Gen_index = hourly_sc_subset(all_frames, setin.T_h2).set_index(index_list)
    setin.generation_ramp_index = hourly_sc_subset(all_frames, setin.T_conv).set_index(index_list)
    setin.generation_dispatchable_ub_index = hourly_sc_subset(all_frames, setin.T_disp).set_index(
        index_list
    )

    setin.ramp_most_hours_balance_index = hr_sub_sc_subset(all_frames, setin.T_conv, setin.hour23)
    setin.ramp_first_hour_balance_index = hr_sub_sc_subset(all_frames, setin.T_conv, setin.hour1)
    setin.storage_most_hours_balance_index = hr_sub_sc_subset(
        all_frames, setin.T_stor, setin.hour23
    )
    setin.storage_first_hour_balance_index = hr_sub_sc_subset(all_frames, setin.T_stor, setin.hour1)

    setin.generation_hydro_ub_index = create_hourly_sets(
        all_frames, step_sub_sc_subset(all_frames, setin.T_hydro, [2])
    ).set_index(index_list)

    setin.capacity_hydro_ub_index = (
        step_sub_sc_subset(all_frames, setin.T_hydro, [1])
        .drop(columns=['step'])
        .set_index(['tech', 'year', 'region', 'season'])
    )

    setin.reserves_procurement_index = pd.merge(
        create_hourly_sets(all_frames, all_frames['SupplyCurve'].reset_index()),
        pd.DataFrame({'restypes': setin.restypes}),
        how='cross',
    ).set_index(['restypes'] + index_list)

    return setin


def create_other_sets(all_frames, setin):
    """creates other (non-supply curve) sets

    Parameters
    ----------
    all_frames : dict of pd.DataFrame
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    Sets
        updated Sets which has non-supply curve-related sets updated
    """
    # other sets
    setin.Build_index = setin.sw_builds[setin.sw_builds['builds'] == 1].set_index(['tech', 'step'])

    setin.capacity_retirements_index = pd.merge(
        all_frames['SupplyCurve']
        .reset_index()
        .drop(columns=['season', 'SupplyCurve'])
        .drop_duplicates(),
        setin.sw_retires[setin.sw_retires['retires'] == 1],
        on=['tech', 'step'],
        how='right',
    ).set_index(['tech', 'year', 'region', 'step'])

    setin.trade_interational_index = (
        pd.merge(
            all_frames['TranLimitGenInt'].reset_index(),
            all_frames['TranLimitCapInt'].reset_index(),
            how='inner',
        )
        .drop(columns=['TranLimitGenInt'])
        .set_index(['region', 'region1', 'year', 'step', 'hour'])
    )

    setin.trade_interregional_index = create_hourly_sets(
        all_frames, all_frames['TranLimit'].reset_index()
    ).set_index(['region', 'region1', 'year', 'hour'])

    return setin


###################################################################################################
def preprocessor(setin):
    """main preprocessor function that generates the final dataframes and sets sent over to the
    electricity model. This function reads in the input data, modifies it based on the temporal
    and regional mapping specified in the inputs, and gets it into the final formatting needed.
    Also adds some additional regional sets to the set class based on parameter inputs.

    Parameters
    ----------
    setin : Sets
        an initial batch of sets that are used to solve electricity model

    Returns
    -------
    all_frames : dictionary
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """

    # READ IN INPUT DATA

    # read in raw data
    all_frames = {}
    if db_switch == 0:
        # add csv input files to all frames
        all_frames = readin_csvs(all_frames)
    elif db_switch == 1:
        # add sql db tables to all frames
        all_frames = readin_sql(all_frames)

    # Ensure emissions data are available for all generation technologies
    emissions_df = all_frames.get('EmissionsRate')
    if emissions_df is None or emissions_df.empty:
        emissions_df = pd.DataFrame(
            {
                'tech': list(DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH.keys()),
                'EmissionsRate': list(DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH.values()),
            }
        )
    else:
        emissions_df = emissions_df.copy()
    if 'tech' not in emissions_df.columns:
        raise ValueError('EmissionsRate input must include a tech column')
    if 'EmissionsRate' not in emissions_df.columns:
        raise ValueError('EmissionsRate input must include an EmissionsRate column')
    emissions_df['EmissionsRate'] = emissions_df['EmissionsRate'].astype(float)
    missing_techs = set(setin.T_gen) - set(emissions_df['tech'])
    if missing_techs:
        emissions_df = pd.concat(
            [
                emissions_df,
                pd.DataFrame(
                    {
                        'tech': list(missing_techs),
                        'EmissionsRate': [
                            DEFAULT_TECH_EMISSIONS_RATE_TON_PER_GWH.get(tech, 0.0)
                            for tech in missing_techs
                        ],
                    }
                ),
            ],
            ignore_index=True,
        )
    emissions_df = emissions_df[emissions_df['tech'].isin(setin.T_gen)]
    all_frames['EmissionsRate'] = emissions_df

    # Allowance procurement and price inputs (tons and $/ton)
    allowances_df = all_frames.get('CarbonAllowanceProcurement')
    if allowances_df is None:
        allowances_df = pd.DataFrame(
            {
                'year': setin.years,
                'CarbonAllowanceProcurement': [0.0] * len(setin.years),
            }
        )
    else:
        allowances_df = allowances_df.copy()
    if 'year' not in allowances_df.columns:
        raise ValueError('CarbonAllowanceProcurement input must include a year column')
    if 'CarbonAllowanceProcurement' not in allowances_df.columns:
        raise ValueError(
            'CarbonAllowanceProcurement input must include a CarbonAllowanceProcurement column'
        )
    allowances_df['CarbonAllowanceProcurement'] = allowances_df[
        'CarbonAllowanceProcurement'
    ].astype(float)
    overrides = getattr(setin, 'carbon_allowance_procurement_overrides', {}) or {}
    for year, value in overrides.items():
        if not isinstance(year, int):
            year = int(year)
        if allowances_df['year'].eq(year).any():
            allowances_df.loc[
                allowances_df['year'] == year, 'CarbonAllowanceProcurement'
            ] = float(value)
        else:
            allowances_df = pd.concat(
                [
                    allowances_df,
                    pd.DataFrame(
                        {
                            'year': [year],
                            'CarbonAllowanceProcurement': [float(value)],
                        }
                    ),
                ],
                ignore_index=True,
            )
    allowances_df = pd.merge(
        pd.DataFrame({'year': setin.years}), allowances_df, on='year', how='left'
    ).fillna(0.0)
    allowances_df['CarbonAllowanceProcurement'] = allowances_df[
        'CarbonAllowanceProcurement'
    ].astype(float)
    all_frames['CarbonAllowanceProcurement'] = allowances_df

    # read in load data from residential input directory
    res_dir = Path(PROJECT_ROOT, 'input', 'residential')
    if setin.load_scalar == 'annual':
        all_frames['Load'] = scale_load(res_dir).reset_index(drop=True)
    elif setin.load_scalar == 'enduse':
        all_frames['Load'] = scale_load_with_enduses(res_dir).reset_index(drop=True)
    else:
        raise ValueError('load_scalar in TOML must be set to "annual" or "enduse"')

    # REGIONALIZE DATA

    # international trade sets
    r_file = all_frames['TranLimitCapInt'][['region', 'region1']].drop_duplicates()
    r_file = r_file[r_file['region'].isin(setin.region)]
    setin.region_int_trade = list(r_file['region'].unique())
    setin.region_int = list(r_file['region1'].unique())
    setin.region1 = setin.region + setin.region_int

    # subset df by region
    all_frames = subset_dfs(all_frames, setin, 'region')
    all_frames = subset_dfs(all_frames, setin, 'region1')

    all_frames, setin = build_cap_group_inputs(all_frames, setin)

    setin.region_trade = all_frames['TranLimit']['region'].unique()

    # TEMPORALIZE DATA

    # create temporal mapping df
    cw_temporal = setin.cw_temporal

    # year weights
    all_frames['WeightYear'] = setin.WeightYear

    # last year values used
    filter_list = ['CapCost']
    if 'CarbonAllowanceProcurement' in all_frames:
        filter_list.append('CarbonAllowanceProcurement')
    for key in filter_list:
        all_frames[key] = all_frames[key].loc[all_frames[key]['year'].isin(getattr(setin, 'years'))]

    # average values in years/hours used
    for key in all_frames.keys():
        if 'year' in all_frames[key].columns:
            all_frames[key] = avg_by_group(all_frames[key], 'year', setin.year_map)
        if 'hour' in all_frames[key].columns:
            all_frames[key] = avg_by_group(
                all_frames[key], 'hour', cw_temporal[['hour', 'Map_hour']]
            )

    all_frames['MapDaySeason'] = time_map(cw_temporal, {'Map_day': 'day', 'Map_s': 'season'})
    all_frames['MapHourDay'] = time_map(cw_temporal, {'Map_hour': 'hour', 'Map_day': 'day'})
    all_frames['MapHourSeason'] = time_map(cw_temporal, {'Map_hour': 'hour', 'Map_s': 'season'})
    all_frames['WeightHour'] = time_map(
        cw_temporal, {'Map_hour': 'hour', 'WeightHour': 'WeightHour'}
    )
    all_frames['WeightDay'] = time_map(cw_temporal, {'Map_day': 'day', 'WeightDay': 'WeightDay'})

    # weights per season
    all_frames['WeightSeason'] = cw_temporal[
        ['Map_s', 'Map_hour', 'WeightDay', 'WeightHour']
    ].drop_duplicates()
    all_frames['WeightSeason'].loc[:, 'WeightSeason'] = (
        all_frames['WeightSeason']['WeightDay'] * all_frames['WeightSeason']['WeightHour']
    )
    all_frames['WeightSeason'] = (
        all_frames['WeightSeason']
        .drop(columns=['WeightDay', 'WeightHour', 'Map_hour'])
        .groupby(['Map_s'])
        .agg('sum')
        .reset_index()
        .rename(columns={'Map_s': 'season'})
    )

    if 'CarbonAllowanceProcurement' in all_frames:
        allowances_df = all_frames['CarbonAllowanceProcurement'].reset_index()
        allowances_df = pd.merge(
            allowances_df, setin.WeightYear, on='year', how='left'
        ).fillna({'WeightYear': 1})
        allowances_df['CarbonAllowanceProcurement'] = (
            allowances_df['CarbonAllowanceProcurement'] * allowances_df['WeightYear']
        )
        allowances_df = allowances_df[['year', 'CarbonAllowanceProcurement']]
        all_frames['CarbonAllowanceProcurement'] = allowances_df
    if 'CarbonAllowanceProcurementByCapGroup' in all_frames:
        allowances_group_df = all_frames[
            'CarbonAllowanceProcurementByCapGroup'
        ].reset_index()
        allowances_group_df = pd.merge(
            allowances_group_df, setin.WeightYear, on='year', how='left'
        ).fillna({'WeightYear': 1})
        allowances_group_df['CarbonAllowanceProcurement'] = (
            allowances_group_df['CarbonAllowanceProcurement']
            * allowances_group_df['WeightYear']
        )
        allowances_group_df = allowances_group_df[
            ['cap_group', 'year', 'CarbonAllowanceProcurement']
        ]
        all_frames['CarbonAllowanceProcurementByCapGroup'] = allowances_group_df

    # using same T_vre capacity factor for all model years and reordering columns
    all_frames['CapFactorVRE'] = pd.merge(
        all_frames['CapFactorVRE'], pd.DataFrame({'year': setin.years}), how='cross'
    )
    all_frames['CapFactorVRE'] = all_frames['CapFactorVRE'][
        ['tech', 'year', 'region', 'step', 'hour', 'CapFactorVRE']
    ]

    # Update load to be the total demand in each time segment rather than the average
    all_frames['Load'] = pd.merge(
        all_frames['Load'], all_frames['WeightHour'], how='left', on=['hour']
    )
    all_frames['Load']['Load'] = all_frames['Load']['Load'] * all_frames['Load']['WeightHour']
    all_frames['Load'] = all_frames['Load'].drop(columns=['WeightHour'])

    # add seasons to data without seasons
    all_frames['SupplyCurve'] = add_season_index(cw_temporal, all_frames['SupplyCurve'], 1)

    def price_MWh_to_GWh(dic, names: list[str]):
        """changing units of prices to all be in $/GWh so obj is $

        Parameters
        ----------
        dic : dict of pd.DataFrames
            all_frames, main dictionary containing all inputs
        names : list[str]
            names of price tables

        Returns
        -------
        dict of pd.DataFrames
            the original dict of data frames with updated price units
        """
        for name in names:
            dic[name].loc[:, name] = dic[name][name] * 1000
        return dic

    all_frames = price_MWh_to_GWh(
        all_frames,
        [
            'SupplyPrice',
            'TranCost',
            'TranCostInt',
            'RegReservesCost',
            'RampUpCost',
            'RampDownCost',
            'CapCost',
            'CapCostInitial',
        ],
    )

    # Recalculate Supply Curve Learning

    # save first year of supply curve summer capacity for learning
    all_frames['SupplyCurveLearning'] = all_frames['SupplyCurve'][
        (all_frames['SupplyCurve']['year'] == setin.start_year)
        & (all_frames['SupplyCurve']['season'] == 2)
    ]

    # set up first year capacity for learning.
    all_frames['SupplyCurveLearning'] = (
        pd.merge(all_frames['SupplyCurveLearning'], all_frames['CapCost'], how='outer')
        .drop(columns=['season', 'year', 'CapCost', 'region', 'step'])
        .rename(columns={'SupplyCurve': 'SupplyCurveLearning'})
        .groupby(['tech'])
        .agg('sum')
        .reset_index()
    )

    # if cap = 0, set to minimum unit size (0.1 for now)
    all_frames['SupplyCurveLearning'].loc[
        all_frames['SupplyCurveLearning']['SupplyCurveLearning'] == 0.0, 'SupplyCurveLearning'
    ] = 0.01

    all_frames['CapacityCredit'] = capacitycredit_df(all_frames, setin)

    # expand a few parameters to be hourly
    TLCI_cols = ['region', 'region1', 'year', 'hour', 'TranLimitCapInt']
    TLGI_cols = ['region1', 'step', 'year', 'hour', 'TranLimitGenInt']
    all_frames['TranLimitCapInt'] = create_hourly_params(all_frames, 'TranLimitCapInt', TLCI_cols)
    all_frames['TranLimitGenInt'] = create_hourly_params(all_frames, 'TranLimitGenInt', TLGI_cols)

    # sets the index for all df in dict
    for key in all_frames:
        index = list(all_frames[key].columns[:-1])
        all_frames[key] = all_frames[key].set_index(index)

    if 'CarbonCapGroupMembership' in all_frames:
        setin.cap_group_membership = all_frames['CarbonCapGroupMembership']
        if not getattr(setin, 'cap_groups', []):
            setin.cap_groups = list(
                all_frames['CarbonCapGroupMembership']
                .index.get_level_values('cap_group')
                .unique()
            )
    if 'CarbonAllowanceProcurementByCapGroup' in all_frames:
        setin.carbon_allowance_by_cap_group = all_frames[
            'CarbonAllowanceProcurementByCapGroup'
        ]
    if 'CarbonAllowancePriceByCapGroup' in all_frames:
        setin.carbon_price_by_cap_group = all_frames[
            'CarbonAllowancePriceByCapGroup'
        ]

    # create more indices for the model
    setin = create_sc_sets(all_frames, setin)
    setin = create_other_sets(all_frames, setin)

    return all_frames, setin


###################################################################################################
# Review Inputs


def makedir(dir_out):
    """creates a folder directory based on the path provided

    Parameters
    ----------
    dir_out : str
        path of directory
    """
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)


def output_inputs(OUTPUT_ROOT):
    """function developed initial for QA purposes, writes out to csv all of the dfs and sets passed
    to the electricity model to an output directory.

    Parameters
    ----------
    OUTPUT_ROOT : str
        path of output directory

    Returns
    -------
    all_frames : dictionary
        dictionary of dataframes where the key is the file name and the value is the table data
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """
    output_path = Path(OUTPUT_ROOT, 'electricity_inputs/')
    makedir(output_path)

    years = list(pd.read_csv(PROJECT_ROOT / 'src/integrator/input/sw_year.csv').dropna()['year'])
    regions = list(pd.read_csv(PROJECT_ROOT / 'src/integrator/input/sw_reg.csv').dropna()['region'])

    # Build sets used for model
    all_frames = {}
    setA = Sets(years, regions)

    # creates the initial data
    all_frames, setB = preprocessor(setA)
    for key in all_frames:
        # print(key, list(all_frames[key].reset_index().columns))
        fname = key + '.csv'
        all_frames[key].to_csv(output_path / fname)

    return all_frames, setB


def print_sets(setin):
    """function developed initially for QA purposes, prints out all of the sets passed to the
    electricity model.

    Parameters
    ----------
    setin : Sets
        an initial batch of sets that are used to solve electricity model
    """
    set_list = dir(setin)
    set_list = sorted([x for x in set_list if '__' not in x])
    for item in set_list:
        if isinstance(getattr(setin, item), pd.DataFrame):
            print(item, ':', getattr(setin, item).reset_index().columns)
        else:
            print(item, ':', getattr(setin, item))


# all_frames, setB = output_inputs(PROJECT_ROOT)
# print_sets(setB)
