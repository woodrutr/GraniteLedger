"""This file is a collection of functions that are used in support of the electricity model."""

###################################################################################################
# Setup

# Import pacakges
from pathlib import Path
import sys as sys
import numpy as np
import pyomo.environ as pyo
import pandas as pd

# import scripts
from definitions import PROJECT_ROOT

# establish paths
data_root = Path(PROJECT_ROOT, 'src/models/electricity/input')


###################################################################################################
# Declare things functions


def declare_set(self, sname, df):
    """Assigns the index from the df to be a pyomo set using the name specified.
    Adds the name and index column names to the column dictionary used for post-processing.

    Parameters
    ----------
    sname : string
        name of the set to be declared
    df : dataframe
        dataframe from which the index will be grabbed to generate the set

    Returns
    -------
    pyomo set
        a pyomo set
    """
    sset = pyo.Set(initialize=df.index)
    scols = list(df.reset_index().columns)
    scols = scols[-1:] + scols[:-1]
    self.cols_dict[sname] = scols

    return sset


def declare_param(self, pname, p_set, data, default=0, mutable=False):
    """Assigns the df to be a pyomo parameter using the name specified.
    Adds the name and index column names to the column dictionary used for post-processing.

    Parameters
    ----------
    pname : string
        name of the parameter to be declared
    p_set : pyomo set
        the pyomo set that cooresponds to the parameter data
    data : dataframe, series, float, or int
        dataframe used generate the parameter
    default : int, optional
        by default 0
    mutable : bool, optional
        by default False

    Returns
    -------
    pyomo parameter
        a pyomo parameter
    """
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        param = pyo.Param(p_set, initialize=data, default=default, mutable=mutable)
        pcols = list(data.reset_index().columns)
        pcols = pcols[-1:] + pcols[:-1]
        self.cols_dict[pname] = pcols
    else:
        param = pyo.Param(initialize=data)
        self.cols_dict[pname] = [pname, 'None']

    return param


def declare_var(self, vname, v_set, bound=(0, 1000000000)):
    """Assigns the set to be the index for the pyomo variable being declared.
    Adds the name and index column names to the column dictionary used for post-processing.

    Parameters
    ----------
    vname : str
        name of pyomo variable
    v_set : pyomo set
        the pyomo set that the variable data will be indexed by
    bound : set, optional
        optional argument for setting variable bounds, default values set to zero to one billion

    Returns
    -------
    pyomo variable
        a pyomo variable
    """
    var = pyo.Var(v_set, within=pyo.NonNegativeReals, bounds=bound)
    sname = v_set.name
    vcols = [vname] + self.cols_dict[sname][1:]
    self.cols_dict[vname] = vcols
    return var


###################################################################################################
# Populate sets functions


def populate_sets_rule(m1, sname, set_base_name='', set_base2=[]):
    """Generic function to create a new re-indexed set for a PowerModel instance which
    should speed up build time. Must pass non-empty (either) set_base_name or set_base2

    Parameters
    ----------
    m1 : PowerModel
        electricity pyomo model instance
    sname : str
        name of input pyomo set to base reindexing
    set_base_name : str, optional
        the name of the set to be the base of the reindexing, if left blank, uses set_base2, by default ''
    set_base2 : list, optional
        the list of names of set columns to be the base of the reindexing, if left blank, should use set_base_name, by default []

    Returns
    -------
    pyomo set
        reindexed set to be added to electricity model
    """
    # TODO: speed up function

    set_in = getattr(m1, sname)
    scols = m1.cols_dict[sname][1:]

    if set_base_name == '':
        # passed in list (set_base2) of column names, these will be the new base set
        scol_base = np.array([s in set_base2 for s in scols], dtype=bool)
        scols2 = list(np.array(scols)[scol_base])
        scol_base_order = np.array([scols2.index(s) for s in set_base2])
        m1.set_out = {}
    else:
        set_base = getattr(m1, set_base_name)  # pull single base set name
        m1.set_out = pyo.Set(set_base)
        scol_base = np.array([s == set_base_name for s in scols], dtype=bool)

    for i in set_in:  # iterate through set passed in
        i = np.array(i)
        rest_i = tuple(i[~scol_base])  # non-base set values
        if set_base_name == '':
            base_i = tuple(i[scol_base][scol_base_order])
            if base_i not in m1.set_out:
                m1.set_out[base_i] = []
            m1.set_out[base_i].append(rest_i)
        else:
            base_i = int(i[scol_base][0])

            m1.set_out[base_i].add(rest_i)
    # return the new set instead of adding directly to model
    set_out = m1.set_out
    m1.del_component('set_out')
    return set_out


def populate_by_hour_sets_rule(m):
    """Creates new reindexed sets for dispatch_cost calculations

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.StorageHour_index = populate_sets_rule(m, 'Storage_index', set_base_name='hr')
    m.GenHour_index = populate_sets_rule(m, 'generation_total_index', set_base_name='hr')
    m.H2GenHour_index = populate_sets_rule(m, 'H2Gen_index', set_base_name='hr')


def populate_demand_balance_sets_rule(m):
    """Creates new reindexed sets for demand balance constraint

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.GenSetDemandBalance = populate_sets_rule(
        m, 'generation_total_index', set_base2=['y', 'r', 'hr']
    )
    m.StorageSetDemandBalance = populate_sets_rule(m, 'Storage_index', set_base2=['y', 'r', 'hr'])

    if m.sw_trade == 1:
        m.TradeSetDemandBalance = populate_sets_rule(
            m, 'trade_interregional_index', set_base2=['y', 'r', 'hr']
        )
        m.TradeCanSetDemandBalance = populate_sets_rule(
            m, 'trade_interational_index', set_base2=['y', 'r', 'hr']
        )


def populate_trade_sets_rule(m):
    """Creates new reindexed sets for trade constraints

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.TradeCanLineSetUpper = populate_sets_rule(
        m, 'trade_interational_index', set_base2=['r', 'r1', 'y', 'hr']
    )
    m.TradeCanSetUpper = populate_sets_rule(
        m, 'trade_interational_index', set_base2=['r1', 'y', 'CSteps', 'hr']
    )


def populate_RM_sets_rule(m):
    """Creates new reindexed sets for reserve margin constraint

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.SupplyCurveRM = populate_sets_rule(m, 'capacity_total_index', set_base2=['y', 'r', 's'])


def populate_hydro_sets_rule(m):
    """Creates new reindexed sets for hydroelectric generation seasonal upper bound constraint

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.HourSeason_index = pyo.Set(m.s)
    for hr, s in (m.Map_hr_s.extract_values()).items():
        m.HourSeason_index[s].add(hr)


def populate_reserves_sets_rule(m):
    """Creates new reindexed sets for operating reserves constraints

    Parameters
    ----------
    m : PowerModel
        pyomo electricity model instance
    """
    m.WindSetReserves = {}
    m.SolarSetReserves = {}

    m.ProcurementSetReserves = populate_sets_rule(
        m, 'reserves_procurement_index', set_base2=['restypes', 'r', 'y', 'hr']
    )
    for pt, year, reg, step, hour in m.generation_vre_ub_index:
        if (year, reg, hour) not in m.WindSetReserves:
            m.WindSetReserves[(year, reg, hour)] = []
        if (year, reg, hour) not in m.SolarSetReserves:
            m.SolarSetReserves[(year, reg, hour)] = []

        if pt in m.ptw:
            m.WindSetReserves[(year, reg, hour)].append((pt, step))
        elif pt in m.ptsol:
            m.SolarSetReserves[(year, reg, hour)].append((pt, step))


###################################################################################################
# Other functionality


def create_obj_df(mod_object):
    """takes pyomo component objects (e.g., variables, parameters, constraints) and processes the
    pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir.
    The dataframe contains a key column which is the original way the pyomo data is structured,
    as well as columns broken out for each set and the final values.

    Parameters
    ----------
    mod_object : pyomo component object
        pyomo component object

    Returns
    -------
    pd.DataFrame
        contains the pyomo model results for the component object
    """
    name = str(mod_object)
    # print(name)

    # creating a dataframe that reads in the paramater info
    df = pd.DataFrame()
    df['Key'] = [str(i) for i in mod_object]

    # add values associated with model objects that have values
    if isinstance(mod_object, pyo.Set):
        pass
    else:
        df[name] = [pyo.value(mod_object[i]) for i in mod_object]

    if not df.empty:
        # breaking out the data from the mod_object info into multiple columns
        df['Key'] = df['Key'].str.replace('(', '', regex=False).str.replace(')', '', regex=False)
        temp = df['Key'].str.split(', ', expand=True)
        for col in temp.columns:
            temp.rename(columns={col: 'i_' + str(col)}, inplace=True)
        df = df.join(temp, how='outer')

    return df


def annual_count(hour, m) -> int:
    """return the aggregate weight of this hour in the representative year
    we know the hour weight, and the hours are unique to days, so we can
    get the day weight

    Parameters
    ----------
    hour : int
        the rep_hour

    Returns
    -------
    int
        the aggregate weight (count) of this hour in the rep_year.  NOT the hour weight!
    """
    day_weight = m.Idaytq[m.Map_hr_d[hour]]
    hour_weight = m.Hr_weights[hour]
    return day_weight * hour_weight
