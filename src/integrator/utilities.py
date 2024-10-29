"""
A gathering of utility functions for dealing with model interconnectivity

Dev Note:  At some review point, some decisions may move these back & forth with parent
models after it is decided if it is a utility job to do .... or a class method.

Additionally, there is probably some renaming due here for consistency
"""

# for easier/accurate indexing
from collections import defaultdict, namedtuple
from logging import getLogger
import typing
import pyomo.opt as pyo
from pyomo.environ import ConcreteModel, value
import pandas as pd
from pathlib import Path
from definitions import PROJECT_ROOT
import logging
import os

if typing.TYPE_CHECKING:
    from src.models.electricity.scripts.electricity_model import PowerModel
    from src.models.hydrogen.model.h2_model import H2Model

logger = getLogger(__name__)


def get_output_root():
    """get the name of the output dir, which includes the name of the mode type and a timestamp

    Returns
    -------
    path
        output directory path
    """

    if os.path.isfile('output_root.txt'):
        with open('output_root.txt', 'r') as file:
            OUTPUT_ROOT = file.read()
            OUTPUT_ROOT = Path(OUTPUT_ROOT)
    else:
        OUTPUT_ROOT = Path(PROJECT_ROOT / 'output' / 'test')
    return OUTPUT_ROOT


def make_dir(dir_name):
    """generates an output directory to write model results, output directory is the date/time
    at the time this function executes. It includes subdirs for vars, params, constraints.

    Returns
    -------
    string
        the name of the output directory
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        logger.info('Asked to make dir that already exists:' + str(dir_name))


# TODO:  This might be a good use case for a persistent solver (1-each) for both the elec & hyd...  hmm
def simple_solve(m: ConcreteModel):
    """a simple solve routine"""

    # Note:  this is a prime candidate to split into 2 persistent solvers!!
    # TODO:  experiment with pyomo's persistent solver interface, one for each ELEC, H2
    opt = select_solver(m)
    res = opt.solve(m)
    if pyo.check_optimal_termination(res):
        return
    raise RuntimeError('failed solve in iterator')


def simple_solve_no_opt(m: ConcreteModel, opt: pyo.SolverFactory):
    """Solve concrete model using solver factory object

    Parameters
    ----------
    m : ConcreteModel
        Pyomo model
    opt: SolverFactory
        Solver object initiated prior to solve
    """

    # Note:  this is a prime candidate to split into 2 persistent solvers!!
    # TODO:  experiment with pyomo's persistent solver interface, one for each ELEC, H2
    logger.info('solving w/ solver-factory object instantiated outside of loop')
    res = opt.solve(m)
    if pyo.check_optimal_termination(res):
        return
    raise RuntimeError('failed solve in iterator')


def select_solver(instance: ConcreteModel):
    """Select solver based on learning method

    Parameters
    ----------
    instance : PowerModel
        electricity pyomo model

    Returns
    -------
    solver type (?)
        The pyomo solver
    """
    # default = linear solver
    solver_name = 'appsi_highs'
    opt = pyo.SolverFactory(solver_name)
    nonlinear_solver = False

    if hasattr(instance, 'sw_learning'):  # check if sw_learning exists in model (electricity model)
        if instance.sw_learning == 2:  # nonlinear solver
            nonlinear_solver = True
    elif hasattr(instance, 'elec'):  # check if sw_learning exists in meta unified model
        if hasattr(instance.elec, 'sw_learning'):
            if instance.elec.sw_learning == 2:  # nonlinear solver
                nonlinear_solver = True

    if nonlinear_solver:  # if nonlinear learning, set to ipopt
        solver_name = 'ipopt'
        opt = pyo.SolverFactory(solver_name, tee=True)  # , tee=True
        # Select options. The prefix "OF_" tells pyomo to create an options file
        opt.options['OF_mu_strategy'] = 'adaptive'
        opt.options['OF_num_linear_variables'] = 100000
        opt.options['OF_mehrotra_algorithm'] = 'yes'
        # Ask IPOPT to print options so you can confirm that they were used by the solver
        opt.options['print_user_options'] = 'yes'

    logger.info('Using Solver: ' + solver_name)

    return opt


# Logger Setup
def setup_logger(output_dir):
    """initiates logging, sets up logger in the output directory specified

    Parameters
    ----------
    output_dir : path
        output directory path
    """
    # set up root logger
    log_path = Path(output_dir)
    if not Path.is_dir(log_path):
        Path.mkdir(log_path)
    logging.basicConfig(
        filename=f'{output_dir}/run.log',
        encoding='utf-8',
        filemode='w',
        # format='[%(asctime)s][%(name)s]' + '[%(funcName)s][%(levelname)s]  :: |%(message)s|',
        format='%(asctime)s | %(name)s | %(levelname)s :: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.DEBUG,
    )
    logging.getLogger('pyomo').setLevel(logging.WARNING)
    logging.getLogger('pandas').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


# a named tuple for common electric model index structure (EI=Electrical Index)
EI = namedtuple('EI', ['region', 'year', 'hour'])
"""(region, year, hour)"""
HI = namedtuple('HI', ['region', 'year'])
"""(region, year)"""


def get_elec_price(instance: typing.Union['PowerModel', ConcreteModel], block=None) -> pd.DataFrame:
    """pulls hourly electricity prices from completed PowerModel and de-weights them

    Prices from the duals are weighted by the day and year weights applied in the OBJ function
    This function retrieves the prices for all hours and removes the day and annual weights to
    return raw prices (and the day weights to use as needed)

    Parameters
    ----------
    instance : PowerModel
        solved electricity model

    block: ConcreteModel
        reference to the block if the electricity model is a block within a larger model

    Returns
    -------
    pd.DataFrame
        df of raw prices and the day weights to re-apply (if needed)
        columns: [r, y, hour, day_weight, raw_price]
    """

    if block:
        c = block.demand_balance
        model = block
    else:
        c = instance.demand_balance
        model = instance

    # get electricity price duals and de-weight them (costs in the OBJ are up-weighted
    # by the day weight and year weight)
    records = []
    for index in c:
        ei = EI(*index)
        weighted_value = float(instance.dual[c[index]])

        # gather the weights for this hour
        day = model.Map_hr_d[ei.hour]
        day_wt = model.Idaytq[day]
        year_wt = model.year_weights[ei.year]

        # remove the weighting & record
        unweighted_cost = weighted_value / day_wt / year_wt
        records.append((*ei, day_wt, unweighted_cost))

    res = pd.DataFrame.from_records(
        data=records, columns=['r', 'y', 'hr', 'day_weight', 'raw_price']
    )
    return res


def get_annual_wt_avg(elec_price: pd.DataFrame) -> dict[HI, float]:
    """takes annual weighted average of hourly electricity prices

    Parameters
    ----------
    elec_price : pd.DataFrame
        hourly electricity prices

    Returns
    -------
    dict[HI, float]
        annual weighted average electricity prices
    """

    def my_agg(x):
        names = {
            'weighted_ave_price': (x['day_weight'] * x['raw_price']).sum() / x['day_weight'].sum()
        }
        return pd.Series(names, index=['weighted_ave_price'])

    # find annual weighted average, weight by day weights
    elec_price_ann = elec_price.groupby(['r', 'y']).apply(my_agg)

    return elec_price_ann


def regional_annual_prices(
    m: typing.Union['PowerModel', ConcreteModel], block=None
) -> dict[HI, float]:
    """pulls all regional annual weighted electricity prices

    Parameters
    ----------
    m : typing.Union['PowerModel', ConcreteModel]
        solved PowerModel
    block :  optional
        solved block model if applicable, by default None

    Returns
    -------
    dict[HI, float]
        dict with regional annual electricity prices
    """
    ep = get_elec_price(m, block)
    ap = get_annual_wt_avg(ep)

    # convert from dataframe to dictionary
    lut = {}
    for r in ap.to_records():
        region, year, price = r
        lut[HI(region=region, year=year)] = price

    return lut


def convert_elec_price_to_lut(prices: list[tuple[EI, float]]) -> dict[EI, float]:
    """convert electricity prices to dictionary, look up table

    Parameters
    ----------
    prices : list[tuple[EI, float]]
        list of prices

    Returns
    -------
    dict[EI, float]
        dict of prices
    """
    res = {}
    for row in prices:
        ei, price = row
        res[ei] = price
    return res


def poll_hydrogen_price(
    model: typing.Union['H2Model', ConcreteModel], block=None
) -> list[tuple[HI, float]]:
    """Retrieve the price of H2 from the H2 model

    Parameters
    ----------
    model : H2Model
        the model to poll
    block: optional
        block model to poll

    Returns
    -------
    list[tuple[HI, float]]
        list of H2 Index, price tuples
    """
    # ensure valid class
    if not isinstance(model, ConcreteModel):
        raise ValueError('invalid input')

    # TODO:  what should happen if there is no entry for a particular region (no hubs)?
    if block:
        demand_constraint = block.demand_constraint
    else:
        demand_constraint = model.demand_constraint
    # print('************************************\n')
    # print(list(demand_constraint.index_set()))
    # print(list(model.dual.keys()))

    rows = [(HI(*k), model.dual[demand_constraint[k]]) for k, v in demand_constraint.items()]  # type: ignore
    logger.debug('current h2 prices:  %s', rows)
    return rows  # type: ignore


def convert_h2_price_records(records: list[tuple[HI, float]]) -> dict[HI, float]:
    """simple coversion from list of records to a dictionary LUT
    repeat entries should not occur and will generate an error"""
    res = {}
    for hi, price in records:
        if hi in res:
            logger.error('Duplicate index for h2 price received in coversion: %s', hi)
            raise ValueError('duplicate index received see log file.')
        res[hi] = price

    return res


def poll_year_avg_elec_price(price_list: list[tuple[EI, float]]) -> dict[HI, float]:
    """retrieve a REPRESENTATIVE price at the annual level from a listing of prices

    This function computes the AVERAGE elec price for each region-year combo

    Parameters
    ----------
    price_list : list[tuple[EI, float]]
        input price list

    Returns
    -------
    dict[HI, float]
        a dictionary of (region, year): price
    """
    year_region_records = defaultdict(list)
    res = {}
    for ei, price in price_list:
        year_region_records[HI(region=ei.region, year=ei.year)].append(price)

    # now gather the averages...
    for hi in year_region_records:
        res[hi] = sum(year_region_records[hi]) / len(year_region_records[hi])

    logger.debug('Computed these region-year averages for elec price: \n\t %s', res)
    return res


def poll_h2_prices_from_elec(
    model: 'PowerModel', tech, regions: typing.Iterable
) -> dict[typing.Any, float]:
    """poll the step-1 H2 price currently in the model for region/year, averaged over any steps"""
    res = {}
    for idx in model.H2Price:
        reg, s, t, step, y = idx
        if t == tech and reg in regions and step == 1:  # TODO:  remove hard coding
            res[reg, s, y] = value(model.H2Price[idx])

    return res


def update_h2_prices(model: 'PowerModel', h2_prices: dict[HI, float]) -> None:
    """Update the H2 prices held in the model

    Parameters
    ----------
    h2_prices : list[tuple[HI, float]]
        new prices
    """

    # TODO:  Fix this hard-coding below!
    h2_techs = {5}  # temp hard-coding of the tech who's price we're going to set

    update_count = 0
    no_update = set()
    good_updates = set()
    for region, season, pt, step, yr in model.H2Price:  # type: ignore
        if pt in h2_techs:
            if (region, yr) in h2_prices:
                model.H2Price[region, season, pt, step, yr] = h2_prices[HI(region=region, year=yr)]
                update_count += 1
                good_updates.add((region, yr))
            else:
                no_update.add((region, yr))
    logger.debug('Updated %d H2 prices: %s', update_count, good_updates)

    # check for any missing data
    if no_update:
        logger.warning('No new price info for region-year combos: %s', no_update)


def update_elec_demand(self, elec_demand: dict[HI, float]) -> None:
    """
    Update the external electical demand parameter with demands from the H2 model

    Parameters
    ----------
    elec_demand : dict[HI, float]
        the new demands broken out by hyd index (region, year)
    """
    # this is kind of a 1-liner right now, but may evolve into something more elaborate when
    # time scale is tweaked

    self.FixedElecRequest.store_values(elec_demand)
    logger.debug('Stored new fixed electrical request in elec model: %s', elec_demand)


def poll_h2_demand(model: 'PowerModel') -> dict[HI, float]:
    """
    Get the hydrogen demand by rep_year and region

    Use the Generation variable for h2 techs

    NOTE:  Not sure about day weighting calculation here!!

    Returns
    -------
    dict[HI, float]
        dictionary of prices by H2 Index: price
    """
    h2_consuming_techs = {5}  # TODO:  get rid of this hard-coding

    # gather results
    res: dict[HI, float] = defaultdict(float)
    tot_by_rep_year = defaultdict(float)
    # iterate over the Generation variable and screen out the H2 "demanders"
    # dimensional analysis for H2 demand:
    #
    # Gwh * kg/Gwh = kg
    # so we need 1/heat_rate for kg/Gwh
    for idx in model.generation_total.index_set():
        tech, y, reg, step, hr = idx
        if tech in h2_consuming_techs:
            h2_demand_weighted = (
                value(model.generation_total[idx])
                * model.Idaytq[model.Map_hr_d[hr]]
                / model.H2_heatrate
            )
            res[HI(region=reg, year=y)] += h2_demand_weighted
            tot_by_rep_year[y] += h2_demand_weighted

    logger.debug('Calculated cumulative H2 demand by year as: %s', tot_by_rep_year)
    return res


def create_temporal_mapping(sw_temporal):
    """Combines the input mapping files within the electricity model to create a master temporal
    mapping dataframe. The df is used to build multiple temporal parameters used within the  model.
    It creates a single dataframe that has 8760 rows for each hour in the year.
    Each hour in the year is assigned a season type, day type, and hour type used in the model.
    This defines the number of time periods the model will use based on cw_s_day and cw_hr inputs.

    Returns
    -------
    dataframe
        a dataframe with 8760 rows that include each hour, hour type, day, day type, and season.
        It also includes the weights for each day type and hour type.
    """

    # Temporal Sets - read data
    # SD = season/day; hr = hour
    data_root = Path(PROJECT_ROOT, 'src/integrator/input')
    if sw_temporal == 'default':
        sd_file = pd.read_csv(data_root / 'cw_s_day.csv')
        hr_file = pd.read_csv(data_root / 'cw_hr.csv')
    else:
        cw_s_day = 'cw_s_day_' + sw_temporal + '.csv'
        cw_hr = 'cw_hr_' + sw_temporal + '.csv'
        sd_file = pd.read_csv(data_root / 'temporal_mapping' / cw_s_day)
        hr_file = pd.read_csv(data_root / 'temporal_mapping' / cw_hr)

    # set up mapping for seasons and days
    df1 = sd_file
    df4 = df1.groupby(by=['Map_day'], as_index=False).count()
    df4 = df4.rename(columns={'Index_day': 'Dayweights'}).drop(columns=['Map_s'])
    df1 = pd.merge(df1, df4, how='left', on=['Map_day'])

    # set up mapping for hours
    df2 = hr_file
    df3 = df2.groupby(by=['Map_hr'], as_index=False).count()
    df3 = df3.rename(columns={'Index_hr': 'Hr_weights'})
    df2 = pd.merge(df2, df3, how='left', on=['Map_hr'])

    # combine season, day, and hour mapping
    df = pd.merge(df1, df2, how='cross')
    df['hr'] = df.index
    df['hr'] = df['hr'] + 1
    df['Map_hr'] = (df['Map_day'] - 1) * df['Map_hr'].max() + df['Map_hr']
    # df.to_csv(data_root/'temporal_map.csv',index=False)

    return df
