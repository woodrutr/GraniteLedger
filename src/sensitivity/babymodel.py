"""Baby Model
This file contains the TestBabyModel class, which is a subclass of ConcreteModel, along with scripts that
help generate the model parameters and structure.

The model structure is randomly generated through the functions generate and connect_subregions, with
model parameters as input.

"""
###################################################################################################
# Setup

# Import packages
import pyomo.environ as pyo
from sensitivity_tools import *
import random as rnd

# Establish logger
logger = getLogger(__name__)

###################################################################################################


class TestBabyModel(pyo.ConcreteModel):
    def __init__(
        self,
        params,
        start_year=2020,
        end_year=2025,
        price_growth_rate=1.2,
        demand_growth_rate=1.1,
        capex=72.3,
    ):
        """Toy dispatch model class containing regions, regional demand for fuel type,
        hubs located in regions, transportation arcs between hubs, and capacity expansion.

        The model finds minimum cost to produce and transport across all hubs and arcs to
        satisfy regional demand, for years from start_year to end_year.


        Parameters
        ----------

        params : dict
            dictionary with str keys
        start_year : int, optional
            the start year for the solve. Defaults to 2020.
        end_year : int, optional
            end year for the solve. Defaults to 2025.
        price_growth_rate : float, optional
            This is the factor prices grow by annually. Defaults to 1.2.
        demand_growth_rate : float, optional
            this is the factor demand grows by annually. Defaults to 1.1.

        Returns
        -------
        obj
            instance of class
        """

        pyo.ConcreteModel.__init__(self)

        # extract data from inputs
        self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        self.base_elec_price = params['elec_price']
        self.prod_capacity = params['prod_capacity']
        self.trans_cap = params['trans_capacity']
        self.start_year = start_year
        self.end_year = end_year
        self.region_map = params['region_map']
        self.hub_map = params['hub_map']
        self.outbound = params['outbound']
        self.inbound = params['inbound']
        demand = params['base_demand']
        electricity_consumption = params['base_elec_consumption']
        transport_cost = params['base_transport_cost']
        region_list = params['region_list']
        hublist = params['hublist']
        arcs = params['arcs']

        #
        # Sets
        # ----

        self.regions = pyo.Set(initialize=region_list)
        self.hubs = pyo.Set(initialize=hublist)
        self.arcs = pyo.Set(initialize=arcs)
        self.years = pyo.RangeSet(start_year, end_year)

        #
        # Parameters
        # ----------

        self.price_growth_rate = pyo.Param(mutable=True, initialize=price_growth_rate)
        self.demand_growth_rate = pyo.Param(mutable=True, initialize=demand_growth_rate)
        self.electricity_consumption = pyo.Param(mutable=True, initialize=electricity_consumption)
        self.transportation_cost = pyo.Param(mutable=True, initialize=transport_cost)
        self.base_demand = pyo.Param(self.regions, mutable=True, initialize=demand)
        self.capex = pyo.Param(mutable=True, initialize=capex)

        def get_elec_price(model, region, year):
            return model.base_elec_price[region] * model.price_growth_rate ** (
                year - self.start_year
            )

        self.electricity_price = pyo.Param(
            self.regions, self.years, mutable=True, initialize=get_elec_price
        )

        self.production_capacity = pyo.Param(self.hubs, mutable=True, initialize=self.prod_capacity)
        self.transportation_capacity = pyo.Param(self.arcs, mutable=True, initialize=self.trans_cap)

        def get_demand(model, region, year):
            return model.base_demand[region] * (
                model.demand_growth_rate ** (year - self.start_year)
            )

        self.demand = pyo.Param(self.regions, self.years, mutable=True, initialize=get_demand)

        #
        # Variables
        # ---------

        self.production = pyo.Var(self.hubs, self.years)
        self.transportation = pyo.Var(self.arcs, self.years)
        self.cap_expansion = pyo.Var(self.hubs, self.years)

        #
        # Constraints
        # -----------

        # Production Constraints

        def max_production(model, hub, year):
            earlier_years = [y for y in self.years if y < year]
            return (
                model.production[hub, year]
                - model.production_capacity[hub]
                - sum(model.cap_expansion[hub, year] for year in earlier_years)
                <= 0
            )

        self.max_production = pyo.Constraint(self.hubs, self.years, expr=max_production)

        def min_production(model, hub, year):
            return -model.production[hub, year] <= 0

        self.min_production = pyo.Constraint(self.hubs, self.years, rule=min_production)

        # Transportation Constraints

        def max_transportation(model, origin, destination, year):
            return (
                model.transportation[origin, destination, year]
                - model.transportation_capacity[origin, destination]
                <= 0
            )

        self.max_transportation = pyo.Constraint(self.arcs, self.years, rule=max_transportation)

        def min_transportation(model, origin, destination, year):
            return -model.transportation[origin, destination, year] <= 0

        self.min_transportation = pyo.Constraint(self.arcs, self.years, rule=min_transportation)

        # cap_expansion constraints

        def min_cap_expansion(model, hub, year):
            return -model.cap_expansion[hub, year] <= 0

        self.min_cap_expansion = pyo.Constraint(self.hubs, self.years, rule=min_cap_expansion)

        # Demand Constraint

        def demand_constraint(model, region, year):
            # the sum of production plus imports minus exports for each region must sum to demand
            return (
                sum(
                    model.production[hub, year]
                    + sum(model.transportation[arc, year] for arc in model.inbound[hub])
                    - sum(model.transportation[arc, year] for arc in model.outbound[hub])
                    for hub in model.region_map[region]
                )
                - model.demand[region, year]
                == 0
            )

        self.demand_constraint = pyo.Constraint(self.regions, self.years, expr=demand_constraint)

        #
        # Objective
        # ---------

        def transportation_cost(model):
            return sum(
                model.transportation[arc, year] * model.transportation_cost
                for arc in model.arcs
                for year in model.years
            )

        def production_cost(model):
            return sum(
                model.production[hub, year]
                * model.electricity_price[self.hub_map[hub], year]
                * model.electricity_consumption
                for hub in model.hubs
                for year in model.years
            )

        def capex_cost(model):
            return sum(
                model.cap_expansion[hub, year] * model.capex
                for hub in model.hubs
                for year in model.years
            )

        def total_cost(model):
            return transportation_cost(model) + production_cost(model) + capex_cost(model)

        self.total_cost = pyo.Objective(rule=total_cost)

    def solve(self):
        solution = pyo.SolverFactory('appsi_highs').solve(self, tee=True)

        return solution


def generate(
    num_regions=3,
    hubs_per_region=2,
    base_elec_price=5.0,
    base_prod_capacity=5000,
    demand_fraction=0.7,
    base_transport_cost=22.3,
    base_elec_consumption=9.8,
):
    """generates a random network with all parameters required to initialize a ToyBabyModel

    Parameters
    ----------

    num_regions : int, optional
        number of regions. Defaults to 3.
    hubs_per_region : int, optional
        number of hubs per region. Defaults to 2.
    base_elec_price : float, optional
        the average electricity price in all regions. Defaults to 5.0.
    base_prod_capacity : int, optional
        the average production capacity for all hubs. Defaults to 5000.
    demand_fraction : float, optional
        the average fraction of capacity initial demand is set to. Defaults to 0.7.
    base_transport_cost : float, optional
        the base transportation cost for all arcs. Defaults to 22.3.
    base_elec_consumption : float, optional
        the electricity consumption rate for production. Defaults to 9.8.

    Returns
    -------
    hublist : list
    region_list : list
    hub_map : dict
    region_map : dict
    elec_price : dict
    prod_capacity : dict
    demand : dict
    base_elec_consumption : float
    base_transport_cost : float
    """
    hublist = []
    hub_map = {}
    region_map = {}
    elec_price = {}
    prod_capacity = {}

    region_list = ['region_' + str(i) for i in range(num_regions)]

    for region in region_list:
        new_hubs = [region + '_hub_' + str(i) for i in range(hubs_per_region)]
        hub_map.update({hub: region for hub in new_hubs})
        region_map.update({region: new_hubs})
        hublist = hublist + new_hubs

    for region in region_list:
        elec_price.update({region: base_elec_price * (0.5 + rnd.random())})

    for hub in hublist:
        prod_capacity.update({hub: base_prod_capacity * (0.5 + rnd.random())})

    demand = demand_fraction * hubs_per_region * base_prod_capacity

    return (
        hublist,
        region_list,
        hub_map,
        region_map,
        elec_price,
        prod_capacity,
        demand,
        base_elec_consumption,
        base_transport_cost,
    )


def connect_regions(region_map, hubmap, base_trans_cap):
    """given a mapping of regions to lists of hubs, and hubs to regions, creates
    a set of arcs between hubs such that:

    1. the grid is connected
    2. each region has a main hub that all other hubs in the region are connected to

    Parameters
    ----------

    region_map : dict
        dictionary of region names to lists of hubs
    hubmap : dict
        dictionary of hub names to their parent region
    base_trans_cap : float
        base transportation capacity for arcs

    Returns
    -------
    arcs : list
        list of tuples of hubs, representing start and endpoints
    outbound : dict
        dictionary of hub:list of arcs originating from hub
    inbound : dict
        dictionary of hub:list of arcs terminating at hub
    trans_capacity : dict
        dictionary of arcs:transportation capacity of arc
    """

    main_hubs = []
    arcs = []
    outbound = {hub: [] for hub in hubmap.keys()}
    inbound = {hub: [] for hub in hubmap.keys()}
    trans_capacity = {}

    for region in region_map.keys():
        main_hubs.append(region_map[region][0])

    for i in range(1, len(main_hubs)):
        start = main_hubs[i]
        end = rnd.choice(main_hubs[0:i])
        arcs = arcs + [(start, end), (end, start)]
        trans_capacity[(start, end)] = base_trans_cap
        trans_capacity[(end, start)] = base_trans_cap

        outbound[start].append((start, end))
        inbound[start].append((end, start))

        outbound[end].append((end, start))
        inbound[end].append((start, end))

    for mainhub in main_hubs:
        region = hubmap[mainhub]
        for hub in region_map[region]:
            if hub != mainhub:
                start = hub
                end = mainhub

                arcs = arcs + [(start, end), (end, start)]
                trans_capacity[(start, end)] = base_trans_cap
                trans_capacity[(end, start)] = base_trans_cap

                outbound[start].append((start, end))
                inbound[start].append((end, start))

                outbound[end].append((end, start))
                inbound[end].append((start, end))

    return arcs, outbound, inbound, trans_capacity
