"""Electricity Model.
This file contains the PowerModel class which contains a pyomo optimization model of the electric
power sector. The class is organized by sections: settings, sets, parameters, variables, objective
function, constraints, plus additional misc support functions.
"""
###################################################################################################
# Setup

# Import packages
from collections import defaultdict
from logging import getLogger
import pyomo.environ as pyo

# Import scripts
from src.integrator.utilities import HI
import src.models.electricity.scripts.utilities as f

# Establish logger
logger = getLogger(__name__)

###################################################################################################
# MODEL


class PowerModel(pyo.ConcreteModel):
    """A PowerModel instance. Builds electricity pyomo model.

    Parameters
    ----------
    all_frames : dictionary of pd.DataFrames
        Contains all dataframes of inputs
    setA : Sets
        Contains all other non-dataframe inputs
    """

    def __init__(self, all_frames, setA, *args, **kwargs):
        pyo.ConcreteModel.__init__(self, *args, **kwargs)

        ###########################################################################################
        # Settings

        self.sw_trade = setA.sw_trade
        self.sw_expansion = setA.sw_expansion
        self.sw_agg_years = setA.sw_agg_years
        self.sw_rm = setA.sw_rm
        self.sw_ramp = setA.sw_ramp
        self.sw_reserves = setA.sw_reserves
        self.sw_h2int = 0

        # 0=no learning, 1=linear iterations, 2=nonlinear learning
        self.sw_learning = setA.sw_learning

        # dictionary to lookup column names for sets, params, variables
        self.cols_dict = {}

        ###########################################################################################
        # Sets

        # temporal sets
        self.hr = pyo.Set(initialize=setA.hr)
        self.day = pyo.Set(initialize=setA.day)
        self.y = pyo.Set(initialize=setA.years)
        self.s = pyo.Set(initialize=setA.s)

        # spatial sets
        self.r = pyo.Set(initialize=setA.r)
        self.r_int = pyo.Set(initialize=setA.r_int)
        self.trade_regs = pyo.Set(initialize=setA.trade_regs)
        self.r_int_conn = pyo.Set(initialize=setA.r_int_conn)

        # Load sets
        self.demand_balance_index = f.declare_set(self, 'demand_balance_index', all_frames['Load'])
        self.unmet_load_index = self.r * self.y * self.hr
        self.cols_dict['unmet_load_index'] = ['unmet_load_index', 'r', 'y', 'hr']
        self.restypes = pyo.Set(initialize=setA.restypes)

        # Supply price and quantity sets and subsets
        self.capacity_total_index = f.declare_set(
            self, 'capacity_total_index', all_frames['SupplyCurve']
        )
        self.generation_total_index = f.declare_set(
            self, 'generation_total_index', setA.generation_total_index
        )
        self.generation_dispatchable_ub_index = f.declare_set(
            self, 'generation_dispatchable_ub_index', setA.generation_dispatchable_ub_index
        )
        self.Storage_index = f.declare_set(self, 'Storage_index', setA.Storage_index)
        self.H2Gen_index = f.declare_set(self, 'H2Gen_index', setA.H2Gen_index)
        self.generation_hydro_ub_index = f.declare_set(
            self, 'generation_hydro_ub_index', setA.generation_hydro_ub_index
        )
        self.ramp_most_hours_balance_index = f.declare_set(
            self, 'ramp_most_hours_balance_index', setA.ramp_most_hours_balance_index
        )
        self.ramp_first_hour_balance_index = f.declare_set(
            self, 'ramp_first_hour_balance_index', setA.ramp_first_hour_balance_index
        )
        self.storage_most_hours_balance_index = f.declare_set(
            self, 'storage_most_hours_balance_index', setA.storage_most_hours_balance_index
        )
        self.storage_first_hour_balance_index = f.declare_set(
            self, 'storage_first_hour_balance_index', setA.storage_first_hour_balance_index
        )
        self.capacity_hydro_ub_index = f.declare_set(
            self, 'capacity_hydro_ub_index', setA.capacity_hydro_ub_index
        )

        # Other technology sets
        self.HydroCapFactor_index = f.declare_set(
            self, 'HydroCapFactor_index', all_frames['HydroCapFactor']
        )
        self.generation_vre_ub_index = f.declare_set(
            self, 'generation_vre_ub_index', all_frames['SolWindCapFactor']
        )
        self.H2Price_index = f.declare_set(self, 'H2Price_index', all_frames['H2Price'])

        for tss in setA.pt_subset_names:
            # create the technology subsets based on the pt_subsets input
            setattr(self, tss, pyo.Set(initialize=getattr(setA, tss)))

        # if capacity expansion is on
        if self.sw_expansion:
            self.capacity_builds_index = f.declare_set(
                self, 'capacity_builds_index', all_frames['CapCost']
            )
            self.FOMCost_index = f.declare_set(self, 'FOMCost_index', all_frames['FOMCost'])
            self.Build_index = f.declare_set(self, 'Build_index', setA.Build_index)
            self.CapacityCredit_index = f.declare_set(
                self, 'CapacityCredit_index', all_frames['CapacityCredit']
            )
            self.capacity_retirements_index = f.declare_set(
                self, 'capacity_retirements_index', setA.capacity_retirements_index
            )

            # if capacity expansion and learning are on
            if self.sw_learning > 0:
                self.LearningRate_index = f.declare_set(
                    self, 'LearningRate_index', all_frames['LearningRate']
                )
                self.CapCostInitial_index = f.declare_set(
                    self, 'CapCostInitial_index', all_frames['CapCostInitial']
                )
                self.SupplyCurveLearning_index = f.declare_set(
                    self, 'SupplyCurveLearning_index', all_frames['SupplyCurveLearning']
                )

        # if trade operation is on
        if self.sw_trade:
            self.TranCost_index = f.declare_set(self, 'TranCost_index', all_frames['TranCost'])
            self.TranLimit_index = f.declare_set(self, 'TranLimit_index', all_frames['TranLimit'])
            self.trade_interregional_index = f.declare_set(
                self, 'trade_interregional_index', setA.trade_interregional_index
            )
            self.TranCostInt_index = f.declare_set(
                self, 'TranCostInt_index', all_frames['TranCostInt']
            )
            self.TranLimitInt_index = f.declare_set(
                self, 'TranLimitInt_index', all_frames['TranLimitGenInt']
            )
            self.trade_interational_index = f.declare_set(
                self, 'trade_interational_index', setA.trade_interational_index
            )
            self.TranLineLimitInt_index = f.declare_set(
                self, 'TranLineLimitInt_index', all_frames['TranLimitCapInt']
            )

        # if ramping requirements are on
        if self.sw_ramp:
            self.RampUpCost_index = f.declare_set(
                self, 'RampUpCost_index', all_frames['RampUpCost']
            )
            self.RampRate_index = f.declare_set(self, 'RampRate_index', all_frames['RampRate'])
            self.generation_ramp_index = f.declare_set(
                self, 'generation_ramp_index', setA.generation_ramp_index
            )

        # if operating reserve requirements are on
        if self.sw_reserves:
            self.reserves_procurement_index = f.declare_set(
                self, 'reserves_procurement_index', setA.reserves_procurement_index
            )
            self.RegReservesCost_index = f.declare_set(
                self, 'RegReservesCost_index', all_frames['RegReservesCost']
            )
            self.ResTechUpperBound_index = f.declare_set(
                self, 'ResTechUpperBound_index', all_frames['ResTechUpperBound']
            )

        ###########################################################################################
        # Parameters

        # temporal parameters
        self.y0 = f.declare_param(self, 'y0', None, setA.start_year)
        self.num_hr_day = f.declare_param(self, 'num_hr_day', None, setA.num_hr_day)
        self.Map_hr_s = f.declare_param(self, 'Map_hr_s', self.hr, all_frames['Map_hr_s'])
        self.Map_hr_d = f.declare_param(self, 'Map_hr_d', self.hr, all_frames['Map_hr_d']['day'])
        self.year_weights = f.declare_param(
            self, 'year_weights', self.y, all_frames['year_weights']
        )
        self.Hr_weights = f.declare_param(
            self, 'Hr_weights', self.hr, all_frames['Hr_weights']['Hr_weights']
        )
        self.Idaytq = f.declare_param(self, 'Idaytq', self.day, all_frames['Idaytq'])
        self.WeightSeason = f.declare_param(
            self, 'WeightSeason', self.s, all_frames['WeightSeason']
        )

        # load and technology parameters
        self.Load = f.declare_param(
            self, 'Load', self.demand_balance_index, all_frames['Load'], mutable=True
        )
        self.UnmetLoadPenalty = f.declare_param(
            self, 'UnmetLoadPenalty', None, 500000
        )  # 500 $/MWh -> 500,000 $/GWh
        self.SupplyPrice = f.declare_param(
            self, 'SupplyPrice', self.capacity_total_index, all_frames['SupplyPrice']
        )
        self.SupplyCurve = f.declare_param(
            self, 'SupplyCurve', self.capacity_total_index, all_frames['SupplyCurve']
        )
        self.SolWindCapFactor = f.declare_param(
            self, 'SolWindCapFactor', self.generation_vre_ub_index, all_frames['SolWindCapFactor']
        )
        self.HydroCapFactor = f.declare_param(
            self, 'HydroCapFactor', self.HydroCapFactor_index, all_frames['HydroCapFactor']
        )
        self.BatteryEfficiency = f.declare_param(
            self, 'BatteryEfficiency', setA.pts, all_frames['BatteryEfficiency']
        )
        self.HourstoBuy = f.declare_param(self, 'HourstoBuy', setA.pts, all_frames['HourstoBuy'])
        self.H2Price = f.declare_param(
            self, 'H2Price', self.H2Price_index, all_frames['H2Price'], mutable=True
        )
        self.StorageLevelCost = f.declare_param(self, 'StorageLevelCost', None, 0.00000001)
        self.H2_heatrate = f.declare_param(self, 'H2_heatrate', None, setA.H2_heatrate)

        # if capacity expansion is on
        if self.sw_expansion:
            self.FOMCost = f.declare_param(
                self, 'FOMCost', self.FOMCost_index, all_frames['FOMCost']
            )
            self.CapacityCredit = f.declare_param(
                self, 'CapacityCredit', self.CapacityCredit_index, all_frames['CapacityCredit']
            )

            # if capacity expansion and learning are on
            if self.sw_learning > 0:
                self.LearningRate = f.declare_param(
                    self, 'LearningRate', self.LearningRate_index, all_frames['LearningRate']
                )
                self.CapCostInitial = f.declare_param(
                    self, 'CapCostInitial', self.CapCostInitial_index, all_frames['CapCostInitial']
                )
                self.SupplyCurveLearning = f.declare_param(
                    self,
                    'SupplyCurveLearning',
                    self.SupplyCurveLearning_index,
                    all_frames['SupplyCurveLearning'],
                )

            # if learning is not to be solved nonlinearly directly in the obj
            if self.sw_learning < 2:
                if self.sw_learning == 0:
                    mute = False
                else:
                    mute = True
                self.CapCostLearning = f.declare_param(
                    self,
                    'CapCostLearning',
                    self.capacity_builds_index,
                    all_frames['CapCost'],
                    mutable=mute,
                )

        # if trade operation is on
        if self.sw_trade:
            self.TransLoss = f.declare_param(self, 'TransLoss', None, setA.TransLoss)
            self.TranCost = f.declare_param(
                self, 'TranCost', self.TranCost_index, all_frames['TranCost']
            )
            self.TranLimit = f.declare_param(
                self, 'TranLimit', self.TranLimit_index, all_frames['TranLimit']
            )
            self.TranCostInt = f.declare_param(
                self, 'TranCostInt', self.TranCostInt_index, all_frames['TranCostInt']
            )
            self.TranLimitGenInt = f.declare_param(
                self, 'TranLimitGenInt', self.TranLimitInt_index, all_frames['TranLimitGenInt']
            )
            self.TranLimitCapInt = f.declare_param(
                self, 'TranLimitCapInt', self.TranLineLimitInt_index, all_frames['TranLimitCapInt']
            )

        # if reserve margin requirements are on
        if self.sw_rm:
            self.ReserveMargin = f.declare_param(
                self, 'ReserveMargin', self.r, all_frames['ReserveMargin']
            )

        # if ramping requirements are on
        if self.sw_ramp:
            self.RampUpCost = f.declare_param(
                self, 'RampUpCost', self.RampUpCost_index, all_frames['RampUpCost']
            )
            self.RampDownCost = f.declare_param(
                self, 'RampDownCost', self.RampUpCost_index, all_frames['RampDownCost']
            )
            self.RampRate = f.declare_param(
                self, 'RampRate', self.RampRate_index, all_frames['RampRate']
            )

        # if operating reserve requirements are on
        if self.sw_reserves:
            self.RegReservesCost = f.declare_param(
                self, 'RegReservesCost', self.RegReservesCost_index, all_frames['RegReservesCost']
            )
            self.ResTechUpperBound = f.declare_param(
                self,
                'ResTechUpperBound',
                self.ResTechUpperBound_index,
                all_frames['ResTechUpperBound'],
            )

        ##########################
        # Cross-talk from H2 model
        # TODO: fit these into the declare param format for consistency
        self.FixedElecRequest = pyo.Param(
            self.r,
            self.y,
            domain=pyo.NonNegativeReals,
            initialize=0,
            mutable=True,
            doc='a known fixed request from H2',
        )
        self.var_elec_request = pyo.Var(
            self.r,
            self.y,
            domain=pyo.NonNegativeReals,
            initialize=0,
            doc='variable request from H2',
        )

        ###########################################################################################
        # Variables

        # Generation, capacity, and technology variables
        self.generation_total = f.declare_var(self, 'generation_total', self.generation_total_index)
        self.unmet_load = f.declare_var(self, 'unmet_load', self.unmet_load_index)
        self.capacity_total = f.declare_var(self, 'capacity_total', self.capacity_total_index)
        self.storage_inflow = f.declare_var(self, 'storage_inflow', self.Storage_index)
        self.storage_outflow = f.declare_var(self, 'storage_outflow', self.Storage_index)
        self.storage_level = f.declare_var(self, 'storage_level', self.Storage_index)

        # if capacity expansion is on
        if self.sw_expansion:
            self.capacity_builds = f.declare_var(
                self, 'capacity_builds', self.capacity_builds_index
            )
            self.capacity_retirements = f.declare_var(
                self, 'capacity_retirements', self.capacity_retirements_index
            )

        # if trade operation is on
        if self.sw_trade:
            self.trade_interregional = f.declare_var(
                self, 'trade_interregional', self.trade_interregional_index
            )
            self.trade_international = f.declare_var(
                self, 'trade_international', self.trade_interational_index
            )

        # if reserve margin constraints are on
        if self.sw_rm:
            self.storage_avail_cap = f.declare_var(self, 'storage_avail_cap', self.Storage_index)

        # if ramping requirements are on
        if self.sw_ramp:
            self.generation_ramp_up = f.declare_var(
                self, 'generation_ramp_up', self.generation_ramp_index
            )
            self.generation_ramp_down = f.declare_var(
                self, 'generation_ramp_down', self.generation_ramp_index
            )

        # if operating reserve requirements are on
        if self.sw_reserves:
            self.reserves_procurement = f.declare_var(
                self, 'reserves_procurement', self.reserves_procurement_index
            )

        ###########################################################################################
        # Objective Function

        self.populate_by_hour_sets = pyo.BuildAction(rule=f.populate_by_hour_sets_rule)

        def dispatch_cost(self):
            """Dispatch cost (e.g., variable O&M cost) component for the objective function.

            Returns
            -------
            int
                Dispatch cost
            """
            return sum(
                self.Idaytq[self.Map_hr_d[hr]]
                * (
                    sum(
                        self.year_weights[y]
                        * self.SupplyPrice[(reg, s, tech, step, y)]
                        * self.generation_total[(tech, y, reg, step, hr)]
                        for (tech, y, reg, step) in self.GenHour_index[hr]
                    )
                    + sum(
                        self.year_weights[y]
                        * (
                            0.5
                            * self.SupplyPrice[(reg, s, tech, step, y)]
                            * (
                                self.storage_inflow[(tech, y, reg, step, hr)]
                                + self.storage_outflow[(tech, y, reg, step, hr)]
                            )
                            + (self.Hr_weights[hr] * self.StorageLevelCost)
                            * self.storage_level[(tech, y, reg, step, hr)]
                        )
                        for (tech, y, reg, step) in self.StorageHour_index[hr]
                    )
                    # dimensional analysis for cost:
                    # $/kg * kg/Gwh * Gwh = $
                    # so we need 1/heatrate for kg/Gwh
                    + sum(
                        self.year_weights[y]
                        * self.H2Price[reg, s, tech, step, y]
                        / self.H2_heatrate
                        * self.generation_total[(tech, y, reg, 1, hr)]
                        for (tech, y, reg, step) in self.H2GenHour_index[hr]
                    )
                )
                for hr in self.hr
                if (s := self.Map_hr_s[hr])
            )

        self.dispatch_cost = pyo.Expression(expr=dispatch_cost)

        def unmet_load_cost(self):
            """Unmet load cost component for the objective function. Should equal zero.

            Returns
            -------
            int
                Unmet load cost
            """
            return sum(
                self.Idaytq[self.Map_hr_d[hour]]
                * self.year_weights[y]
                * self.unmet_load[(reg, y, hour)]
                * self.UnmetLoadPenalty
                for (reg, y, hour) in self.unmet_load_index
            )

        self.unmet_load_cost = pyo.Expression(expr=unmet_load_cost)

        # if capacity expansion is on
        if self.sw_expansion:
            # TODO: choosing summer for capacity, may want to revisit this assumption
            def fixed_om_cost(self):
                """Fixed operation and maintenance (FOM) cost component for the objective function.

                Returns
                -------
                int
                    FOM cost component
                """
                return sum(
                    self.year_weights[y]
                    * self.FOMCost[(reg, pt, steps)]
                    * self.capacity_total[(reg, s, pt, steps, y)]
                    for (reg, s, pt, steps, y) in self.capacity_total_index
                    if s == 2
                )

            self.fixed_om_cost = pyo.Expression(expr=fixed_om_cost)

            # nonlinear expansion costs
            if self.sw_learning == 2:

                def capacity_expansion_cost(self):
                    """Capacity expansion cost component for the objective function if
                    learning switch is set to nonlinear option.

                    Returns
                    -------
                    int
                        Capacity expansion cost component (nonlinear learning)
                    """
                    return sum(
                        (
                            self.CapCostInitial[(reg, pt, step)]
                            * (
                                (
                                    (
                                        self.SupplyCurveLearning[pt]
                                        + 0.0001 * (y - self.y0)
                                        + sum(
                                            sum(
                                                self.capacity_builds[(r, pt, year, steps)]
                                                for year in self.y
                                                if year < y
                                            )
                                            for (r, tech, steps) in self.CapCostInitial_index
                                            if tech == pt
                                        )
                                    )
                                    / self.SupplyCurveLearning[pt]
                                )
                                ** (-1.0 * self.LearningRate[pt])
                            )
                        )
                        * self.capacity_builds[(reg, pt, y, step)]
                        for (reg, pt, y, step) in self.capacity_builds_index
                    )

                self.capacity_expansion_cost = pyo.Expression(expr=capacity_expansion_cost)

            # linear expansion costs
            else:

                def capacity_expansion_cost(self):
                    """Capacity expansion cost component for the objective function if
                    learning switch is set to linear option.

                    Returns
                    -------
                    int
                        Capacity expansion cost component (linear learning)
                    """
                    return sum(
                        self.CapCostLearning[(reg, pt, y, step)]
                        * self.capacity_builds[(reg, pt, y, step)]
                        for (reg, pt, y, step) in self.capacity_builds_index
                    )

                self.capacity_expansion_cost = pyo.Expression(expr=capacity_expansion_cost)

        # if trade operation is on
        if self.sw_trade:

            def trade_cost(self):
                """Interregional and international trade cost component for the objective function.

                Returns
                -------
                int
                    Interregional trade cost component
                """
                return sum(
                    self.Idaytq[self.Map_hr_d[hour]]
                    * self.year_weights[y]
                    * self.trade_interregional[(reg, reg1, y, hour)]
                    * self.TranCost[(reg, reg1, y)]
                    for (reg, reg1, y, hour) in self.trade_interregional_index
                ) + sum(
                    self.Idaytq[self.Map_hr_d[hour]]
                    * self.year_weights[y]
                    * self.trade_international[(reg, reg_can, y, CSteps, hour)]
                    * self.TranCostInt[(reg, reg_can, CSteps, y)]
                    for (reg, reg_can, y, CSteps, hour) in self.trade_interational_index
                )

            self.trade_cost = pyo.Expression(expr=trade_cost)

        # if ramping requirements are on
        if self.sw_ramp:

            def ramp_cost(self):
                """Ramping cost component for the objective function.

                Returns
                -------
                int
                    Ramping cost component
                """
                return sum(
                    self.Idaytq[self.Map_hr_d[hour]]
                    * self.year_weights[y]
                    * (
                        self.generation_ramp_up[(ptc, y, reg, step, hour)] * self.RampUpCost[ptc]
                        + self.generation_ramp_down[(ptc, y, reg, step, hour)]
                        * self.RampDownCost[ptc]
                    )
                    for (ptc, y, reg, step, hour) in self.generation_ramp_index
                )

            self.ramp_cost = pyo.Expression(expr=ramp_cost)

        # if operating reserve requirements are on
        if self.sw_reserves:

            def operating_reserves_cost(self):
                """Operating reserve cost component for the objective function.

                Returns
                -------
                int
                    Operating reserve cost component
                """
                return sum(
                    (self.RegReservesCost[pt] if restype == 2 else 0.01)
                    * self.Idaytq[self.Map_hr_d[hr]]
                    * self.year_weights[y]
                    * self.reserves_procurement[(restype, pt, y, r, steps, hr)]
                    for (restype, pt, y, r, steps, hr) in self.reserves_procurement_index
                )

            self.operating_reserves_cost = pyo.Expression(expr=operating_reserves_cost)

        # Final Objective Function
        def electricity_objective_function(self):
            """Objective function, objective is to minimize costs to the electric power system.

            Returns
            -------
            int
                Objective function
            """
            return (
                self.dispatch_cost
                + self.unmet_load_cost
                + (self.ramp_cost if self.sw_ramp else 0)
                + (self.trade_cost if self.sw_trade else 0)
                + (self.capacity_expansion_cost + self.fixed_om_cost if self.sw_expansion else 0)
                + (self.operating_reserves_cost if self.sw_reserves else 0)
            )

        self.totalCost = pyo.Objective(rule=electricity_objective_function, sense=pyo.minimize)

        ###########################################################################################
        # Constraints

        self.populate_demand_balance_sets = pyo.BuildAction(
            rule=f.populate_demand_balance_sets_rule
        )

        # Property: ShadowPrice
        @self.Constraint(self.demand_balance_index)
        def demand_balance(self, r, y, hr):
            """Demand balance constraint where Load <= Generation.

            Parameters
            ----------
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            hr : pyomo.core.base.set.OrderedScalarSet
                time segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Demand balance constraint
            """
            return self.Load[(r, y, hr)] <= sum(
                self.generation_total[(tech, y, r, step, hr)]
                for (tech, step) in self.GenSetDemandBalance[(y, r, hr)]
            ) + sum(
                self.storage_outflow[(tech, y, r, step, hr)]
                - self.storage_inflow[(tech, y, r, step, hr)]
                for (tech, step) in self.StorageSetDemandBalance[(y, r, hr)]
            ) + self.unmet_load[(r, y, hr)] + (
                sum(
                    self.trade_interregional[(r, reg1, y, hr)] * (1 - self.TransLoss)
                    - self.trade_interregional[(reg1, r, y, hr)]
                    for (reg1) in self.TradeSetDemandBalance[(y, r, hr)]
                )
                if self.sw_trade and r in self.trade_regs
                else 0
            ) + (
                sum(
                    self.trade_international[(r, r_int, y, CSteps, hr)] * (1 - self.TransLoss)
                    for (r_int, CSteps) in self.TradeCanSetDemandBalance[(y, r, hr)]
                )
                if (self.sw_trade == 1 and r in self.r_int_conn)
                else 0
            )

        # #First hour
        @self.Constraint(self.storage_first_hour_balance_index)
        def storage_first_hour_balance(self, pts, y, r, steps, hr1):
            """Storage balance constraint for the first hour time-segment in each day-type where
            Storage level == Storage level (in final hour time-segment in current day-type)
                            + Storage inflow * Battery efficiency
                            - Storage outflow

            Parameters
            ----------
            pts : pyomo.core.base.set.OrderedScalarSet
                storage technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr1 : pyomo.core.base.set.OrderedScalarSet
                set containing first hour time-segment in each day-type

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Storage balance constraint for the first hour time-segment in each day-type
            """
            return (
                self.storage_level[(pts, y, r, steps, hr1)]
                == self.storage_level[(pts, y, r, steps, hr1 + self.num_hr_day - 1)]
                + self.BatteryEfficiency[pts] * self.storage_inflow[(pts, y, r, steps, hr1)]
                - self.storage_outflow[(pts, y, r, steps, hr1)]
            )

        # #Not first hour
        @self.Constraint(self.storage_most_hours_balance_index)
        def storage_most_hours_balance(self, pts, y, r, steps, hr23):
            """Storage balance constraint for the time-segment in each day-type other than
            the first hour time-segment where
            Storage level == Storage level (in previous hour time-segment)
                            + Storage inflow * Battery efficiency
                            - Storage outflow

            Parameters
            ----------
            pts : pyomo.core.base.set.OrderedScalarSet
                storage technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr23 : pyomo.core.base.set.OrderedScalarSet
                set containing time-segment except first hour in each day-type

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Storage balance constraint for the time-segment in each day-type other than
            the first hour time-segment
            """
            return (
                self.storage_level[(pts, y, r, steps, hr23)]
                == self.storage_level[(pts, y, r, steps, hr23 - 1)]
                + self.BatteryEfficiency[pts] * self.storage_inflow[(pts, y, r, steps, hr23)]
                - self.storage_outflow[(pts, y, r, steps, hr23)]
            )

        self.populate_hydro_sets = pyo.BuildAction(rule=f.populate_hydro_sets_rule)

        @self.Constraint(self.capacity_hydro_ub_index)
        def capacity_hydro_ub(self, pth, y, r, s):
            """hydroelectric generation seasonal upper bound where
            Hydo generation <= Hydo capacity * Hydro capacity factor

            Parameters
            ----------
            pth : pyomo.core.base.set.OrderedScalarSet
                hydro technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            s : pyomo.core.base.set.OrderedScalarSet
                season set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                hydroelectric generation seasonal upper bound
            """
            return (
                sum(
                    self.generation_total[pth, y, r, 1, hr] * self.Idaytq[self.Map_hr_d[hr]]
                    for hr in self.HourSeason_index[s]
                )
                <= self.capacity_total[(r, s, pth, 1, y)]
                * self.HydroCapFactor[r, s]
                * self.WeightSeason[s]
            )

        @self.Constraint(self.generation_dispatchable_ub_index)
        def generation_dispatchable_ub(self, ptd, y, r, steps, hr):
            """Dispatchable generation upper bound where
            Dispatchable generation + reserve procurement <= capacity * capacity factor

            Parameters
            ----------
            ptd : pyomo.core.base.set.OrderedScalarSet
                dispatchable technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Dispatchable generation upper bound
            """
            return (
                self.generation_total[(ptd, y, r, steps, hr)]
                + (
                    sum(
                        self.reserves_procurement[(restype, ptd, y, r, steps, hr)]
                        for restype in self.restypes
                    )
                    if self.sw_reserves
                    else 0
                )
                <= self.capacity_total[(r, self.Map_hr_s[hr], ptd, steps, y)] * self.Hr_weights[hr]
            )

        @self.Constraint(self.generation_hydro_ub_index)
        def generation_hydro_ub(self, pth, y, r, steps, hr):
            """Hydroelectric generation upper bound where
            Hydroelectric generation + reserve procurement <= capacity * capacity factor

            Parameters
            ----------
            pth : pyomo.core.base.set.OrderedScalarSet
                hydro technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Hydroelectric generation upper bound
            """
            return (
                self.generation_total[(pth, y, r, steps, hr)]
                + sum(
                    self.reserves_procurement[(restype, pth, y, r, steps, hr)]
                    for restype in self.restypes
                )
                if self.sw_reserves
                else 0
            ) <= self.capacity_total[(r, self.Map_hr_s[hr], pth, steps, y)] * self.HydroCapFactor[
                (r, self.Map_hr_s[hr])
            ] * self.Hr_weights[hr]

        @self.Constraint(self.generation_vre_ub_index)
        def generation_vre_ub(self, pti, y, r, steps, hr):
            """Intermittent generation upper bound where
            Intermittent generation + reserve procurement <= capacity * capacity factor

            Parameters
            ----------
            pti : pyomo.core.base.set.OrderedScalarSet
                intermittent technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                intermittent generation upper bound
            """
            return (
                self.generation_total[(pti, y, r, steps, hr)]
                + (
                    sum(
                        self.reserves_procurement[(restype, pti, y, r, steps, hr)]
                        for restype in self.restypes
                    )
                    if self.sw_reserves
                    else 0
                )
                <= self.capacity_total[(r, self.Map_hr_s[hr], pti, steps, y)]
                * self.SolWindCapFactor[(pti, y, r, steps, hr)]
                * self.Hr_weights[hr]
            )

        @self.Constraint(self.Storage_index)
        def storage_inflow_ub(self, pt, y, r, steps, hr):
            """Storage inflow upper bound where
            Storage inflow <= Storage Capacity

            Parameters
            ----------
            pt : pyomo.core.base.set.OrderedScalarSet
                technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Storage inflow upper bound
            """
            return (
                self.storage_inflow[(pt, y, r, steps, hr)]
                <= self.capacity_total[(r, self.Map_hr_s[hr], pt, steps, y)] * self.Hr_weights[hr]
            )

        # TODO check if it's only able to build in regions with existing capacity?
        @self.Constraint(self.Storage_index)
        def storage_outflow_ub(self, pt, y, r, steps, hr):
            """Storage outflow upper bound where
            Storage outflow <= Storage Capacity

            Parameters
            ----------
            pt : pyomo.core.base.set.OrderedScalarSet
                technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Storage outflow upper bound
            """
            return (
                self.storage_outflow[(pt, y, r, steps, hr)]
                + (
                    sum(
                        self.reserves_procurement[(restype, pt, y, r, steps, hr)]
                        for restype in self.restypes
                    )
                    if self.sw_reserves
                    else 0
                )
                <= self.capacity_total[(r, self.Map_hr_s[hr], pt, steps, y)] * self.Hr_weights[hr]
            )

        @self.Constraint(self.Storage_index)
        def storage_level_ub(self, pt, y, r, steps, hr):
            """Storage level upper bound where
            Storage level <= Storage power capacity * storage energy capacity

            Parameters
            ----------
            pt : pyomo.core.base.set.OrderedScalarSet
                technology set
            y : pyomo.core.base.set.OrderedScalarSet
                year set
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            hr : pyomo.core.base.set.OrderedScalarSet
                time-segment set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Storage level upper bound
            """
            return (
                self.storage_level[(pt, y, r, steps, hr)]
                <= self.capacity_total[(r, self.Map_hr_s[hr], pt, steps, y)] * self.HourstoBuy[(pt)]
            )

        @self.Constraint(self.capacity_total_index)
        def capacity_balance(self, r, s, pt, steps, y):
            """Capacity Equality constraint where
            Capacity = Operating Capacity
                      + New Builds Capacity
                      - Retired Capacity

            Parameters
            ----------
            r : pyomo.core.base.set.OrderedScalarSet
                region set
            s : pyomo.core.base.set.OrderedScalarSet
                season set
            pt : pyomo.core.base.set.OrderedScalarSet
                technology set
            steps : pyomo.core.base.set.OrderedScalarSet
                supply curve price/quantity step set
            y : pyomo.core.base.set.OrderedScalarSet
                year set

            Returns
            -------
            pyomo.core.base.constraint.IndexedConstraint
                Capacity Equality

            """
            return self.capacity_total[(r, s, pt, steps, y)] == self.SupplyCurve[
                (r, s, pt, steps, y)
            ] + (
                sum(self.capacity_builds[(r, pt, year, steps)] for year in self.y if year <= y)
                if self.sw_expansion and (pt, steps) in self.Build_index
                else 0
            ) - (
                sum(self.capacity_retirements[(pt, year, r, steps)] for year in self.y if year <= y)
                if self.sw_expansion and (pt, y, r, steps) in self.capacity_retirements_index
                else 0
            )

        # if capacity expansion is on
        if self.sw_expansion:

            @self.Constraint(self.capacity_retirements_index)
            def capacity_retirements_ub(self, pt, y, r, steps):
                """Retirement upper bound where
                Capacity Retired <= Operating Capacity
                                   + New Builds Capacity
                                   - Retired Capacity

                Parameters
                ----------
                pt : pyomo.core.base.set.OrderedScalarSet
                    technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Retirement upper bound
                """
                return self.capacity_retirements[(pt, y, r, steps)] <= (
                    (
                        self.SupplyCurve[(r, 2, pt, steps, y)]
                        if (r, 2, pt, steps, y) in self.capacity_total_index
                        else 0
                    )
                    + (
                        sum(
                            self.capacity_builds[(r, pt, year, steps)]
                            for year in self.y
                            if year < y
                        )
                        if (pt, steps) in self.Build_index
                        else 0
                    )
                    - sum(
                        self.capacity_retirements[(pt, year, r, steps)]
                        for year in self.y
                        if year < y
                    )
                )

        # if trade operation is on
        if self.sw_trade and len(self.TranLineLimitInt_index) != 0:
            self.populate_trade_sets = pyo.BuildAction(rule=f.populate_trade_sets_rule)

            @self.Constraint(self.TranLineLimitInt_index)
            def trade_interational_capacity_ub(self, r, r_int, y, hr):
                """International interregional trade upper bound where
                Interregional Trade <= Interregional Transmission Capabilities * Time

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                r_int : pyomo.core.base.set.OrderedScalarSet
                    international region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    International interregional trade capacity upper bound
                """
                return (
                    sum(
                        self.trade_international[(r, r_int, y, c, hr)]
                        for c in self.TradeCanLineSetUpper[(r, r_int, y, hr)]
                    )
                    <= self.TranLimitCapInt[(r, r_int, y, hr)] * self.Hr_weights[hr]
                )

            @self.Constraint(self.TranLimitInt_index)
            def trade_interational_generation_ub(self, r_int, CSteps, y, hr):
                """International electricity supply upper bound where
                Interregional Trade <= Interregional Supply

                Parameters
                ----------
                r_int : pyomo.core.base.set.OrderedScalarSet
                    international region set
                CSteps : pyomo.core.base.set.OrderedScalarSet
                    international trade supply curve steps set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    International electricity supply upper bound
                """
                return (
                    sum(
                        self.trade_international[(r, r_int, y, CSteps, hr)]
                        for r in self.TradeCanSetUpper[(r_int, y, CSteps, hr)]
                    )
                    <= self.TranLimitGenInt[(r_int, CSteps, y, hr)] * self.Hr_weights[hr]
                )

            @self.Constraint(self.trade_interregional_index)
            def trade_domestic_ub(self, r, r1, y, hr):
                """Interregional trade upper bound where
                Interregional Trade <= Interregional Transmission Capabilities * Time

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                r1 : pyomo.core.base.set.OrderedScalarSet
                    region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Interregional trade capacity upper bound
                """
                return (
                    self.trade_interregional[(r, r1, y, hr)]
                    <= self.TranLimit[(r, r1, self.Map_hr_s[hr], y)] * self.Hr_weights[hr]
                )

        # if reserve margin requirements are on
        if self.sw_expansion and self.sw_rm:
            self.populate_RM_sets = pyo.BuildAction(rule=f.populate_RM_sets_rule)

            @self.Constraint(self.demand_balance_index)
            def reserve_margin_lb(self, r, y, hr):
                """Reserve margin requirement where
                Load * Reserve Margin <= Capacity * Capacity Credit * Time

                # must meet reserve margin requirement
                # apply to every hour, a fraction above the final year's load
                # ReserveMarginReq <= sum(Max capacity in that hour)

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Reserve margin requirement
                """
                return self.Load[(r, y, hr)] * (1 + self.ReserveMargin[r]) <= self.Hr_weights[
                    hr
                ] * sum(
                    (
                        self.CapacityCredit[(pt, y, r, steps, hr)]
                        * (
                            self.storage_avail_cap[(pt, y, r, steps, hr)]
                            if pt in self.pts
                            else self.capacity_total[(r, self.Map_hr_s[hr], pt, steps, y)]
                        )
                    )
                    for (pt, steps) in self.SupplyCurveRM[(y, r, self.Map_hr_s[hr])]
                )

            @self.Constraint(self.Storage_index)
            def reserve_margin_storage_avail_cap_ub(self, pts, y, r, steps, hr):
                """Available storage power capacity for meeting reserve margin

                # ensure available capacity to meet RM for storage < power capacity

                Parameters
                ----------
                pts : pyomo.core.base.set.OrderedScalarSet
                    storage technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time-segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Available storage power capacity for meeting reserve margin
                """
                return (
                    self.storage_avail_cap[(pts, y, r, steps, hr)]
                    <= self.capacity_total[(r, self.Map_hr_s[hr], pts, steps, y)]
                )

            @self.Constraint(self.Storage_index)
            def reserve_margin_storage_avail_level_ub(self, pts, y, r, steps, hr):
                """Available storage energy capacity for meeting reserve margin

                # ensure available capacity to meet RM for storage < existing SOC

                Parameters
                ----------
                pts : pyomo.core.base.set.OrderedScalarSet
                    storage technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time-segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Available storage energy capacity for meeting reserve margin
                """
                return (
                    self.storage_avail_cap[(pts, y, r, steps, hr)]
                    <= self.storage_level[(pts, y, r, steps, hr)]
                )

        # if ramping requirements are on
        if self.sw_ramp:

            @self.Constraint(self.ramp_first_hour_balance_index)
            def ramp_first_hour_balance(self, ptc, y, r, steps, hr1):
                """Ramp constraint for the first hour time-segment in each day-type where
                Generation == Generation (in final hour time-segment in current day-type)
                            + Ramp Up
                            - Ramp Down

                Parameters
                ----------
                ptc : pyomo.core.base.set.OrderedScalarSet
                    conventional technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr1 : pyomo.core.base.set.OrderedScalarSet
                    set containing first hour time-segment in each day-type

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Ramp constraint for the first hour
                """
                return (
                    self.generation_total[(ptc, y, r, steps, hr1)]
                    == self.generation_total[(ptc, y, r, steps, hr1 + self.num_hr_day - 1)]
                    + self.generation_ramp_up[(ptc, y, r, steps, hr1)]
                    - self.generation_ramp_down[(ptc, y, r, steps, hr1)]
                )

            @self.Constraint(self.ramp_most_hours_balance_index)
            def ramp_most_hours_balance(self, ptc, y, r, steps, hr23):
                """Ramp constraint for the time-segment in each day-type other than
                the first hour time-segment where
                Generation == Generation (in previous hour time-segment)
                            + Ramp Up
                            - Ramp Down

                Parameters
                ----------
                ptc : pyomo.core.base.set.OrderedScalarSet
                    conventional technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr23 : pyomo.core.base.set.OrderedScalarSet
                    set containing time-segment except first hour in each day-type

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Ramp constraint for the first hour
                """
                return (
                    self.generation_total[(ptc, y, r, steps, hr23)]
                    == self.generation_total[(ptc, y, r, steps, hr23 - 1)]
                    + self.generation_ramp_up[(ptc, y, r, steps, hr23)]
                    - self.generation_ramp_down[(ptc, y, r, steps, hr23)]
                )

            @self.Constraint(self.generation_ramp_index)
            def ramp_up_ub(self, ptc, y, r, steps, hr):
                """Ramp rate up upper constraint where
                Ramp Up <= Capaciry * Ramp Rate * Time

                Parameters
                ----------
                ptc : pyomo.core.base.set.OrderedScalarSet
                    conventional technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Ramp rate up upper constraint
                """
                return (
                    self.generation_ramp_up[(ptc, y, r, steps, hr)]
                    <= self.Hr_weights[hr]
                    * self.RampRate[ptc]
                    * self.capacity_total[(r, self.Map_hr_s[hr], ptc, steps, y)]
                )

            @self.Constraint(self.generation_ramp_index)
            def ramp_down_ub(self, ptc, y, r, steps, hr):
                """Ramp rate down upper constraint where
                Ramp Up <= Capaciry * Ramp Rate * Time

                Parameters
                ----------
                ptc : pyomo.core.base.set.OrderedScalarSet
                    conventional technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Ramp rate down upper constraint
                """
                return (
                    self.generation_ramp_down[(ptc, y, r, steps, hr)]
                    <= self.Hr_weights[hr]
                    * self.RampRate[ptc]
                    * self.capacity_total[(r, self.Map_hr_s[hr], ptc, steps, y)]
                )

        # if operating reserve requirements are on
        if self.sw_reserves:
            self.populate_reserves_sets = pyo.BuildAction(rule=f.populate_reserves_sets_rule)

            @self.Constraint(self.demand_balance_index)
            def reserve_requirement_spin_lb(self, r, y, hr):
                """Spinning reserve requirements (3% of load) where
                Spinning reserve procurement >= 0.03 * Load

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time-segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Spinning reserve requirements
                """
                return (
                    sum(
                        self.reserves_procurement[(1, pt, y, r, step, hr)]
                        for (pt, step) in self.ProcurementSetReserves[(1, r, y, hr)]
                    )
                    >= 0.03 * self.Load[(r, y, hr)]
                )

            @self.Constraint(self.demand_balance_index)
            def reserve_requirement_reg_lb(self, r, y, hr):
                """Regulation Reserve Requirement (1% of load + 0.5% of wind gen + 0.3% of solar cap) where
                Reserves Requirement >= 0.01 * Load
                                      + 0.005 * Wind Gen
                                      + 0.003 * Solar Cap

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time-segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Regulation reserve requirement
                """
                return sum(
                    self.reserves_procurement[(2, pt, y, r, step, hr)]
                    for (pt, step) in self.ProcurementSetReserves[(2, r, y, hr)]
                ) >= 0.01 * self.Load[(r, y, hr)] + 0.005 * sum(
                    self.generation_total[(ptw, y, r, step, hr)]
                    for (ptw, step) in self.WindSetReserves[(y, r, hr)]
                ) + 0.003 * self.Hr_weights[hr] * sum(
                    self.capacity_total[(r, self.Map_hr_s[hr], ptsol, step, y)]
                    for (ptsol, step) in self.SolarSetReserves[(y, r, hr)]
                )

            @self.Constraint(self.demand_balance_index)
            def reserve_requirement_flex_lb(self, r, y, hr):
                """Flexible Reserve Requirement (10% of wind gen + 4% of solar cap) where
                Reserves Requirement >= 0.01 * Wind Gen
                                      + 0.04 * Solar Cap

                Parameters
                ----------
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time-segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Flexible reserve requirement
                """
                return sum(
                    self.reserves_procurement[(3, pt, y, r, step, hr)]
                    for (pt, step) in self.ProcurementSetReserves[(3, r, y, hr)]
                ) >= +0.1 * sum(
                    self.generation_total[(ptw, y, r, step, hr)]
                    for (ptw, step) in self.WindSetReserves[(y, r, hr)]
                ) + 0.04 * self.Hr_weights[hr] * sum(
                    self.capacity_total[(r, self.Map_hr_s[hr], ptsol, step, y)]
                    for (ptsol, step) in self.SolarSetReserves[(y, r, hr)]
                )

            @self.Constraint(self.reserves_procurement_index)
            def reserve_procurement_ub(self, restypes, pt, y, r, steps, hr):
                """Reserve Requirement Procurement Upper Bound where
                Reserve Procurement <= Capacity
                                    * Tech Reserve Contribution Share
                                    * Time

                Parameters
                ----------
                restypes : pyomo.core.base.set.OrderedScalarSet
                    reserve requirement type set
                pt : pyomo.core.base.set.OrderedScalarSet
                    technology set
                y : pyomo.core.base.set.OrderedScalarSet
                    year set
                r : pyomo.core.base.set.OrderedScalarSet
                    region set
                steps : pyomo.core.base.set.OrderedScalarSet
                    supply curve price/quantity step set
                hr : pyomo.core.base.set.OrderedScalarSet
                    time segment set

                Returns
                -------
                pyomo.core.base.constraint.IndexedConstraint
                    Reserve Requirement Procurement Upper Bound
                """
                return (
                    self.reserves_procurement[(restypes, pt, y, r, steps, hr)]
                    <= self.ResTechUpperBound[(restypes, pt)]
                    * self.Hr_weights[hr]
                    * self.capacity_total[(r, self.Map_hr_s[hr], pt, steps, y)]
                )
