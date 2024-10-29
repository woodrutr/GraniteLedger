"""Speed Test
This is a script with some functions to run speed and accuracy tests on test models constructed with babymodel.

It can be run directly. Parameter values are at the top of the file and any desired value can be entered into the declarations.
The script will then run a sequence of functions to time the build, sympification, and sensitivity calculation for the TestBabyModel
constructed with those input values.

"""

###################################################################################################
# Import packages
from collections import defaultdict
import pyomo.environ as pyo
from sensitivity_tools import *
from sympy import *
import random as rnd
import time
from babymodel import *
from sensitivity_tools import *
from logging import getLogger

# Establish logger
# logger = getLogger(__name__)

###################################################################################################

start_year = 2020
end_year = 2025
price_growth_rate = 1.2
demand_growth_rate = 1.1
num_regions = 3
hubs_per_region = 2
base_elec_price = 5.1
base_prod_capacity = 5000
demand_fraction = 0.8
transportation_cap = 2000
p = 0.1

name_of_parameter_to_test = 'transportation_capacity'


(
    hublist,
    region_list,
    hub_map,
    region_map,
    elec_price,
    prod_capacity,
    demand,
    elec_consumption,
    base_transport_cost,
) = generate(
    num_regions=num_regions,
    hubs_per_region=hubs_per_region,
    base_elec_price=base_elec_price,
    base_prod_capacity=base_prod_capacity,
    demand_fraction=demand_fraction,
)

arcs, outbound, inbound, trans_capacity = connect_regions(region_map, hub_map, transportation_cap)

params = {
    'hublist': hublist,
    'region_list': region_list,
    'hub_map': hub_map,
    'region_map': region_map,
    'elec_price': elec_price,
    'prod_capacity': prod_capacity,
    'arcs': arcs,
    'outbound': outbound,
    'inbound': inbound,
    'trans_capacity': trans_capacity,
    'base_demand': demand,
    'base_elec_consumption': elec_consumption,
    'base_transport_cost': base_transport_cost,
}


def run_timed(params, parameter_name):
    """generates an instance of TestBabyModel with given parameters, sympifies it, and follows the sequence of steps
        to get the sensitivity matrix. Prints times for each step.

    Parameters
    ----------
    params : dict
        dictionary of parameter values for the TestBabyModel instance generated. These are set in the beginning of the script.
        Values can be changed in the declaration statements.
    parameter_name : str
        the name of the scalar parameter to evaluate sensitivities with respect to.

    Returns
    -------
    tuple
        tuple of model to be solved, the sympification of the model, the SensitivityMatrix, DifferentialMapping associated with parameter and model solve values
    """

    start = time.time()
    model = TestBabyModel(
        params,
        start_year=start_year,
        end_year=end_year,
        price_growth_rate=price_growth_rate,
        demand_growth_rate=demand_growth_rate,
    )
    model.solve()
    stop = time.time()
    time_solve = stop - start
    start = time.time()
    auto = AutoSympy(model)
    stop = time.time()
    time_sympification = stop - start
    start = time.time()
    M = auto.get_sensitivity_matrix([parameter_name])
    stop = time.time()
    time_sensitivity_matrix = stop - start
    start = time.time()
    sen = M.get_sensitivities()
    stop = time.time()
    time_sub_values = stop - start

    print('summary statistics:')
    print('~~~~~~~~~~~~~~~~~~~')
    print('number of variables: ', len(M.variable_vector))
    print('number of parameters: ', len(M.parameter_vector))
    print(
        'number of active, non-degenerate equality constraints: ',
        len(M.sympification['equality_constraints']),
    )
    print('number of equality duals: ', len(M.duals['equality_duals']))
    print('number of inequality_constraints: ', len(M.sympification['inequality_constraints']))
    print(
        'number of active, non-degenerate inequality constraints: ',
        len(M.duals['inequality_duals']),
    )
    print('time to solve model: ', time_solve)
    print('time to sympify: ', time_sympification)
    print('time to generate sensitivity matrix: ', time_sensitivity_matrix)
    print('time to substitute values: ', time_sub_values)

    return model, auto, M, sen


def speed_accuracy_test(params, parameter_name, p):
    """performs a run_timed execution with given parameters, takes the returned values, and measures the effect of increasing and
        decreasing the named parameter by p, and compares the result to resolving the model with that same perturbation.

    Returns
    -------
    params : dict
        dictionary of parameter values for the TestBabyModel instance generated. These are set in the beginning of the script.
        Values can be changed in the declaration statements.
    parameter_name : str
        the name of the scalar parameter to evaluate sensitivities with respect to.
    p : float
        percentage change up and down to be measured, as a decimal.
    """

    model, auto, M, sen = run_timed(params, parameter_name)
    parameter = getattr(model, parameter_name)
    var_values = list(M.sympification['variable_values'].values())
    eq_dual_values = [x for x in M.sympification['duals']['equality_duals'].values()]
    ineq_dual_values = [x[0] for x in M.sympification['duals']['inequality_duals'].values()]
    z = [pyo.value(model.total_cost)]
    val = var_values + eq_dual_values + ineq_dual_values + z

    val_vector = Matrix(val)

    obj_sensitivity_to_param = sen.US.row(-1)
    old_value = Matrix(z)

    param_value = auto.parameter_values[auto.parameters[parameter_name]]
    delta_param = p * param_value

    predicted_new_value_up = obj_sensitivity_to_param * delta_param + old_value
    predicted_new_value_down = -obj_sensitivity_to_param * delta_param + old_value

    parameter.set_value(param_value + delta_param)
    model.solve()
    new_z_up = pyo.value(model.total_cost)

    parameter.set_value(param_value - delta_param)
    model.solve()
    new_z_down = pyo.value(model.total_cost)
    print('summary:')
    print('~~~~~~~~')
    print('objective value: ', z)
    print(
        'increase',
        parameter_name,
        100 * p,
        '% actual: ',
        new_z_up,
        'predicted:',
        predicted_new_value_up.values()[0],
    )
    print(
        'decrease',
        parameter_name,
        100 * p,
        '% actual:',
        new_z_down,
        'predicted: ',
        predicted_new_value_down.values()[0],
    )


# run speed_accuracy_test with given inputs (can be changed directly at beginning of script)
speed_accuracy_test(params, name_of_parameter_to_test, p)
