# src.models.hydrogen.model.h2_model

The Hydrogen Model takes in a Grid object and uses it to populate a Pyomo model that solves for the
least cost to produce and distribute Hydrogen by electrolysis across the grid to satisfy a given
demand, returning the duals as shadow prices. It can be run in stand-alone or integrated runs. If
stand-alone, a function for generated temporally varying data must be supplied. By default it simply
projects geometric growth for electricity price and demand.

### Functions

| `check_optimal_termination`(results)                                            | This function returns True if the termination condition for the solver is 'optimal', 'locallyOptimal', or 'globallyOptimal', and the status is 'ok'   |
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `getLogger`([name])                                                             | Return a logger with the specified name, creating it if necessary.                                                                                    |
| [`resolve`](#src.models.hydrogen.model.h2_model.resolve)(hm[, new_demand, ...]) | For convenience: After building and solving the model initially:                                                                                      |
| [`solve`](#src.models.hydrogen.model.h2_model.solve)(hm)                        | \_summary_                                                                                                                                            |
| `value`(obj[, exception])                                                       | A utility function that returns the value of a Pyomo object or expression.                                                                            |

### Classes

| `Block`(\*args, \*\*kwds)                                                  | Blocks are indexed components that contain other components (including blocks).   |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `ConcreteModel`(\*args, \*\*kwds)                                          | A concrete optimization model that does not defer construction of components.     |
| `Constraint`(\*args, \*\*kwds)                                             | This modeling component defines a constraint expression using a rule function.    |
| `Grid`([data])                                                             |                                                                                   |
| [`H2Model`](#src.models.hydrogen.model.h2_model.H2Model)(\*args, \*\*kwds) |                                                                                   |
| `HI`(region, year)                                                         | (region, year)                                                                    |
| `LinearExpression`([args, constant, ...])                                  | An expression object for linear polynomials.                                      |
| `Objective`(\*args, \*\*kwds)                                              | This modeling component defines an objective expression.                          |
| `Param`(\*args, \*\*kwds)                                                  | A parameter value, which may be defined over an index.                            |
| `RangeSet`(\*args, \*\*kwds)                                               | A set object that represents a set of numeric values                              |
| `Set`(\*args, \*\*kwds)                                                    | A component used to index other Pyomo components.                                 |
| `SolverResults`(\*args, \*\*kwargs)                                        |                                                                                   |
| `Suffix`(\*args, \*\*kwargs)                                               | A model suffix, representing extraneous model data                                |
| `Var`(\*args, \*\*kwargs)                                                  | A numeric variable, which may be defined over an index.                           |
| `defaultdict`                                                              | defaultdict(default_factory=None, /, [...]) --> dict with default factory         |

### *class* src.models.hydrogen.model.h2_model.H2Model(\*args, \*\*kwds)

#### \_active

#### \_filter_update_info(data: dict[[HI](src.integrator.utilities.md#src.integrator.utilities.HI), float]) → dict[[HI](src.integrator.utilities.md#src.integrator.utilities.HI), float]

quick filter to remove regions that don’t exist in the model

> > It is possible (right now) that the H2 network is unaware of particular regions

> because no baseline data for them was ever provided…. so it is possible to
> recieve and “unkown” region here, even though it was selected, due to lack of
> data
* **Parameters:**
  * **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – self
  * **data** (*dict* *[*[*HI*](src.integrator.utilities.md#src.integrator.utilities.HI) *,* *float* *]*) – hydrogen index : value
* **Returns:**
  regions index: value with missing data removed
* **Return type:**
  dict[[HI](src.integrator.utilities.md#src.integrator.utilities.HI), float]

#### \_update_demand(new_demand)

update the demand parameter with new demand data

> insert new demand as a dict in the format: new_demand[region, year]
* **Parameters:**
  * **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – self
  * **new_demand** (*dict*) – new demand values

#### \_update_electricity_price(new_electricity_price)

update electricity price parameter

* **Parameters:**
  * **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – self
  * **new_electricity_price** (*dict*) – region, year : electricity price

#### poll_electric_demand() → dict[[HI](src.integrator.utilities.md#src.integrator.utilities.HI), float]

compute the electrical demand by region-year after solve

Note:  we will use production \* 1/eff to compute electrical demand

* **Parameters:**
  **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – self
* **Returns:**
  electricity demand by region, year. (region, year):demand
* **Return type:**
  dict[[HI](src.integrator.utilities.md#src.integrator.utilities.HI), float]

#### update_exchange_params(new_demand=None, new_electricity_price=None)

update exchange parameters in integrated mode

* **Parameters:**
  * **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **new_demand** (*dict* *,* *optional*) – new demand (region, year):value. Defaults to None.
  * **new_electricity_price** (*dict* *,* *optional*) – new electricity prices (region,year):value . Defaults to None.

### src.models.hydrogen.model.h2_model.resolve(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model), new_demand=None, new_electricity_price=None, test=False)

For convenience: After building and solving the model initially:

if you want to solve without annual data by applying a geometric growth rate to exhcange params

* **Parameters:**
  * **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **new_demand** (*dict* *,* *optional*) – new_demand[region,year] for H2demand in (region,year). Defaults to None.
  * **new_electricity_price** (*dict* *,* *optional*) – new_electricity_price[region,year]. Defaults to None.
  * **test** (*bool* *,* *optional*) – is this just a test? Defaults to False.

### src.models.hydrogen.model.h2_model.solve(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model))

\_summary_

* **Parameters:**
  **hm** ([*H2Model*](#src.models.hydrogen.model.h2_model.H2Model)) – self
* **Raises:**
  **RuntimeError** – no optimal solution to problem
