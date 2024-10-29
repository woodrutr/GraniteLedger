# src.models.hydrogen.model package

## Submodules

## src.models.hydrogen.model.actions module

A sequencer for actions in the model.
This may change up a bit, but it is a place to assert control of the execution sequence for now

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.build_grid(grid_data: [GridData](src.models.hydrogen.network.md#src.models.hydrogen.network.grid_data.GridData))

build a grid from grid_data

* **Parameters:**
  **grid_data: obj**
  : GridData object to build grid from
* **Returns:**
  **Grid**
  : Grid object

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.build_model(grid: [Grid](src.models.hydrogen.network.md#src.models.hydrogen.network.grid.Grid), \*\*kwds)

build model from grd

* **Parameters:**
  **grid**
  : Grid object to build model from
* **Returns:**
  **H2Model**
  : H2Model object

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.load_data(path_to_input: Path, \*\*kwds)

load data for model

* **Parameters:**
  **path_to_input**
  : Data folder path
* **Returns:**
  **GridData**
  : Grid Data object from path

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.make_h2_outputs(model)

save model outputs

* **Parameters:**
  **model**
  : Solved H2Model

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.quick_summary(solved_hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model))

print and return summary of solve

* **Parameters:**
  **solved_hm**
  : Solved H2Model
* **Returns:**
  **res**
  : Printed summary

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.run_hydrogen_model(settings)

run hydrogen model in standalone

* **Parameters:**
  **settings**
  : Config_setup instance

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.actions.solve_it(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model))

solve hm

* **Parameters:**
  **hm**
  : H2Model to solve
* **Returns:**
  **SolverResults**
  : results of solve

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.model.h2_model module

The Hydrogen Model takes an a Grid object and uses it to populate a Pyomo model that solves for the least cost to produce and distribute Hydrogen by electrolysis
across the grid to satisfy a given demand, returning the duals as shadow prices. It can be run in stand-alone or integrated runs. If stand-alone, a function
for generated temporally varying data must be supplied. By default it simply projects geometric growth for electricity price and demand.

<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.model.h2_model.H2Model(\*args, \*\*kwds)

Bases: `ConcreteModel`

<!-- !! processed by numpydoc !! -->

#### poll_electric_demand()

compute the electrical demand by region-year after solve

Note:  we will use production \* 1/eff to compute electrical demand

* **Parameters:**
  **hm**
  : self
* **Returns:**
  dict[HI, float]
  : electricity demand by region, year. (region, year):demand

<!-- !! processed by numpydoc !! -->

#### update_exchange_params(new_demand=None, new_electricity_price=None)

update exchange parameters in integrated mode

* **Parameters:**
  **hm**
  : model

  **new_demand**
  : new demand (region, year):value. Defaults to None.

  **new_electricity_price**
  : new electricity prices (region,year):value . Defaults to None.

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.h2_model.resolve(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model), new_demand=None, new_electricity_price=None, test=False)

For convenience: After building and solving the model initially:

> if you want to solve without annual data by applying a geometric growth rate to exhcange parameters
* **Parameters:**
  **hm**
  : model

  **new_demand**
  : new_demand[region,year] for H2demand in (region,year). Defaults to None.

  **new_electricity_price**
  : new_electricity_price[region,year]. Defaults to None.

  **test**
  : is this just a test? Defaults to False.

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.h2_model.solve(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model))

\_summary_

* **Parameters:**
  **hm**
  : self
* **Raises:**
  RuntimeError
  : no optimal solution to problem

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.model.validators module

set of validator functions for use in model

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.model.validators.region_validator(hm: [H2Model](#src.models.hydrogen.model.h2_model.H2Model), region)

checks if region name is string or numeric

* **Parameters:**
  **hm**
  : > model
    <br/>
    region
    : region name

  **Raises:**
  : ValueError: region wrong type

  **Returns:**
  : bool: is correct type

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
