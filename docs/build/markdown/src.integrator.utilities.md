# src.integrator.utilities

A gathering of utility functions for dealing with model interconnectivity

Dev Note:  At some review point, some decisions may move these back & forth with parent
models after it is decided if it is a utility job to do …. or a class method.

Additionally, there is probably some renaming due here for consistency

### Module Attributes

| [`EI`](#src.integrator.utilities.EI)(region, year, hour)   | (region, year, hour)   |
|------------------------------------------------------------|------------------------|
| [`HI`](#src.integrator.utilities.HI)(region, year)         | (region, year)         |

### Functions

| [`convert_elec_price_to_lut`](#src.integrator.utilities.convert_elec_price_to_lut)(prices)             | convert electricity prices to dictionary, look up table                                                              |
|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`convert_h2_price_records`](#src.integrator.utilities.convert_h2_price_records)(records)              | simple coversion from list of records to a dictionary LUT repeat entries should not occur and will generate an error |
| [`create_temporal_mapping`](#src.integrator.utilities.create_temporal_mapping)(sw_temporal)            | Combines the input mapping files within the electricity model to create a master temporal mapping dataframe.         |
| `getLogger`([name])                                                                                    | Return a logger with the specified name, creating it if necessary.                                                   |
| [`get_annual_wt_avg`](#src.integrator.utilities.get_annual_wt_avg)(elec_price)                         | takes annual weighted average of hourly electricity prices                                                           |
| [`get_elec_price`](#src.integrator.utilities.get_elec_price)(instance[, block])                        | pulls hourly electricity prices from completed PowerModel and de-weights them.                                       |
| `namedtuple`(typename, field_names, \*[, ...])                                                         | Returns a new subclass of tuple with named fields.                                                                   |
| [`poll_h2_demand`](#src.integrator.utilities.poll_h2_demand)(model)                                    | Get the hydrogen demand by rep_year and region                                                                       |
| [`poll_h2_prices_from_elec`](#src.integrator.utilities.poll_h2_prices_from_elec)(model, tech, regions) | poll the step-1 H2 price currently in the model for region/year, averaged over any steps                             |
| [`poll_hydrogen_price`](#src.integrator.utilities.poll_hydrogen_price)(model[, block])                 | Retrieve the price of H2 from the H2 model                                                                           |
| [`poll_year_avg_elec_price`](#src.integrator.utilities.poll_year_avg_elec_price)(price_list)           | retrieve a REPRESENTATIVE price at the annual level from a listing of prices                                         |
| [`regional_annual_prices`](#src.integrator.utilities.regional_annual_prices)(m[, block])               | pulls all regional annual weighted electricity prices                                                                |
| [`select_solver`](#src.integrator.utilities.select_solver)(instance)                                   | Select solver based on learning method                                                                               |
| [`simple_solve`](#src.integrator.utilities.simple_solve)(m)                                            | a simple solve routine                                                                                               |
| [`simple_solve_no_opt`](#src.integrator.utilities.simple_solve_no_opt)(m, opt)                         | Solve concrete model using solver factory object                                                                     |
| [`update_elec_demand`](#src.integrator.utilities.update_elec_demand)(self, elec_demand)                | Update the external electical demand parameter with demands from the H2 model                                        |
| [`update_h2_prices`](#src.integrator.utilities.update_h2_prices)(model, h2_prices)                     | Update the H2 prices held in the model                                                                               |
| `value`(obj[, exception])                                                                              | A utility function that returns the value of a Pyomo object or expression.                                           |

### Classes

| `ConcreteModel`(\*args, \*\*kwds)                        | A concrete optimization model that does not defer construction of components.   |
|----------------------------------------------------------|---------------------------------------------------------------------------------|
| [`EI`](#src.integrator.utilities.EI)(region, year, hour) | (region, year, hour)                                                            |
| [`HI`](#src.integrator.utilities.HI)(region, year)       | (region, year)                                                                  |
| `Path`(\*args, \*\*kwargs)                               | PurePath subclass that can make system calls.                                   |
| `defaultdict`                                            | defaultdict(default_factory=None, /, [...]) --> dict with default factory       |

### *class* src.integrator.utilities.EI(region, year, hour)

(region, year, hour)

#### \_asdict()

Return a new dict which maps field names to their values.

#### \_field_defaults *= {}*

#### \_fields *= ('region', 'year', 'hour')*

#### *classmethod* \_make(iterable)

Make a new EI object from a sequence or iterable

#### \_replace(\*\*kwds)

Return a new EI object replacing specified fields with new values

#### hour

Alias for field number 2

#### region

Alias for field number 0

#### year

Alias for field number 1

### *class* src.integrator.utilities.HI(region, year)

(region, year)

#### \_asdict()

Return a new dict which maps field names to their values.

#### \_field_defaults *= {}*

#### \_fields *= ('region', 'year')*

#### *classmethod* \_make(iterable)

Make a new HI object from a sequence or iterable

#### \_replace(\*\*kwds)

Return a new HI object replacing specified fields with new values

#### region

Alias for field number 0

#### year

Alias for field number 1

### src.integrator.utilities.convert_elec_price_to_lut(prices: list[tuple[[EI](#src.integrator.utilities.EI), float]]) → dict[[EI](#src.integrator.utilities.EI), float]

convert electricity prices to dictionary, look up table

* **Parameters:**
  **prices** (*list* *[**tuple* *[*[*EI*](#src.integrator.utilities.EI) *,* *float* *]* *]*) – list of prices
* **Returns:**
  dict of prices
* **Return type:**
  dict[[EI](#src.integrator.utilities.EI), float]

### src.integrator.utilities.convert_h2_price_records(records: list[tuple[[HI](#src.integrator.utilities.HI), float]]) → dict[[HI](#src.integrator.utilities.HI), float]

simple coversion from list of records to a dictionary LUT
repeat entries should not occur and will generate an error

### src.integrator.utilities.create_temporal_mapping(sw_temporal)

Combines the input mapping files within the electricity model to create a master temporal
mapping dataframe. The df is used to build multiple temporal parameters used within the  model.
It creates a single dataframe that has 8760 rows for each hour in the year.
Each hour in the year is assigned a season type, day type, and hour type used in the model.
This defines the number of time periods the model will use based on cw_s_day and cw_hr inputs.

* **Returns:**
  a dataframe with 8760 rows that include each hour, hour type, day, day type, and season.
  It also includes the weights for each day type and hour type.
* **Return type:**
  dataframe

### src.integrator.utilities.get_annual_wt_avg(elec_price: DataFrame) → dict[[HI](#src.integrator.utilities.HI), float]

takes annual weighted average of hourly electricity prices

* **Parameters:**
  **elec_price** (*pd.DataFrame*) – hourly electricity prices
* **Returns:**
  annual weighted average electricity prices
* **Return type:**
  dict[[HI](#src.integrator.utilities.HI), float]

### src.integrator.utilities.get_elec_price(instance: [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel) | ConcreteModel, block=None) → DataFrame

pulls hourly electricity prices from completed PowerModel and de-weights them.

Prices from the duals are weighted by the day and year weights applied in the OBJ function
This function retrieves the prices for all hours and removes the day and annual weights to
return raw prices (and the day weights to use as needed)

* **Parameters:**
  * **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – solved electricity model
  * **block** (*ConcreteModel*) – reference to the block if the electricity model is a block within a larger model
* **Returns:**
  df of raw prices and the day weights to re-apply (if needed)
  columns: [r, y, hour, day_weight, raw_price]
* **Return type:**
  pd.DataFrame

### src.integrator.utilities.poll_h2_demand(model: [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) → dict[[HI](#src.integrator.utilities.HI), float]

Get the hydrogen demand by rep_year and region

Use the Generation variable for h2 techs

NOTE:  Not sure about day weighting calculation here!!

* **Returns:**
  dictionary of prices by H2 Index: price
* **Return type:**
  dict[[HI](#src.integrator.utilities.HI), float]

### src.integrator.utilities.poll_h2_prices_from_elec(model: [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel), tech, regions: Iterable) → dict[Any, float]

poll the step-1 H2 price currently in the model for region/year, averaged over any steps

* **Parameters:**
  * **model** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – solved PowerModel
  * **tech** (*str*) – h2 tech
  * **regions** (*Iterable*)
* **Returns:**
  a dictionary of (region, seasons, year): price
* **Return type:**
  dict[*Any*, float]

### src.integrator.utilities.poll_hydrogen_price(model: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model) | ConcreteModel, block=None) → list[tuple[[HI](#src.integrator.utilities.HI), float]]

Retrieve the price of H2 from the H2 model

* **Parameters:**
  * **model** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – the model to poll
  * **block** (*optional*) – block model to poll
* **Returns:**
  list of H2 Index, price tuples
* **Return type:**
  list[tuple[[HI](#src.integrator.utilities.HI), float]]

### src.integrator.utilities.poll_year_avg_elec_price(price_list: list[tuple[[EI](#src.integrator.utilities.EI), float]]) → dict[[HI](#src.integrator.utilities.HI), float]

retrieve a REPRESENTATIVE price at the annual level from a listing of prices

This function computes the AVERAGE elec price for each region-year combo

* **Parameters:**
  **price_list** (*list* *[**tuple* *[*[*EI*](#src.integrator.utilities.EI) *,* *float* *]* *]*) – input price list
* **Returns:**
  a dictionary of (region, year): price
* **Return type:**
  dict[[HI](#src.integrator.utilities.HI), float]

### src.integrator.utilities.regional_annual_prices(m: [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel) | ConcreteModel, block=None) → dict[[HI](#src.integrator.utilities.HI), float]

pulls all regional annual weighted electricity prices

* **Parameters:**
  * **m** (*Union* *[* *'PowerModel'* *,* *ConcreteModel* *]*) – solved PowerModel
  * **block** (*optional*) – solved block model if applicable, by default None
* **Returns:**
  dict with regional annual electricity prices
* **Return type:**
  dict[[HI](#src.integrator.utilities.HI), float]

### src.integrator.utilities.select_solver(instance: ConcreteModel)

Select solver based on learning method

* **Parameters:**
  **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – electricity pyomo model
* **Returns:**
  The pyomo solver
* **Return type:**
  solver type (?)

### src.integrator.utilities.simple_solve(m: ConcreteModel)

a simple solve routine

### src.integrator.utilities.simple_solve_no_opt(m: ~pyomo.core.base.PyomoModel.ConcreteModel, opt: <pyomo.opt.base.solvers.SolverFactoryClass object at 0x000001D9F51290A0>)

Solve concrete model using solver factory object

* **Parameters:**
  * **m** (*ConcreteModel*) – Pyomo model
  * **opt** (*SolverFactory*) – Solver object initiated prior to solve

### src.integrator.utilities.update_elec_demand(self, elec_demand: dict[[HI](#src.integrator.utilities.HI), float]) → None

Update the external electical demand parameter with demands from the H2 model

* **Parameters:**
  **elec_demand** (*dict* *[*[*HI*](#src.integrator.utilities.HI) *,* *float* *]*) – the new demands broken out by hyd index (region, year)

### src.integrator.utilities.update_h2_prices(model: [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel), h2_prices: dict[[HI](#src.integrator.utilities.HI), float]) → None

Update the H2 prices held in the model

* **Parameters:**
  **h2_prices** (*list* *[**tuple* *[*[*HI*](#src.integrator.utilities.HI) *,* *float* *]* *]*) – new prices
