# src.models.electricity.scripts.utilities

This file is a collection of functions that are used in support of the electricity model.

### Functions

| [`annual_count`](#src.models.electricity.scripts.utilities.annual_count)(hour, m)                        | return the aggregate weight of this hour in the representative year we know the hour weight, and the hours are unique to days, so we can get the day weight                              |
|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`check_results`](#src.models.electricity.scripts.utilities.check_results)(results, SolutionStatus, ...) | Check results for termination condition and solution status                                                                                                                              |
| [`create_obj_df`](#src.models.electricity.scripts.utilities.create_obj_df)(mod_object)                   | takes pyomo component objects (e.g., variables, parameters, constraints) and processes the pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir. |

### Classes

| [`ElectricityMethods`](#src.models.electricity.scripts.utilities.ElectricityMethods)(\*args, \*\*kwds)   | a collection of functions used within the electricity model that aid in building the model.   |
|----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `Model`(\*args, \*\*kwds)                                                                                | This is the base model class for the models.                                                  |
| `Path`(\*args, \*\*kwargs)                                                                               | PurePath subclass that can make system calls.                                                 |
| `defaultdict`                                                                                            | defaultdict(default_factory=None, /, [...]) --> dict with default factory                     |

### *class* src.models.electricity.scripts.utilities.ElectricityMethods(\*args, \*\*kwds)

a collection of functions used within the electricity model that aid in building the model.

* **Parameters:**
  **Model** (*Class*) – generic model class

#### \_active

#### populate_RM_sets_rule()

Creates new reindexed sets for reserve margin constraint

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

#### populate_by_hour_sets_rule()

Creates new reindexed sets for dispatch_cost calculations

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

#### populate_demand_balance_sets_rule()

Creates new reindexed sets for demand balance constraint

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

#### populate_hydro_sets_rule()

Creates new reindexed sets for hydroelectric generation seasonal upper bound constraint

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

#### populate_reserves_sets_rule()

Creates new reindexed sets for operating reserves constraints

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

#### populate_trade_sets_rule()

Creates new reindexed sets for trade constraints

* **Parameters:**
  **m** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – pyomo electricity model instance

### src.models.electricity.scripts.utilities.annual_count(hour, m) → int

return the aggregate weight of this hour in the representative year
we know the hour weight, and the hours are unique to days, so we can
get the day weight

* **Parameters:**
  **hour** (*int*) – the rep_hour
* **Returns:**
  the aggregate weight (count) of this hour in the rep_year.  NOT the hour weight!
* **Return type:**
  int

### src.models.electricity.scripts.utilities.check_results(results, SolutionStatus, TerminationCondition)

Check results for termination condition and solution status

* **Parameters:**
  * **results** (*str*) – Results from pyomo
  * **SolutionStatus** (*str*) – Solution Status from pyomo
  * **TerminationCondition** (*str*) – Termination Condition from pyomo
* **Return type:**
  results

### src.models.electricity.scripts.utilities.create_obj_df(mod_object)

takes pyomo component objects (e.g., variables, parameters, constraints) and processes the
pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir.
The dataframe contains a key column which is the original way the pyomo data is structured,
as well as columns broken out for each set and the final values.

* **Parameters:**
  **mod_object** (*pyomo component object*) – pyomo component object
* **Returns:**
  contains the pyomo model results for the component object
* **Return type:**
  pd.DataFrame
