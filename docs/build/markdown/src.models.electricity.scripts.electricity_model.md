# src.models.electricity.scripts.electricity_model

Electricity Model, a pyomo optimization model of the electric power sector.

The class is organized by sections: settings, sets, parameters, variables, objective function,
constraints, plus additional misc support functions.

### Functions

| `getLogger`([name])   | Return a logger with the specified name, creating it if necessary.   |
|-----------------------|----------------------------------------------------------------------|

### Classes

| `HI`(region, year)                                                                             | (region, year)                                                                                                                           |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `Model`(\*args, \*\*kwds)                                                                      | This is the base model class for the models.                                                                                             |
| [`PowerModel`](#src.models.electricity.scripts.electricity_model.PowerModel)(\*args, \*\*kwds) | A PowerModel instance.                                                                                                                   |
| `defaultdict`                                                                                  | defaultdict(default_factory=None, /, [...]) --> dict with default factory                                                                |
| `em`                                                                                           | alias of [`ElectricityMethods`](src.models.electricity.scripts.utilities.md#src.models.electricity.scripts.utilities.ElectricityMethods) |

### *class* src.models.electricity.scripts.electricity_model.PowerModel(\*args, \*\*kwds)

A PowerModel instance. Builds electricity pyomo model.

* **Parameters:**
  * **all_frames** (*dictionary* *of* *pd.DataFrames*) – Contains all dataframes of inputs
  * **setA** ([*Sets*](src.models.electricity.scripts.preprocessor.md#src.models.electricity.scripts.preprocessor.Sets)) – Contains all other non-dataframe inputs

#### \_active
