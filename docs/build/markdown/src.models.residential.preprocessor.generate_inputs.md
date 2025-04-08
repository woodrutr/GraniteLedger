# src.models.residential.preprocessor.generate_inputs

This file contains the options to re-create the input files. It creates:
: - Load.csv: electricity demand for all model years (used in residential and electricity)
  - BaseElecPrice.csv: electricity prices for initial model year (used in residential only)

Uncomment out the functions at the end of this file in the “if \_\_name_\_ == ‘_\_main_\_’” statement
in order to generate new load or base electricity prices.

### Functions

| [`base_price`](#src.models.residential.preprocessor.generate_inputs.base_price)()   | Runs the electricity model with base price configuration settings and then merges the electricity prices and temporal crosswalk data produced from the run to generate base year electricity prices.   |
|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main`([settings])                                                                  | Runs model as defined in settings                                                                                                                                                                      |

### Classes

| `Path`(\*args, \*\*kwargs)   | PurePath subclass that can make system calls.   |
|------------------------------|-------------------------------------------------|

### src.models.residential.preprocessor.generate_inputs.base_price()

Runs the electricity model with base price configuration settings and then
merges the electricity prices and temporal crosswalk data produced from the run
to generate base year electricity prices.

* **Returns:**
  dataframe that contains base year electricity prices for all regions/hours
* **Return type:**
  pandas.core.frame.DataFrame
