# src.common.utilities

A gathering of utility functions for dealing with model interconnectivity

### Functions

| `getLogger`([name])                                                                   | Return a logger with the specified name, creating it if necessary.                                                                                                                                                          |
|---------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`get_args`](#src.common.utilities.get_args)()                                        | Parses args                                                                                                                                                                                                                 |
| [`make_dir`](#src.common.utilities.make_dir)(dir_name)                                | generates an output directory to write model results, output directory is the date/time at the time this function executes.                                                                                                 |
| [`scale_load`](#src.common.utilities.scale_load)(data_root)                           | Reads in BaseLoad.csv (load for all regions/hours for first year) and LoadScalar.csv (a multiplier for all model years).                                                                                                    |
| [`scale_load_with_enduses`](#src.common.utilities.scale_load_with_enduses)(data_root) | Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv (the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier for all model years by enduse category). |
| [`setup_logger`](#src.common.utilities.setup_logger)(settings)                        | initiates logging, sets up logger in the output directory specified                                                                                                                                                         |

### Classes

| `Path`(\*args, \*\*kwargs)   | PurePath subclass that can make system calls.   |
|------------------------------|-------------------------------------------------|

### src.common.utilities.get_args()

Parses args

* **Returns:**
  **args** – Contains arguments pass to main.py executable
* **Return type:**
  Namespace

### src.common.utilities.make_dir(dir_name)

generates an output directory to write model results, output directory is the date/time
at the time this function executes. It includes subdirs for vars, params, constraints.

* **Returns:**
  the name of the output directory
* **Return type:**
  string

### src.common.utilities.scale_load(data_root)

Reads in BaseLoad.csv (load for all regions/hours for first year)
and LoadScalar.csv (a multiplier for all model years). Merges the
data and multiplies the load by the scalar to generate new load
estimates for all model years.

* **Returns:**
  dataframe that contains load for all regions/years/hours
* **Return type:**
  pandas.core.frame.DataFrame

### src.common.utilities.scale_load_with_enduses(data_root)

Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv
(the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier
for all model years by enduse category). Merges the data and multiplies the load by the
adjusted enduse scalar and then sums up to new load estimates for all model years.

* **Returns:**
  dataframe that contains load for all regions/years/hours
* **Return type:**
  pandas.core.frame.DataFrame

### src.common.utilities.setup_logger(settings)

initiates logging, sets up logger in the output directory specified

* **Parameters:**
  **output_dir** (*path*) – output directory path
