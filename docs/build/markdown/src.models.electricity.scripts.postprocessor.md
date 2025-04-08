# src.models.electricity.scripts.postprocessor

This file is the main postprocessor for the electricity model.

It writes out all relevant model outputs (e.g., variables, parameters, constraints). It contains:
: - A function that converts pyomo component objects to dataframes
  - A function that writes the dataframes to output directories
  - A function to make the electricity output sub-directories
  - The postprocessor function, which loops through the model component objects and applies the
  <br/>
  functions to convert and write out the data to dfs to the electricity output sub-directories

### Functions

| `create_obj_df`(mod_object)                                                                                        | takes pyomo component objects (e.g., variables, parameters, constraints) and processes the pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir.   |
|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `getLogger`([name])                                                                                                | Return a logger with the specified name, creating it if necessary.                                                                                                                         |
| [`make_elec_output_dir`](#src.models.electricity.scripts.postprocessor.make_elec_output_dir)(output_dir)           | generates an output subdirectory to write electricity model results.                                                                                                                       |
| [`postprocessor`](#src.models.electricity.scripts.postprocessor.postprocessor)(instance)                           | master postprocessor function that writes out the final dataframes from to the electricity model.                                                                                          |
| [`report_obj_df`](#src.models.electricity.scripts.postprocessor.report_obj_df)(mod_object, instance, dir_out, ...) | Creates a df of the component object within the pyomo model, separates the key data into different columns and then names the columns if the names are included in the cols_dict.          |

### Classes

| `Path`(\*args, \*\*kwargs)   | PurePath subclass that can make system calls.   |
|------------------------------|-------------------------------------------------|

### src.models.electricity.scripts.postprocessor.make_elec_output_dir(output_dir)

generates an output subdirectory to write electricity model results. It includes subdirs for
vars, params, constraints.

* **Returns:**
  the name of the output directory
* **Return type:**
  string

### src.models.electricity.scripts.postprocessor.postprocessor(instance)

master postprocessor function that writes out the final dataframes from to the electricity
model. Creates the output directories and writes out dataframes for variables, parameters, and
constraints. Gets the correct columns names for each dataframe using the cols_dict.

* **Parameters:**
  **instance** (*pyomo model*) – electricity concrete model
* **Returns:**
  output directory name
* **Return type:**
  string

### src.models.electricity.scripts.postprocessor.report_obj_df(mod_object, instance, dir_out, sub_dir)

Creates a df of the component object within the pyomo model, separates the key data into
different columns and then names the columns if the names are included in the cols_dict.
Writes the df out to the output directory.

* **Parameters:**
  * **obj** (*pyomo component object*) – e.g., pyo.Var, pyo.Set, pyo.Param, pyo.Constraint
  * **instance** (*pyomo model*) – electricity concrete model
  * **dir_out** (*str*) – output electricity directory
  * **sub_dir** (*str*) – output electricity sub-directory
