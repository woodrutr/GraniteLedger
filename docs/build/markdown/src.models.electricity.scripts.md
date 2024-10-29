# src.models.electricity.scripts package

## Subpackages

* [src.models.electricity.scripts.common package](src.models.electricity.scripts.common.md)
  * [Submodules](src.models.electricity.scripts.common.md#submodules)
  * [src.models.electricity.scripts.common.common module](src.models.electricity.scripts.common.md#module-src.models.electricity.scripts.common.common)
    * [`check_results()`](src.models.electricity.scripts.common.md#src.models.electricity.scripts.common.common.check_results)
  * [Module contents](src.models.electricity.scripts.common.md#module-src.models.electricity.scripts.common)

## Submodules

## src.models.electricity.scripts.electricity_model module

Electricity Model.
This file contains the PowerModel class which contains a pyomo optimization model of the electric
power sector. The class is organized by sections: settings, sets, parameters, variables, objective
function, constraints, plus additional misc support functions.

<!-- !! processed by numpydoc !! -->

### *class* src.models.electricity.scripts.electricity_model.PowerModel(\*args, \*\*kwds)

Bases: `ConcreteModel`

A PowerModel instance. Builds electricity pyomo model.

* **Parameters:**
  **all_frames**
  : Contains all dataframes of inputs

  **setA**
  : Contains all other non-dataframe inputs

<!-- !! processed by numpydoc !! -->

## src.models.electricity.scripts.postprocessor module

This file is the main postprocessor for the electricity model. It writes out all relevant model
outputs (e.g., variables, sets, parameters, constraints). It contains:

> - A function that converts pyomo component objects to dataframes
> - A function that writes the dataframes to output directories
> - A function to make the electricity output sub-directories
> - The postprocessor function, which loops through the model component objects and applies the

> functions to convert and write out the data to dfs to the electricity output sub-directories
<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.postprocessor.make_elec_output_dir()

generates an output directory to write model results, output directory is the date/time
at the time this function executes. It includes subdirs for vars, params, constraints.

* **Returns:**
  string
  : the name of the output directory

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.postprocessor.postprocessor(instance)

master postprocessor function that writes out the final dataframes from to the electricity
model. Creates the output directories and writes out dataframes for variables, parameters, and
constraints. Gets the correct columns names for each dataframe using the cols_dict.

* **Parameters:**
  **instance**
  : electricity concrete model
* **Returns:**
  string
  : output directory name

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.postprocessor.report_obj_df(mod_object, instance, dir_out, sub_dir)

Creates a df of the component object within the pyomo model, separates the key data into
different columns and then names the columns if the names are included in the cols_dict.
Writes the df out to the output directory.

* **Parameters:**
  **obj**
  : e.g., pyo.Var, pyo.Set, pyo.Param, pyo.Constraint

  **instance**
  : electricity concrete model

  **dir_out**
  : output electricity directory

  **sub_dir**
  : output electricity sub-directory

<!-- !! processed by numpydoc !! -->

## src.models.electricity.scripts.preprocessor module

This file is the main preprocessor for the electricity model. It established the parameters
and sets that will be used in the model. It contains:

> - A class that contains all sets used in the model
> - A collection of support functions to read in and setup parameter data
> - The preprocessor function, which produces an instance of the Set class and a dict of params
> - A collection of support functions to write out the inputs to the output directory
<!-- !! processed by numpydoc !! -->

### *class* src.models.electricity.scripts.preprocessor.Sets(settings)

Bases: `object`

Generates an initial batch of sets that are used to solve electricity model. Sets include:
- Scenario descriptor and model switches
- Regional sets
- Temporal sets
- Technology type sets
- Supply curve step sets
- Other

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.add_season_index(cw_temporal, df, pos)

adds a season index to the input dataframe

* **Parameters:**
  **cw_temporal**
  : dataframe that includes the season index

  **df**
  : parameter data to be modified

  **pos**
  : column position for the seasonal set
* **Returns:**
  dataframe
  : modified parameter data now indexed by season

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.avg_by_group(df, set_name, map_frame)

takes in a dataframe and groups it by the set specified and then averages the data.

* **Parameters:**
  **df**
  : parameter data to be modified

  **set_name**
  : name of the column/set to average the data by

  **map_frame**
  : data that maps the set name to the new grouping for that set
* **Returns:**
  dataframe
  : parameter data that is averaged by specified set mapping

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.fill_values(row, subset_list)

Function to fill in the subset values, is used to assign all years within the year
solve range to each year the model will solve for.

* **Parameters:**
  **row**
  : row number in df

  **subset_list**
  : list of values to map
* **Returns:**
  int
  : value from subset_list

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.load_data(tablename, metadata, engine)

loads the data from the SQL database; used in readin_sql function.

* **Parameters:**
  **tablename**
  : table name

  **metadata**
  : SQL metadata

  **engine**
  : SQL engine
* **Returns:**
  dataframe
  : table from SQL db as a dataframe

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.makedir(dir_out)

creates a folder directory based on the path provided

* **Parameters:**
  **dir_out**
  : path of directory

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.output_inputs()

function developed initial for QA purposes, writes out to csv all of the dfs and sets passed
to the electricity model to an output directory.

* **Returns:**
  **all_frames**
  : dictionary of dataframes where the key is the file name and the value is the table data

  **setin**
  : an initial batch of sets that are used to solve electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.preprocessor(setin)

master preprocessor function that generates the final dataframes and sets sent over to the
electricity model. This function reads in the input data, modifies it based on the temporal
and regional mapping specified in the inputs, and gets it into the final formatting needed.
Also adds some additional regional sets to the set class based on parameter inputs.

* **Parameters:**
  **setin**
  : an initial batch of sets that are used to solve electricity model
* **Returns:**
  **all_frames**
  : dictionary of dataframes where the key is the file name and the value is the table data

  **setin**
  : an initial batch of sets that are used to solve electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.print_sets(setin)

function developed initially for QA purposes, prints out all of the sets passed to the
electricity model.

* **Parameters:**
  **setin**
  : an initial batch of sets that are used to solve electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.readin_csvs(all_frames)

Reads in all of the CSV files from the input dir and returns a dictionary of dataframes,
where the key is the file name and the value is the table data.

* **Parameters:**
  **all_frames**
  : empty dictionary to be filled with dataframes
* **Returns:**
  dictionary
  : completed dictionary filled with dataframes from the input directory

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.readin_sql(all_frames)

Reads in all of the tables from a SQL databased and returns a dictionary of dataframes,
where the key is the table name and the value is the table data.

* **Parameters:**
  **all_frames**
  : empty dictionary to be filled with dataframes
* **Returns:**
  dictionary
  : completed dictionary filled with dataframes from the input directory

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.preprocessor.subset_dfs(all_frames, setin, i)

filters dataframes based on the values within the set

* **Parameters:**
  **all_frames**
  : dictionary of dataframes where the key is the file name and the value is the table data

  **setin**
  : contains an initial batch of sets that are used to solve electricity model

  **i**
  : name of the set contained within the sets class that the df will be filtered based on.
* **Returns:**
  dictionary
  : completed dictionary filled with dataframes filtered based on set inputs specified

<!-- !! processed by numpydoc !! -->

## src.models.electricity.scripts.runner module

This file is a collection of functions that are used to build, run, and solve the electricity model.

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.build_elec_model(all_frames, setin)

building pyomo electricity model

* **Parameters:**
  **all_frames**
  : input data frames

  **setin**
  : input settings Sets
* **Returns:**
  PowerModel
  : built (but unsolved) electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.cost_learning_func(instance, pt, y)

function for updating learning costs by technology and year

* **Parameters:**
  **instance**
  : electricity pyomo model

  **pt**
  : technology type

  **y**
  : year
* **Returns:**
  int
  : updated capital cost based on learning calculation

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.init_old_cap(instance)

initialize capacity for 0th iteration

* **Parameters:**
  **instance**
  : unsolved electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.run_elec_model(settings, solve=True)

build electricity model (and solve if solve=True) after passing in settings

* **Parameters:**
  **settings**
  : Configuration settings

  **solve**
  : solve electricity model?, by default True
* **Returns:**
  PowerModel
  : electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.set_new_cap(instance)

calculate new capacity after solve iteration

* **Parameters:**
  **instance**
  : solved electricity pyomo model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.solve_elec_model(instance)

solve electicity model

* **Parameters:**
  **instance**
  : built (but not solved) electricity pyomo model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.runner.update_cost(instance)

update capital cost based on new capacity learning

* **Parameters:**
  **instance**
  : electricity pyomo model

<!-- !! processed by numpydoc !! -->

## src.models.electricity.scripts.utilities module

This file is a collection of functions that are used in support of the electricity model.

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.annual_count(hour, m)

return the aggregate weight of this hour in the representative year
we know the hour weight, and the hours are unique to days, so we can
get the day weight

* **Parameters:**
  **hour**
  : the rep_hour
* **Returns:**
  int
  : the aggregate weight (count) of this hour in the rep_year.  NOT the hour weight!

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.create_obj_df(mod_object)

takes pyomo component objects (e.g., variables, parameters, constraints) and processes the
pyomo data and converts it to a dataframe and then writes the dataframe out to an output dir.
The dataframe contains a key column which is the original way the pyomo data is structured,
as well as columns broken out for each set and the final values.

* **Parameters:**
  **mod_object**
  : pyomo component object
* **Returns:**
  pd.DataFrame
  : contains the pyomo model results for the component object

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.declare_param(self, pname, p_set, data, default=0, mutable=False)

Assigns the df to be a pyomo parameter using the name specified.
Adds the name and index column names to the column dictionary used for post-processing.

* **Parameters:**
  **pname**
  : name of the parameter to be declared

  **p_set**
  : the pyomo set that cooresponds to the parameter data

  **data**
  : dataframe used generate the parameter

  **default**
  : by default 0

  **mutable**
  : by default False
* **Returns:**
  pyomo parameter
  : a pyomo parameter

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.declare_set(self, sname, df)

Assigns the index from the df to be a pyomo set using the name specified.
Adds the name and index column names to the column dictionary used for post-processing.

* **Parameters:**
  **sname**
  : name of the set to be declared

  **df**
  : dataframe from which the index will be grabbed to generate the set
* **Returns:**
  pyomo set
  : a pyomo set

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.declare_var(self, vname, v_set, bound=(0, 1000000000))

Assigns the set to be the index for the pyomo variable being declared.
Adds the name and index column names to the column dictionary used for post-processing.

* **Parameters:**
  **vname**
  : name of pyomo variable

  **v_set**
  : the pyomo set that the variable data will be indexed by

  **bound**
  : optional argument for setting variable bounds, default values set to zero to one billion
* **Returns:**
  pyomo variable
  : a pyomo variable

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_RM_sets_rule(m)

Creates new reindexed sets for reserve margin constraint

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_by_hour_sets_rule(m)

Creates new reindexed sets for dispatch_cost calculations

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_demand_balance_sets_rule(m)

Creates new reindexed sets for demand balance constraint

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_hydro_sets_rule(m)

Creates new reindexed sets for hydroelectric generation seasonal upper bound constraint

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_reserves_sets_rule(m)

Creates new reindexed sets for operating reserves constraints

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_sets_rule(m1, sname, set_base_name='', set_base2=[])

Generic function to create a new re-indexed set for a PowerModel instance which
should speed up build time. Must pass non-empty (either) set_base_name or set_base2

* **Parameters:**
  **m1**
  : electricity pyomo model instance

  **sname**
  : name of input pyomo set to base reindexing

  **set_base_name**
  : the name of the set to be the base of the reindexing, if left blank, uses set_base2, by default ‘’

  **set_base2**
  : the list of names of set columns to be the base of the reindexing, if left blank, should use set_base_name, by default []
* **Returns:**
  pyomo set
  : reindexed set to be added to electricity model

<!-- !! processed by numpydoc !! -->

### src.models.electricity.scripts.utilities.populate_trade_sets_rule(m)

Creates new reindexed sets for trade constraints

* **Parameters:**
  **m**
  : pyomo electricity model instance

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
