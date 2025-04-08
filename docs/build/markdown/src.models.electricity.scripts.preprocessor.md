# src.models.electricity.scripts.preprocessor

This file is the main preprocessor for the electricity model.

It established the parameters and sets that will be used in the model. It contains:
: - A class that contains all sets used in the model
  - A collection of support functions to read in and setup parameter data
  - The preprocessor function, which produces an instance of the Set class and a dict of params
  - A collection of support functions to write out the inputs to the output directory

### Functions

| [`add_season_index`](#src.models.electricity.scripts.preprocessor.add_season_index)(cw_temporal, df, pos)            | adds a season index to the input dataframe                                                                                                                                                                                  |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`avg_by_group`](#src.models.electricity.scripts.preprocessor.avg_by_group)(df, set_name, map_frame)                 | takes in a dataframe and groups it by the set specified and then averages the data.                                                                                                                                         |
| [`capacitycredit_df`](#src.models.electricity.scripts.preprocessor.capacitycredit_df)(all_frames, setin)             | builds the capacity credit dataframe                                                                                                                                                                                        |
| [`create_hourly_params`](#src.models.electricity.scripts.preprocessor.create_hourly_params)(all_frames, key, cols)   | Expands params that are indexed by season to be indexed by hour                                                                                                                                                             |
| [`create_hourly_sets`](#src.models.electricity.scripts.preprocessor.create_hourly_sets)(all_frames, df)              | expands sets that are indexed by season to be indexed by hour                                                                                                                                                               |
| [`create_other_sets`](#src.models.electricity.scripts.preprocessor.create_other_sets)(all_frames, setin)             | creates other (non-supply curve) sets                                                                                                                                                                                       |
| [`create_sc_sets`](#src.models.electricity.scripts.preprocessor.create_sc_sets)(all_frames, setin)                   | creates supply curve sets                                                                                                                                                                                                   |
| [`create_subsets`](#src.models.electricity.scripts.preprocessor.create_subsets)(df, col, subset)                     | Create subsets off of full sets                                                                                                                                                                                             |
| [`fill_values`](#src.models.electricity.scripts.preprocessor.fill_values)(row, subset_list)                          | Function to fill in the subset values, is used to assign all years within the year solve range to each year the model will solve for.                                                                                       |
| [`hourly_sc_subset`](#src.models.electricity.scripts.preprocessor.hourly_sc_subset)(all_frames, subset)              | Creates sets/subsets that are related to the supply curve                                                                                                                                                                   |
| [`hr_sub_sc_subset`](#src.models.electricity.scripts.preprocessor.hr_sub_sc_subset)(all_frames, T_subset, hr_subset) | creates supply curve subsets by hour                                                                                                                                                                                        |
| [`load_data`](#src.models.electricity.scripts.preprocessor.load_data)(tablename, metadata, engine)                   | loads the data from the SQL database; used in readin_sql function.                                                                                                                                                          |
| [`makedir`](#src.models.electricity.scripts.preprocessor.makedir)(dir_out)                                           | creates a folder directory based on the path provided                                                                                                                                                                       |
| [`output_inputs`](#src.models.electricity.scripts.preprocessor.output_inputs)(OUTPUT_ROOT)                           | function developed initial for QA purposes, writes out to csv all of the dfs and sets passed to the electricity model to an output directory.                                                                               |
| [`preprocessor`](#src.models.electricity.scripts.preprocessor.preprocessor)(setin)                                   | main preprocessor function that generates the final dataframes and sets sent over to the electricity model.                                                                                                                 |
| [`print_sets`](#src.models.electricity.scripts.preprocessor.print_sets)(setin)                                       | function developed initially for QA purposes, prints out all of the sets passed to the electricity model.                                                                                                                   |
| [`readin_csvs`](#src.models.electricity.scripts.preprocessor.readin_csvs)(all_frames)                                | Reads in all of the CSV files from the input dir and returns a dictionary of dataframes, where the key is the file name and the value is the table data.                                                                    |
| [`readin_sql`](#src.models.electricity.scripts.preprocessor.readin_sql)(all_frames)                                  | Reads in all of the tables from a SQL databased and returns a dictionary of dataframes, where the key is the table name and the value is the table data.                                                                    |
| `scale_load`(data_root)                                                                                              | Reads in BaseLoad.csv (load for all regions/hours for first year) and LoadScalar.csv (a multiplier for all model years).                                                                                                    |
| `scale_load_with_enduses`(data_root)                                                                                 | Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv (the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier for all model years by enduse category). |
| [`step_sub_sc_subset`](#src.models.electricity.scripts.preprocessor.step_sub_sc_subset)(all_frames, T_subset, ...)   | creates supply curve subsets by step                                                                                                                                                                                        |
| [`subset_dfs`](#src.models.electricity.scripts.preprocessor.subset_dfs)(all_frames, setin, i)                        | filters dataframes based on the values within the set                                                                                                                                                                       |
| [`time_map`](#src.models.electricity.scripts.preprocessor.time_map)(cw_temporal, rename_cols)                        | create temporal mapping parameters                                                                                                                                                                                          |

### Classes

| `Path`(\*args, \*\*kwargs)                                            | PurePath subclass that can make system calls.                                |
|-----------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`Sets`](#src.models.electricity.scripts.preprocessor.Sets)(settings) | Generates an initial batch of sets that are used to solve electricity model. |

### *class* src.models.electricity.scripts.preprocessor.Sets(settings)

Generates an initial batch of sets that are used to solve electricity model. Sets include:

- Scenario descriptor and model switches
- Regional sets
- Temporal sets
- Technology type sets
- Supply curve step sets
- Other

### src.models.electricity.scripts.preprocessor.add_season_index(cw_temporal, df, pos)

adds a season index to the input dataframe

* **Parameters:**
  * **cw_temporal** (*dataframe*) – dataframe that includes the season index
  * **df** (*dataframe*) – parameter data to be modified
  * **pos** (*int*) – column position for the seasonal set
* **Returns:**
  modified parameter data now indexed by season
* **Return type:**
  dataframe

### src.models.electricity.scripts.preprocessor.avg_by_group(df, set_name, map_frame)

takes in a dataframe and groups it by the set specified and then averages the data.

* **Parameters:**
  * **df** (*dataframe*) – parameter data to be modified
  * **set_name** (*str*) – name of the column/set to average the data by
  * **map_frame** (*dataframe*) – data that maps the set name to the new grouping for that set
* **Returns:**
  parameter data that is averaged by specified set mapping
* **Return type:**
  dataframe

### src.models.electricity.scripts.preprocessor.capacitycredit_df(all_frames, setin)

builds the capacity credit dataframe

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – an initial batch of sets that are used to solve electricity model
* **Returns:**
  formatted capacity credit data frame
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.create_hourly_params(all_frames, key, cols)

Expands params that are indexed by season to be indexed by hour

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **key** (*str*) – name of data frame to access
  * **cols** (*list* *[**str* *]*) – column names to keep in data frame
* **Returns:**
  data frame with name key with new hourly index
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.create_hourly_sets(all_frames, df)

expands sets that are indexed by season to be indexed by hour

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **df** (*pd.DataFrame*) – data frame containing seasonal data
* **Returns:**
  data frame containing updated hourly set
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.create_other_sets(all_frames, setin)

creates other (non-supply curve) sets

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – an initial batch of sets that are used to solve electricity model
* **Returns:**
  updated Sets which has non-supply curve-related sets updated
* **Return type:**
  [Sets](#src.models.electricity.scripts.preprocessor.Sets)

### src.models.electricity.scripts.preprocessor.create_sc_sets(all_frames, setin)

creates supply curve sets

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – an initial batch of sets that are used to solve electricity model
* **Returns:**
  updated Set containing all sets related to supply curve
* **Return type:**
  [Sets](#src.models.electricity.scripts.preprocessor.Sets)

### src.models.electricity.scripts.preprocessor.create_subsets(df, col, subset)

Create subsets off of full sets

* **Parameters:**
  * **df** (*pd.DataFrame*) – data frame of full data
  * **col** (*str*) – column name
  * **subset** (*list* *[**str* *]*) – names of values to subset
* **Returns:**
  data frame containing subset of full data
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.fill_values(row, subset_list)

Function to fill in the subset values, is used to assign all years within the year
solve range to each year the model will solve for.

* **Parameters:**
  * **row** (*int*) – row number in df
  * **subset_list** (*list*) – list of values to map
* **Returns:**
  value from subset_list
* **Return type:**
  int

### src.models.electricity.scripts.preprocessor.hourly_sc_subset(all_frames, subset)

Creates sets/subsets that are related to the supply curve

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **subset** (*list*) – list of technologies to subset
* **Returns:**
  data frame containing sets/subsets related to supply curve
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.hr_sub_sc_subset(all_frames, T_subset, hr_subset)

creates supply curve subsets by hour

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **T_subset** (*list*) – list of technologies to subset
  * **hr_subset** (*list*) – list of hours to subset
* **Returns:**
  data frame containing supply curve related hourly subset
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.load_data(tablename, metadata, engine)

loads the data from the SQL database; used in readin_sql function.

* **Parameters:**
  * **tablename** (*string*) – table name
  * **metadata** (*SQL metadata*) – SQL metadata
  * **engine** (*SQL engine*) – SQL engine
* **Returns:**
  table from SQL db as a dataframe
* **Return type:**
  dataframe

### src.models.electricity.scripts.preprocessor.makedir(dir_out)

creates a folder directory based on the path provided

* **Parameters:**
  **dir_out** (*str*) – path of directory

### src.models.electricity.scripts.preprocessor.output_inputs(OUTPUT_ROOT)

function developed initial for QA purposes, writes out to csv all of the dfs and sets passed
to the electricity model to an output directory.

* **Parameters:**
  **OUTPUT_ROOT** (*str*) – path of output directory
* **Returns:**
  * **all_frames** (*dictionary*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** (*Sets*) – an initial batch of sets that are used to solve electricity model

### src.models.electricity.scripts.preprocessor.preprocessor(setin)

main preprocessor function that generates the final dataframes and sets sent over to the
electricity model. This function reads in the input data, modifies it based on the temporal
and regional mapping specified in the inputs, and gets it into the final formatting needed.
Also adds some additional regional sets to the set class based on parameter inputs.

* **Parameters:**
  **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – an initial batch of sets that are used to solve electricity model
* **Returns:**
  * **all_frames** (*dictionary*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** (*Sets*) – an initial batch of sets that are used to solve electricity model

### src.models.electricity.scripts.preprocessor.print_sets(setin)

function developed initially for QA purposes, prints out all of the sets passed to the
electricity model.

* **Parameters:**
  **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – an initial batch of sets that are used to solve electricity model

### src.models.electricity.scripts.preprocessor.readin_csvs(all_frames)

Reads in all of the CSV files from the input dir and returns a dictionary of dataframes,
where the key is the file name and the value is the table data.

* **Parameters:**
  **all_frames** (*dictionary*) – empty dictionary to be filled with dataframes
* **Returns:**
  completed dictionary filled with dataframes from the input directory
* **Return type:**
  dictionary

### src.models.electricity.scripts.preprocessor.readin_sql(all_frames)

Reads in all of the tables from a SQL databased and returns a dictionary of dataframes,
where the key is the table name and the value is the table data.

* **Parameters:**
  **all_frames** (*dictionary*) – empty dictionary to be filled with dataframes
* **Returns:**
  completed dictionary filled with dataframes from the input directory
* **Return type:**
  dictionary

### src.models.electricity.scripts.preprocessor.step_sub_sc_subset(all_frames, T_subset, step_subset)

creates supply curve subsets by step

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **T_subset** (*list*) – technologies to subset
  * **step_subset** (*list*) – step numbers to subset
* **Returns:**
  data frame containing supply curve subsets by step
* **Return type:**
  pd.DataFrame

### src.models.electricity.scripts.preprocessor.subset_dfs(all_frames, setin, i)

filters dataframes based on the values within the set

* **Parameters:**
  * **all_frames** (*dictionary*) – dictionary of dataframes where the key is the file name and the value is the table data
  * **setin** ([*Sets*](#src.models.electricity.scripts.preprocessor.Sets)) – contains an initial batch of sets that are used to solve electricity model
  * **i** (*string*) – name of the set contained within the sets class that the df will be filtered based on.
* **Returns:**
  completed dictionary filled with dataframes filtered based on set inputs specified
* **Return type:**
  dictionary

### src.models.electricity.scripts.preprocessor.time_map(cw_temporal, rename_cols)

create temporal mapping parameters

* **Parameters:**
  * **cw_temporal** (*pd.DataFrame*) – temporal crosswalks
  * **rename_cols** (*dict*) – columns to rename from/to
* **Returns:**
  data frame with temporal mapping parameters
* **Return type:**
  pd.DataFrame
