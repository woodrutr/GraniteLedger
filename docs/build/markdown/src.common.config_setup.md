# src.common.config_setup

This file contains Config_settings class. It establishes the main settings used when running
the model. It takes these settings from the run_config.toml file. It has universal configurations
(e.g., configs that cut across modules and/or solve options) and module specific configs.

### Functions

| `create_temporal_mapping`(sw_temporal)   | Combines the input mapping files within the electricity model to create a master temporal mapping dataframe.                |
|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `make_dir`(dir_name)                     | generates an output directory to write model results, output directory is the date/time at the time this function executes. |

### Classes

| [`Config_settings`](#src.common.config_setup.Config_settings)(config_path[, args, test])   | Generates the model settings that are used to solve.   |
|--------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `Path`(\*args, \*\*kwargs)                                                                 | PurePath subclass that can make system calls.          |
| `datetime`(year, month, day[, hour[, minute[, ...)                                         | The year, month and day arguments are required.        |

### *class* src.common.config_setup.Config_settings(config_path: Path, args: Namespace | None = None, test=False)

Generates the model settings that are used to solve. Settings include:

- Iterative Solve Config Settings
- Spatial Config Settings
- Temporal Config Settings
- Electricity Config Settings
- Other

#### \_additional_year_settings(name, value)

Checks year related settings to see if values are within expected ranges and updates
other settings linked to years if years is changed.

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_elec_expansion_settings(name, value)

Checks that switches for reserve margin and learning are on only if expansion is on.

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_int(name, value)

Checks if attribute is an integer

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_regions(name, value)

Checks to see if region is between the current default values of 1 and 25.

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_res_settings(name, value)

Checks if view year or region settings are subsets of year or region

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_true_false(name, value)

Checks if attribute is either true or false

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_check_zero_one(name, value)

Checks if attribute is either zero or one

* **Parameters:**
  * **name** (*str*) – attribute name
  * **value** ( *\_type_*) – attribute value
* **Raises:**
  **TypeError** – Error

#### \_has_all_attributes(attrs: set)

Determines if all attributes within the set exist or not

* **Parameters:**
  **attrs** (*set*) – set of setting attributes
* **Returns:**
  True or False
* **Return type:**
  bool
