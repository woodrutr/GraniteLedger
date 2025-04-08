# src.models.residential.scripts.residential

Residential Model.
This file contains the residentialModule class which contains a representation of residential
electricity prices and demands.

### Functions

| `getLogger`([name])                                                                        | Return a logger with the specified name, creating it if necessary.                                                                                                                                                          |
|--------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`run_residential`](#src.models.residential.scripts.residential.run_residential)(settings) | This runs the residential model in stand-alone mode.                                                                                                                                                                        |
| `scale_load`(data_root)                                                                    | Reads in BaseLoad.csv (load for all regions/hours for first year) and LoadScalar.csv (a multiplier for all model years).                                                                                                    |
| `scale_load_with_enduses`(data_root)                                                       | Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv (the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier for all model years by enduse category). |

### Classes

| `Config_settings`(config_path[, args, test])                                                                    | Generates the model settings that are used to solve.              |
|-----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `Path`(\*args, \*\*kwargs)                                                                                      | PurePath subclass that can make system calls.                     |
| [`residentialModule`](#src.models.residential.scripts.residential.residentialModule)([settings, loadFile, ...]) | This contains the Residential model and its associated functions. |

### *class* src.models.residential.scripts.residential.residentialModule(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings) | None = None, loadFile: str | None = None, load_df: DataFrame | None = None, calibrate: bool | None = False)

This contains the Residential model and its associated functions. Once an object is
instantiated, it can calculate new Load values for updated prices. It can also calculate
estimated changes to the Load if one of the input variables is changed by a specified percent.
The model will be created in a symbolic form to be easily manipulated, and then values can be
filled in for calculations.

#### baseYear *= 0*

#### complex_step_sensitivity(prices, change_var, percent)

This estimates how much the output Load will change due to a change in one of the input
variables. It can calculate these values for changes in price, price elasticity, income,
income elasticity, or long term trend. The Load calculation requires input prices, so this
function requires that as well for the base output Load. Then, an estimate for Load is
calculated for the case where the named ‘change_var’ is changed by ‘percent’ %.

* **Parameters:**
  * **prices** (*dataframe* *or* *Pyomo Indexed Parameter*) – Price values used to calculate the Load value
  * **change_var** (*string*) – 

    Name of variable of interest for sensitivity. This can be:
    : ’income’, ‘i_elas’, ‘price’, ‘p_elas’, ‘trendGR’
  * **percent** (*float*) – A value 0 - 100 for the percent that the variable of interest can change.
* **Returns:**
  Indexed values for the calculated Load at the given prices, the Load if the variable
  of interest is increased by ‘percent’%, and the Load if the variable of interest is
  decreased by ‘percent’%
* **Return type:**
  dataframe

#### demandF(price, load, year, basePrice=1, p_elas=-0.1, baseYear=None, baseIncome=1, income=1, i_elas=1, trend=0, priceIndex=1, incomeIndex=1, p_lag=1, i_lag=1)

The demand function.  Wraps the sympy demand function with some defaults

* **Parameters:**
  * **price** ( *\_type_*) – \_description_
  * **load** ( *\_type_*) – \_description_
  * **year** ( *\_type_*) – \_description_
  * **basePrice** (*int* *,* *optional*) – \_description_, by default 1
  * **p_elas** (*float* *,* *optional*) – \_description_, by default -0.10
  * **baseYear** ( *\_type_* *,* *optional*) – \_description_, by default None
  * **baseIncome** (*int* *,* *optional*) – \_description_, by default 1
  * **income** (*int* *,* *optional*) – \_description_, by default 1
  * **i_elas** (*int* *,* *optional*) – \_description_, by default 1
  * **trend** (*int* *,* *optional*) – \_description_, by default 0
  * **priceIndex** (*int* *,* *optional*) – \_description_, by default 1
  * **incomeIndex** (*int* *,* *optional*) – \_description_, by default 1
  * **p_lag** (*int* *,* *optional*) – \_description_, by default 1
  * **i_lag** (*int* *,* *optional*) – \_description_, by default 1
* **Returns:**
  \_description_
* **Return type:**
  \_type_

#### hr_map *= Empty DataFrame Columns: [] Index: []*

#### loads *= {}*

#### make_block(prices, pricesindex)

Updates the value of ‘Load’ based on the new prices given.
The new prices are fed into the equations from the residential model.
The new calculated Loads are used to constrain ‘Load’ in pyomo blocks.

* **Parameters:**
  * **prices** (*pyo.Param*) – Pyomo Parameter of newly updated prices
  * **pricesindex** (*pyo.Set*) – Pyomo Set of indexes that matches the prices given
* **Returns:**
  Block containing constraints that set ‘Load’ variable equal to the updated load values
* **Return type:**
  pyo.Block

#### prices *= {}*

#### sensitivity(prices, change_var, percent)

This estimates how much the output Load will change due to a change in one of the input
variables. It can calculate these values for changes in price, price elasticity, income,
income elasticity, or long term trend. The Load calculation requires input prices, so this
function requires that as well for the base output Load. Then, an estimate for Load is
calculated for the case where the named ‘change_var’ is changed by ‘percent’ %.

* **Parameters:**
  * **prices** (*dataframe* *or* *Pyomo Indexed Parameter*) – Price values used to calculate the Load value
  * **change_var** (*string*) – 

    Name of variable of interest for sensitivity. This can be:
    : ’income’, ‘i_elas’, ‘price’, ‘p_elas’, ‘trendGR’
  * **percent** (*float*) – A value 0 - 100 for the percent that the variable of interest can change.
* **Returns:**
  Indexed values for the calculated Load at the given prices, the Load if the variable of
  interest is increased by ‘percent’%, and the Load if the variable of interest is
  decreased by ‘percent’%
* **Return type:**
  dataframe

#### update_load(p)

Takes in Dual pyomo Parameters or dataframes to update Load values

* **Parameters:**
  **p** (*pyo.Param*) – Pyomo Parameter or dataframe of newly updated prices from Duals
* **Returns:**
  Load values indexed by region, year, and hour
* **Return type:**
  pandas DataFrame

#### view_output_load(values: DataFrame, regions: list[int] = [1], years: list[int] = [2023])

This is used to display the updated Load values after calculation. It will create a
graph for each region and year combination.

* **Parameters:**
  * **values** (*pd.DataFrame*) – The Load values calculated in update_load
  * **regions** (*list* *[**int* *]* *,* *optional*) – The regions to be displayed
  * **years** (*list* *[**int* *]* *,* *optional*) – The years to be displayed

#### view_sensitivity(values: DataFrame, regions: list[int] = [1], years: list[int] = [2023])

This is used by the sensitivity method to display graphs of the calculated values

* **Parameters:**
  * **values** (*pd.DataFrame*) – indexed values for the Load, upper change, and lower change
  * **regions** (*list* *[**int* *]* *,* *optional*) – regions to be graphed
  * **years** (*list* *[**int* *]* *,* *optional*) – years to be graphed

### src.models.residential.scripts.residential.run_residential(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings))

This runs the residential model in stand-alone mode. It can run update_load to calculate new
Load values based on prices, or it can calculate the new Load value along with estimates for
the Load if one of the input variables changes.

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – information given from run_config to set several values
